#!/usr/bin/env python3
"""
Weather -> Kalshi Prediction Market Trading Dashboard (single file)
====================================================================

Pipeline:
  NWS hourly forecast -> Normal(mu, sigma) model -> bracket probabilities
  -> match brackets to live Kalshi markets -> compare model vs. market-implied
  probability -> (optionally) place limit orders on Kalshi's DEMO API
  -> serve a live HTML dashboard showing all of it.

SAFETY:
  - DRY_RUN=True by default: orders are computed and logged, never sent.
  - ENABLE_TRADING=False by default: even with DRY_RUN off, no order fires
    until this is explicitly set True.
  - Defaults to Kalshi's DEMO host (paper money), not production.

Dependencies: requests, cryptography
  pip install requests cryptography
"""

from __future__ import annotations

import base64
import json
import logging
import math
import os
import re
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, date
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Optional
from urllib.parse import urlparse, parse_qs

import requests
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("weather_kalshi_bot")


# =============================================================================
# CONFIG
# =============================================================================

@dataclass
class Config:
    # --- NWS ---
    nws_lat: float = 41.9742
    nws_lon: float = -87.9073
    city_name: str = "Chicago, IL"
    nws_user_agent: str = "MyWeatherBot/1.0 (contact@example.com)"

    # --- Model ---
    sigma_f: float = 1.5
    bracket_width: int = 2
    brackets_each_side: int = 4

    # --- Kalshi ---
    # Demo (paper-money) host. Do not point this at the production host
    # until you've validated behavior end-to-end.
    kalshi_base_url: str = "https://demo-api.kalshi.co/trade-api/v2"
    kalshi_key_id: str = os.environ.get("KALSHI_KEY_ID", "")
    kalshi_private_key_path: str = os.environ.get(
        "KALSHI_PRIVATE_KEY_PATH", "kalshi_private_key.pem"
    )
    # Verify this against the actual series ticker for your city on kalshi.com
    # (weather series tickers vary, e.g. KXHIGHNY for NYC). Adjust as needed.
    kalshi_series_ticker: str = os.environ.get("KALSHI_SERIES_TICKER", "KXHIGHCHI")

    # --- Trading logic ---
    edge_threshold: float = 0.05     # min |model_prob - market_prob| to act on
    max_order_contracts: int = 1     # conservative per-trade size, demo money
    poll_interval_sec: int = 120

    # --- Safety switches ---
    dry_run: bool = True             # True: compute + log orders, never send
    enable_trading: bool = False     # True required (in addition to dry_run=False) to send

    # --- Dashboard ---
   dashboard_host: str = "0.0.0.0"
   dashboard_port: int = int(os.environ.get("PORT", 8787))


CONFIG = Config()


# =============================================================================
# SHARED STATE (thread-safe, read by dashboard, written by the poll loop)
# =============================================================================

STATE_LOCK = threading.Lock()
STATE: dict[str, Any] = {
    "last_updated": None,
    "city": CONFIG.city_name,
    "projected_high_f": None,
    "brackets": [],       # list of bracket dicts, see build_bracket_rows()
    "balance_dollars": None,
    "positions": [],
    "recent_orders": [],
    "errors": [],
}


def update_state(**kwargs: Any) -> None:
    with STATE_LOCK:
        STATE.update(kwargs)
        STATE["last_updated"] = datetime.utcnow().isoformat() + "Z"


def push_error(msg: str) -> None:
    logger.error(msg)
    with STATE_LOCK:
        errs = STATE["errors"]
        errs.append({"time": datetime.utcnow().isoformat() + "Z", "message": msg})
        STATE["errors"] = errs[-20:]  # keep last 20


def push_order_log(entry: dict[str, Any]) -> None:
    with STATE_LOCK:
        orders = STATE["recent_orders"]
        orders.append(entry)
        STATE["recent_orders"] = orders[-30:]


# =============================================================================
# NWS: hourly forecast -> today's projected high
# =============================================================================

def get_forecast_hourly_url(lat: float, lon: float) -> str:
    url = f"https://api.weather.gov/points/{lat},{lon}"
    resp = requests.get(
        url,
        headers={"User-Agent": CONFIG.nws_user_agent, "Accept": "application/geo+json"},
        timeout=10,
    )
    resp.raise_for_status()
    data = resp.json()
    try:
        return data["properties"]["forecastHourly"]
    except KeyError as e:
        raise KeyError(f"Unexpected /points response shape, missing key: {e}") from e


def get_todays_projected_high(forecast_hourly_url: str) -> float:
    resp = requests.get(
        forecast_hourly_url,
        headers={"User-Agent": CONFIG.nws_user_agent, "Accept": "application/geo+json"},
        timeout=10,
    )
    resp.raise_for_status()
    data = resp.json()
    try:
        periods = data["properties"]["periods"]
    except KeyError as e:
        raise KeyError(f"Unexpected forecastHourly response shape, missing key: {e}") from e

    if not periods:
        raise ValueError("No hourly periods returned by NWS.")

    today = datetime.fromisoformat(periods[0]["startTime"]).date()
    todays_temps = [
        float(p["temperature"])
        for p in periods
        if datetime.fromisoformat(p["startTime"]).date() == today
    ]
    if not todays_temps:
        raise ValueError(f"No forecast hours found for {today}.")
    return max(todays_temps)


# =============================================================================
# Probability model
# =============================================================================

def normal_cdf(x: float, mu: float, sigma: float) -> float:
    return 0.5 * (1 + math.erf((x - mu) / (sigma * math.sqrt(2))))


def bracket_probability(mu: float, sigma: float, low: float, high: float) -> float:
    return normal_cdf(high, mu, sigma) - normal_cdf(low, mu, sigma)


def build_brackets(mu: float, width: int, n_each_side: int) -> list[tuple[int, int]]:
    center_low = (round(mu) // width) * width
    return [
        (center_low + i * width, center_low + i * width + (width - 1))
        for i in range(-n_each_side, n_each_side + 1)
    ]


# =============================================================================
# Kalshi client: RSA-PSS-signed requests against the demo API
# =============================================================================

class KalshiClient:
    """
    Minimal Kalshi v2 REST client.

    Auth: RSA-PSS signature over `timestamp_ms + METHOD + path`
    (path includes /trade-api/v2 prefix, excludes query string), SHA-256,
    MGF1(SHA-256), salt length = digest length. Three headers are attached
    to every request; public endpoints accept-but-don't-require them,
    /portfolio/* endpoints require them.
    """

    def __init__(self, base_url: str, key_id: str, private_key_path: str):
        self.base_url = base_url.rstrip("/")
        self.key_id = key_id
        self._private_key = None
        if key_id and os.path.exists(private_key_path):
            with open(private_key_path, "rb") as f:
                self._private_key = serialization.load_pem_private_key(
                    f.read(), password=None
                )
        else:
            logger.warning(
                "No Kalshi key configured (KALSHI_KEY_ID / KALSHI_PRIVATE_KEY_PATH). "
                "Public endpoints will still work; authenticated calls will fail."
            )
        self.session = requests.Session()

    def _sign(self, method: str, path_no_query: str) -> dict[str, str]:
        if self._private_key is None:
            return {}
        timestamp_ms = str(int(time.time() * 1000))
        message = (timestamp_ms + method.upper() + path_no_query).encode("utf-8")
        signature = self._private_key.sign(
            message,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.DIGEST_LENGTH,
            ),
            hashes.SHA256(),
        )
        return {
            "KALSHI-ACCESS-KEY": self.key_id,
            "KALSHI-ACCESS-TIMESTAMP": timestamp_ms,
            "KALSHI-ACCESS-SIGNATURE": base64.b64encode(signature).decode("utf-8"),
        }

    def _request(self, method: str, path: str, **kwargs: Any) -> requests.Response:
        full_url = self.base_url + path
        path_no_query = urlparse(full_url).path
        headers = self._sign(method, path_no_query)
        resp = self.session.request(method, full_url, headers=headers, timeout=15, **kwargs)
        resp.raise_for_status()
        return resp

    # --- Public discovery ---

    def get_markets(self, series_ticker: str, status: str = "open") -> list[dict]:
        markets: list[dict] = []
        cursor: Optional[str] = None
        while True:
            params = {"series_ticker": series_ticker, "status": status, "limit": 100}
            if cursor:
                params["cursor"] = cursor
            resp = self._request("GET", "/markets", params=params)
            data = resp.json()
            markets.extend(data.get("markets", []))
            cursor = data.get("cursor")
            if not cursor:
                break
        return markets

    def get_orderbook(self, ticker: str) -> dict:
        resp = self._request("GET", f"/markets/{ticker}/orderbook")
        return resp.json()

    # --- Authenticated portfolio ---

    def get_balance(self) -> Optional[float]:
        try:
            resp = self._request("GET", "/portfolio/balance")
            data = resp.json()
            # Historically an integer-cents field; tolerate either shape.
            if "balance" in data:
                return data["balance"] / 100.0
            if "balance_dollars" in data:
                return float(data["balance_dollars"])
        except Exception as e:  # noqa: BLE001
            push_error(f"get_balance failed: {e}")
        return None

    def get_positions(self) -> list[dict]:
        try:
            resp = self._request("GET", "/portfolio/positions")
            return resp.json().get("market_positions", [])
        except Exception as e:  # noqa: BLE001
            push_error(f"get_positions failed: {e}")
            return []

    def create_order(
        self,
        ticker: str,
        side: str,          # "yes" or "no"
        action: str,         # "buy" or "sell"
        price_dollars: float,
        count: int,
    ) -> dict:
        """
        NOTE: this payload shape is a best-effort construction based on
        Kalshi's documented fixed-point conventions (string dollar prices,
        string contract counts). Verify exact field names against
        docs.kalshi.com's order-creation reference before relying on this
        in anything beyond dry-run / demo testing.
        """
        body = {
            "ticker": ticker,
            "client_order_id": str(uuid.uuid4()),
            "action": action,
            "side": side,
            "type": "limit",
            "yes_price_dollars" if side == "yes" else "no_price_dollars": f"{price_dollars:.2f}",
            "count_fp": f"{count:.2f}",
            "time_in_force": "good_till_cancelled",
        }
        resp = self._request("POST", "/portfolio/events/orders", json=body)
        return resp.json()


kalshi = KalshiClient(
    CONFIG.kalshi_base_url, CONFIG.kalshi_key_id, CONFIG.kalshi_private_key_path
)


# =============================================================================
# Bracket <-> Kalshi market matching
# =============================================================================

_RANGE_RE = re.compile(r"(\d{2,3})\s*(?:°|degrees)?\s*(?:-|to)\s*(\d{2,3})\s*(?:°|degrees)?")


def extract_bracket_from_market(market: dict) -> Optional[tuple[int, int]]:
    """
    Try to recover the (low, high) integer-degree bracket a Kalshi weather
    market represents. Kalshi's schema for this varies by market family;
    this checks a few plausible field names, then falls back to regex on
    the human-readable subtitle. If your market payloads use different
    field names, adjust this function first (it's the single place bracket
    matching happens) -- the dashboard's "unmatched markets" panel is meant
    to help you see raw payloads for that purpose.
    """
    floor_strike = market.get("floor_strike")
    cap_strike = market.get("cap_strike")
    if floor_strike is not None and cap_strike is not None:
        try:
            return int(round(float(floor_strike))), int(round(float(cap_strike)))
        except (TypeError, ValueError):
            pass

    for field_name in ("yes_sub_title", "subtitle", "title"):
        text = market.get(field_name, "")
        m = _RANGE_RE.search(text)
        if m:
            return int(m.group(1)), int(m.group(2))

    return None


def best_yes_bid_ask(orderbook: dict) -> Optional[tuple[float, float]]:
    """Derive (best_yes_bid, best_yes_ask) from Kalshi's bids-only orderbook."""
    book = orderbook.get("orderbook_fp") or orderbook.get("orderbook")
    if not book:
        return None
    yes_side = book.get("yes_dollars") or book.get("yes")
    no_side = book.get("no_dollars") or book.get("no")
    if not yes_side or not no_side:
        return None
    try:
        best_yes_bid = float(yes_side[-1][0])
        best_no_bid = float(no_side[-1][0])
    except (IndexError, ValueError, TypeError):
        return None
    best_yes_ask = 1.0 - best_no_bid
    return best_yes_bid, best_yes_ask


# =============================================================================
# Core cycle: forecast -> model -> match -> decide -> (maybe) trade
# =============================================================================

def build_bracket_rows(mu: float, markets: list[dict]) -> list[dict]:
    brackets = build_brackets(mu, CONFIG.bracket_width, CONFIG.brackets_each_side)
    market_by_range: dict[tuple[int, int], dict] = {}
    for m in markets:
        rng = extract_bracket_from_market(m)
        if rng:
            market_by_range[rng] = m

    rows = []
    for low, high in brackets:
        model_prob = bracket_probability(mu, CONFIG.sigma_f, low - 0.5, high + 0.5)
        market = market_by_range.get((low, high))
        row = {
            "low": low,
            "high": high,
            "model_prob": model_prob,
            "ticker": market.get("ticker") if market else None,
            "market_prob": None,
            "edge": None,
            "action": "no market match" if not market else "-",
        }

        if market:
            try:
                orderbook = kalshi.get_orderbook(market["ticker"])
                quote = best_yes_bid_ask(orderbook)
            except Exception as e:  # noqa: BLE001
                push_error(f"orderbook fetch failed for {market.get('ticker')}: {e}")
                quote = None

            if quote:
                yes_bid, yes_ask = quote
                market_prob = (yes_bid + yes_ask) / 2
                edge = model_prob - market_prob
                row["market_prob"] = market_prob
                row["edge"] = edge

                if abs(edge) >= CONFIG.edge_threshold:
                    side = "yes" if edge > 0 else "no"
                    action = "buy"
                    limit_price = yes_ask if side == "yes" else round(1 - yes_bid, 2)
                    row["action"] = f"EDGE: buy {side.upper()} @ ~{limit_price:.2f}"
                    place_order_if_enabled(
                        ticker=market["ticker"],
                        side=side,
                        action=action,
                        price_dollars=limit_price,
                        count=CONFIG.max_order_contracts,
                        reason=f"model={model_prob:.3f} market={market_prob:.3f} edge={edge:+.3f}",
                    )
                else:
                    row["action"] = "within edge threshold, no trade"

        rows.append(row)
    return rows


def place_order_if_enabled(
    ticker: str, side: str, action: str, price_dollars: float, count: int, reason: str
) -> None:
    log_entry = {
        "time": datetime.utcnow().isoformat() + "Z",
        "ticker": ticker,
        "side": side,
        "action": action,
        "price": price_dollars,
        "count": count,
        "reason": reason,
        "dry_run": CONFIG.dry_run,
        "sent": False,
        "response": None,
    }

    if CONFIG.dry_run or not CONFIG.enable_trading:
        logger.info(f"[DRY RUN / trading disabled] Would place order: {log_entry}")
        push_order_log(log_entry)
        return

    try:
        resp = kalshi.create_order(
            ticker=ticker, side=side, action=action,
            price_dollars=price_dollars, count=count,
        )
        log_entry["sent"] = True
        log_entry["response"] = resp
        logger.info(f"Order placed: {resp}")
    except Exception as e:  # noqa: BLE001
        push_error(f"create_order failed for {ticker}: {e}")
        log_entry["response"] = {"error": str(e)}

    push_order_log(log_entry)


def run_cycle() -> None:
    try:
        forecast_url = get_forecast_hourly_url(CONFIG.nws_lat, CONFIG.nws_lon)
        mu = get_todays_projected_high(forecast_url)

        markets = kalshi.get_markets(CONFIG.kalshi_series_ticker)
        rows = build_bracket_rows(mu, markets)

        balance = kalshi.get_balance()
        positions = kalshi.get_positions()

        update_state(
            projected_high_f=mu,
            brackets=rows,
            balance_dollars=balance,
            positions=positions,
        )
        logger.info(f"Cycle complete. Projected high={mu:.1f}F, {len(rows)} brackets evaluated.")
    except Exception as e:  # noqa: BLE001
        push_error(f"run_cycle failed: {e}")


def poll_loop() -> None:
    while True:
        run_cycle()
        time.sleep(CONFIG.poll_interval_sec)


# =============================================================================
# Dashboard (stdlib http.server, no extra dependencies)
# =============================================================================

DASHBOARD_HTML = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Weather x Kalshi Dashboard</title>
<style>
  body { font-family: -apple-system, Segoe UI, Roboto, sans-serif; background: #0f1117; color: #e6e6e6; margin: 0; padding: 24px; }
  h1 { font-size: 20px; margin-bottom: 4px; }
  .sub { color: #8a8f98; font-size: 13px; margin-bottom: 20px; }
  .cards { display: flex; gap: 16px; margin-bottom: 20px; flex-wrap: wrap; }
  .card { background: #171a21; border: 1px solid #262b36; border-radius: 8px; padding: 14px 18px; min-width: 160px; }
  .card .label { font-size: 12px; color: #8a8f98; }
  .card .value { font-size: 22px; font-weight: 600; margin-top: 4px; }
  table { width: 100%; border-collapse: collapse; font-size: 13px; }
  th, td { padding: 8px 10px; text-align: right; border-bottom: 1px solid #262b36; }
  th:first-child, td:first-child { text-align: left; }
  th { color: #8a8f98; font-weight: 500; }
  .pos { color: #4ade80; } .neg { color: #f87171; }
  .badge { padding: 2px 8px; border-radius: 4px; font-size: 12px; }
  .badge.trade { background: #1e3a8a; color: #93c5fd; }
  .badge.none { background: #262b36; color: #8a8f98; }
  .footer { margin-top: 20px; font-size: 12px; color: #8a8f98; }
  .errors { margin-top: 20px; }
  .errors .err { color: #f87171; font-size: 12px; margin-bottom: 4px; }
  .safety { display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 12px; margin-left: 8px; }
  .safety.on { background: #3f2b1d; color: #fbbf24; }
  .safety.off { background: #1a2e1a; color: #86efac; }
</style>
</head>
<body>
  <h1>Weather &rarr; Kalshi Prediction Market Dashboard</h1>
  <div class="sub" id="subheader">Loading...</div>
  <div class="cards" id="cards"></div>
  <table id="bracketTable">
    <thead>
      <tr><th>Bracket</th><th>Model Prob</th><th>Market Prob</th><th>Edge</th><th>Ticker</th><th>Status</th></tr>
    </thead>
    <tbody id="bracketBody"></tbody>
  </table>
  <div class="errors" id="errors"></div>
  <div class="footer">Auto-refreshes every 5s. Demo/paper trading only unless explicitly reconfigured.</div>

<script>
async function refresh() {
  const res = await fetch('/api/state');
  const s = await res.json();

  document.getElementById('subheader').innerHTML =
    `${s.city} &middot; last updated: ${s.last_updated ?? 'never'}`;

  const cards = document.getElementById('cards');
  cards.innerHTML = `
    <div class="card"><div class="label">Projected High</div><div class="value">${s.projected_high_f != null ? s.projected_high_f.toFixed(1) + '&deg;F' : '-'}</div></div>
    <div class="card"><div class="label">Demo Balance</div><div class="value">${s.balance_dollars != null ? '$' + s.balance_dollars.toFixed(2) : '-'}</div></div>
    <div class="card"><div class="label">Open Positions</div><div class="value">${s.positions.length}</div></div>
  `;

  const body = document.getElementById('bracketBody');
  body.innerHTML = s.brackets.map(b => {
    const edgeClass = b.edge == null ? '' : (b.edge > 0 ? 'pos' : 'neg');
    const badge = (b.action && b.action.startsWith('EDGE')) ? 'trade' : 'none';
    return `<tr>
      <td>${b.low}&deg;F-${b.high}&deg;F</td>
      <td>${(b.model_prob*100).toFixed(2)}%</td>
      <td>${b.market_prob != null ? (b.market_prob*100).toFixed(2) + '%' : '-'}</td>
      <td class="${edgeClass}">${b.edge != null ? (b.edge*100).toFixed(2) + '%' : '-'}</td>
      <td>${b.ticker ?? '-'}</td>
      <td><span class="badge ${badge}">${b.action}</span></td>
    </tr>`;
  }).join('');

  const errDiv = document.getElementById('errors');
  errDiv.innerHTML = s.errors.slice(-5).map(e => `<div class="err">${e.time}: ${e.message}</div>`).join('');
}
refresh();
setInterval(refresh, 5000);
</script>
</body>
</html>
"""


class DashboardHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args: Any) -> None:
        pass  # keep console clean; use logger instead

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self._send(200, DASHBOARD_HTML.encode("utf-8"), "text/html")
        elif parsed.path == "/api/state":
            with STATE_LOCK:
                payload = json.dumps(STATE).encode("utf-8")
            self._send(200, payload, "application/json")
        else:
            self._send(404, b"not found", "text/plain")

    def _send(self, code: int, body: bytes, content_type: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main() -> None:
    logger.info(
        f"Starting. DRY_RUN={CONFIG.dry_run} ENABLE_TRADING={CONFIG.enable_trading} "
        f"host={CONFIG.kalshi_base_url}"
    )
    threading.Thread(target=poll_loop, daemon=True).start()

    server = ThreadingHTTPServer((CONFIG.dashboard_host, CONFIG.dashboard_port), DashboardHandler)
    logger.info(f"Dashboard live at http://{CONFIG.dashboard_host}:{CONFIG.dashboard_port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Shutting down.")
        server.shutdown()


if __name__ == "__main__":
    main()
