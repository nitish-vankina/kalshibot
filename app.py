#!/usr/bin/env python3
"""
Multi-City Weather -> Kalshi Prediction Market Trading Dashboard
==================================================================
pip install requests cryptography

Render deploy: binds 0.0.0.0 + $PORT automatically. Set env vars:
  KALSHI_KEY_ID, KALSHI_PRIVATE_KEY_PATH (point at a Render Secret File)

SAFETY DEFAULTS: dry_run=True, enable_trading=False. Demo host only.
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
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Optional
from urllib.parse import urlparse

import requests
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("weather_kalshi_bot")


# =============================================================================
# CITY REGISTRY
# =============================================================================

@dataclass
class CityConfig:
    name: str
    lat: float
    lon: float
    high_ticker: str
    low_ticker: str
    station: str


# Approximate public station coordinates. Verify against api.weather.gov
# if a specific city's forecast looks misaligned with its settlement station.
ALL_CITIES: list[CityConfig] = [
    CityConfig("Atlanta", 33.6407, -84.4277, "KXHIGHTATL", "KXLOWTATL", "KATL"),
    CityConfig("Austin", 30.1975, -97.6664, "KXHIGHAUS", "KXLOWTAUS", "KAUS"),
    CityConfig("Boston", 42.3656, -71.0096, "KXHIGHTBOS", "KXLOWTBOS", "KBOS"),
    CityConfig("Chicago", 41.7868, -87.7522, "KXHIGHCHI", "KXLOWTCHI", "KMDW"),
    CityConfig("Dallas", 32.8998, -97.0403, "KXHIGHTDAL", "KXLOWTDAL", "KDFW"),
    CityConfig("Denver", 39.8561, -104.6737, "KXHIGHDEN", "KXLOWTDEN", "KDEN"),
    CityConfig("Houston", 29.6454, -95.2789, "KXHIGHTHOU", "KXLOWTHOU", "KHOU"),
    CityConfig("Las Vegas", 36.0840, -115.1537, "KXHIGHTLV", "KXLOWTLV", "KLAS"),
    CityConfig("Los Angeles", 33.9416, -118.4085, "KXHIGHLAX", "KXLOWTLAX", "KLAX"),
    CityConfig("Miami", 25.7959, -80.2870, "KXHIGHMIA", "KXLOWTMIA", "KMIA"),
    CityConfig("Minneapolis", 44.8848, -93.2223, "KXHIGHTMIN", "KXLOWTMIN", "KMSP"),
    CityConfig("New Orleans", 29.9934, -90.2580, "KXHIGHTNOLA", "KXLOWTNOLA", "KMSY"),
    CityConfig("New York City", 40.7825, -73.9655, "KXHIGHNY", "KXLOWTNYC", "KNYC"),
    CityConfig("Oklahoma City", 35.3931, -97.6007, "KXHIGHTOKC", "KXLOWTOKC", "KOKC"),
    CityConfig("Philadelphia", 39.8744, -75.2424, "KXHIGHPHIL", "KXLOWTPHIL", "KPHL"),
    CityConfig("Phoenix", 33.4342, -112.0116, "KXHIGHTPHX", "KXLOWTPHX", "KPHX"),
    CityConfig("San Antonio", 29.5337, -98.4698, "KXHIGHTSATX", "KXLOWTSATX", "KSAT"),
    CityConfig("San Francisco", 37.6213, -122.3790, "KXHIGHTSFO", "KXLOWTSFO", "KSFO"),
    CityConfig("Seattle", 47.4502, -122.3088, "KXHIGHTSEA", "KXLOWTSEA", "KSEA"),
    CityConfig("Washington D.C.", 38.8512, -77.0402, "KXHIGHTDC", "KXLOWTDC", "KDCA"),
]

# Start small; add more names once you've confirmed rate limits & ticker
# accuracy. Must match `.name` exactly from ALL_CITIES above.
ENABLED_CITIES = ["Chicago", "New York City", "Miami"]


# =============================================================================
# CONFIG
# =============================================================================

@dataclass
class Config:
    nws_user_agent: str = "MyWeatherBot/1.0 (contact@example.com)"

    sigma_f: float = 1.5
    bracket_width: int = 2
    brackets_each_side: int = 4

    kalshi_base_url: str = "https://demo-api.kalshi.co/trade-api/v2"
    kalshi_key_id: str = os.environ.get("KALSHI_KEY_ID", "")
    kalshi_private_key_path: str = os.environ.get(
        "KALSHI_PRIVATE_KEY_PATH", "kalshi_private_key.pem"
    )

    edge_threshold: float = 0.05
    max_order_contracts: int = 1
    poll_interval_sec: int = 300  # longer default given multi-city rate-limit cost

    dry_run: bool = True
    enable_trading: bool = False

    dashboard_host: str = "0.0.0.0"
    dashboard_port: int = int(os.environ.get("PORT", 8787))


CONFIG = Config()


# =============================================================================
# SHARED STATE
# =============================================================================

STATE_LOCK = threading.Lock()
STATE: dict[str, Any] = {
    "last_updated": None,
    "cities": {},          # name -> {high: {...}, low: {...}}
    "balance_dollars": None,
    "positions": [],
    "recent_orders": [],
    "errors": [],
}


def update_state(**kwargs: Any) -> None:
    with STATE_LOCK:
        STATE.update(kwargs)
        STATE["last_updated"] = datetime.utcnow().isoformat() + "Z"


def set_city_state(city_name: str, key: str, value: Any) -> None:
    with STATE_LOCK:
        STATE["cities"].setdefault(city_name, {})[key] = value


def push_error(msg: str) -> None:
    logger.error(msg)
    with STATE_LOCK:
        errs = STATE["errors"]
        errs.append({"time": datetime.utcnow().isoformat() + "Z", "message": msg})
        STATE["errors"] = errs[-30:]


def push_order_log(entry: dict[str, Any]) -> None:
    with STATE_LOCK:
        orders = STATE["recent_orders"]
        orders.append(entry)
        STATE["recent_orders"] = orders[-30:]


# =============================================================================
# NWS
# =============================================================================

def get_forecast_hourly_url(lat: float, lon: float) -> str:
    resp = requests.get(
        f"https://api.weather.gov/points/{lat},{lon}",
        headers={"User-Agent": CONFIG.nws_user_agent, "Accept": "application/geo+json"},
        timeout=10,
    )
    resp.raise_for_status()
    return resp.json()["properties"]["forecastHourly"]


def get_todays_projected_extreme(forecast_hourly_url: str, mode: str) -> float:
    """
    mode: "max" for today's high, "min" for today's low.
    NOTE: overnight lows often settle against the *next* calendar day on
    Kalshi, not the same day as this pull -- this is a simplification, not
    an exact match to Kalshi's settlement window. Treat with appropriate
    skepticism for the low-temp series specifically.
    """
    resp = requests.get(
        forecast_hourly_url,
        headers={"User-Agent": CONFIG.nws_user_agent, "Accept": "application/geo+json"},
        timeout=10,
    )
    resp.raise_for_status()
    periods = resp.json()["properties"]["periods"]
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
    return max(todays_temps) if mode == "max" else min(todays_temps)


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
# Kalshi client
# =============================================================================

class KalshiClient:
    def __init__(self, base_url: str, key_id: str, private_key_path: str):
        self.base_url = base_url.rstrip("/")
        self.key_id = key_id
        self._private_key = None
        if key_id and os.path.exists(private_key_path):
            with open(private_key_path, "rb") as f:
                self._private_key = serialization.load_pem_private_key(f.read(), password=None)
        else:
            logger.warning("No Kalshi key configured; authenticated calls will fail.")
        self.session = requests.Session()

    def _sign(self, method: str, path_no_query: str) -> dict[str, str]:
        if self._private_key is None:
            return {}
        ts = str(int(time.time() * 1000))
        message = (ts + method.upper() + path_no_query).encode("utf-8")
        sig = self._private_key.sign(
            message,
            padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.DIGEST_LENGTH),
            hashes.SHA256(),
        )
        return {
            "KALSHI-ACCESS-KEY": self.key_id,
            "KALSHI-ACCESS-TIMESTAMP": ts,
            "KALSHI-ACCESS-SIGNATURE": base64.b64encode(sig).decode("utf-8"),
        }

    def _request(self, method: str, path: str, **kwargs: Any) -> requests.Response:
        full_url = self.base_url + path
        headers = self._sign(method, urlparse(full_url).path)
        resp = self.session.request(method, full_url, headers=headers, timeout=15, **kwargs)
        resp.raise_for_status()
        return resp

    def get_markets(self, series_ticker: str, status: str = "open") -> list[dict]:
        markets: list[dict] = []
        cursor: Optional[str] = None
        while True:
            params = {"series_ticker": series_ticker, "status": status, "limit": 100}
            if cursor:
                params["cursor"] = cursor
            data = self._request("GET", "/markets", params=params).json()
            markets.extend(data.get("markets", []))
            cursor = data.get("cursor")
            if not cursor:
                break
        return markets

    def get_orderbooks_batched(self, tickers: list[str]) -> dict[str, dict]:
        """Up to 100 tickers per call -- far cheaper than one call per market."""
        if not tickers:
            return {}
        result: dict[str, dict] = {}
        for i in range(0, len(tickers), 100):
            chunk = tickers[i : i + 100]
            data = self._request(
                "GET", "/markets/orderbooks", params={"tickers": ",".join(chunk)}
            ).json()
            for ob in data.get("orderbooks", []):
                ticker = ob.get("ticker")
                if ticker:
                    result[ticker] = ob
        return result

    def get_balance(self) -> Optional[float]:
        try:
            data = self._request("GET", "/portfolio/balance").json()
            if "balance" in data:
                return data["balance"] / 100.0
            if "balance_dollars" in data:
                return float(data["balance_dollars"])
        except Exception as e:  # noqa: BLE001
            push_error(f"get_balance failed: {e}")
        return None

    def get_positions(self) -> list[dict]:
        try:
            return self._request("GET", "/portfolio/positions").json().get("market_positions", [])
        except Exception as e:  # noqa: BLE001
            push_error(f"get_positions failed: {e}")
            return []

    def create_order(self, ticker: str, side: str, action: str, price_dollars: float, count: int) -> dict:
        """Best-effort payload shape -- verify field names against docs.kalshi.com
        before trusting beyond dry-run testing."""
        body = {
            "ticker": ticker,
            "client_order_id": str(uuid.uuid4()),
            "action": action,
            "side": side,
            "type": "limit",
            ("yes_price_dollars" if side == "yes" else "no_price_dollars"): f"{price_dollars:.2f}",
            "count_fp": f"{count:.2f}",
            "time_in_force": "good_till_cancelled",
        }
        return self._request("POST", "/portfolio/events/orders", json=body).json()


kalshi = KalshiClient(CONFIG.kalshi_base_url, CONFIG.kalshi_key_id, CONFIG.kalshi_private_key_path)


# =============================================================================
# Bracket <-> market matching
# =============================================================================

_RANGE_RE = re.compile(r"(\d{1,3})\s*(?:°|degrees)?\s*(?:-|to)\s*(\d{1,3})\s*(?:°|degrees)?")


def extract_bracket_from_market(market: dict) -> Optional[tuple[int, int]]:
    floor_strike = market.get("floor_strike")
    cap_strike = market.get("cap_strike")
    if floor_strike is not None and cap_strike is not None:
        try:
            return int(round(float(floor_strike))), int(round(float(cap_strike)))
        except (TypeError, ValueError):
            pass
    for field_name in ("yes_sub_title", "subtitle", "title"):
        m = _RANGE_RE.search(market.get(field_name, ""))
        if m:
            return int(m.group(1)), int(m.group(2))
    return None


def best_yes_bid_ask(orderbook: dict) -> Optional[tuple[float, float]]:
    book = orderbook.get("orderbook_fp") or orderbook.get("orderbook") or orderbook
    yes_side = book.get("yes_dollars") or book.get("yes")
    no_side = book.get("no_dollars") or book.get("no")
    if not yes_side or not no_side:
        return None
    try:
        best_yes_bid = float(yes_side[-1][0])
        best_no_bid = float(no_side[-1][0])
    except (IndexError, ValueError, TypeError):
        return None
    return best_yes_bid, 1.0 - best_no_bid


# =============================================================================
# Core cycle
# =============================================================================

def evaluate_series(mu: float, series_ticker: str) -> list[dict]:
    brackets = build_brackets(mu, CONFIG.bracket_width, CONFIG.brackets_each_side)

    try:
        markets = kalshi.get_markets(series_ticker)
    except Exception as e:  # noqa: BLE001
        push_error(f"get_markets failed for {series_ticker}: {e}")
        markets = []

    market_by_range: dict[tuple[int, int], dict] = {}
    for m in markets:
        rng = extract_bracket_from_market(m)
        if rng:
            market_by_range[rng] = m

    matched_tickers = [
        m["ticker"] for m in market_by_range.values() if m.get("ticker")
    ]
    try:
        orderbooks = kalshi.get_orderbooks_batched(matched_tickers)
    except Exception as e:  # noqa: BLE001
        push_error(f"batched orderbook fetch failed for {series_ticker}: {e}")
        orderbooks = {}

    rows = []
    for low, high in brackets:
        model_prob = bracket_probability(mu, CONFIG.sigma_f, low - 0.5, high + 0.5)
        market = market_by_range.get((low, high))
        row = {
            "low": low, "high": high, "model_prob": model_prob,
            "ticker": market.get("ticker") if market else None,
            "market_prob": None, "edge": None,
            "action": "no market match" if not market else "-",
        }

        if market:
            quote = best_yes_bid_ask(orderbooks.get(market["ticker"], {}))
            if quote:
                yes_bid, yes_ask = quote
                market_prob = (yes_bid + yes_ask) / 2
                edge = model_prob - market_prob
                row["market_prob"] = market_prob
                row["edge"] = edge

                if abs(edge) >= CONFIG.edge_threshold:
                    side = "yes" if edge > 0 else "no"
                    limit_price = yes_ask if side == "yes" else round(1 - yes_bid, 2)
                    row["action"] = f"EDGE: buy {side.upper()} @ ~{limit_price:.2f}"
                    place_order_if_enabled(
                        ticker=market["ticker"], side=side, action="buy",
                        price_dollars=limit_price, count=CONFIG.max_order_contracts,
                        reason=f"{series_ticker} model={model_prob:.3f} market={market_prob:.3f} edge={edge:+.3f}",
                    )
                else:
                    row["action"] = "within edge threshold, no trade"
        rows.append(row)
    return rows


def place_order_if_enabled(ticker, side, action, price_dollars, count, reason) -> None:
    log_entry = {
        "time": datetime.utcnow().isoformat() + "Z", "ticker": ticker, "side": side,
        "action": action, "price": price_dollars, "count": count, "reason": reason,
        "dry_run": CONFIG.dry_run, "sent": False, "response": None,
    }
    if CONFIG.dry_run or not CONFIG.enable_trading:
        logger.info(f"[DRY RUN / trading disabled] Would place order: {log_entry}")
        push_order_log(log_entry)
        return
    try:
        resp = kalshi.create_order(ticker, side, action, price_dollars, count)
        log_entry["sent"] = True
        log_entry["response"] = resp
        logger.info(f"Order placed: {resp}")
    except Exception as e:  # noqa: BLE001
        push_error(f"create_order failed for {ticker}: {e}")
        log_entry["response"] = {"error": str(e)}
    push_order_log(log_entry)


def run_cycle() -> None:
    active_cities = [c for c in ALL_CITIES if c.name in ENABLED_CITIES]
    for city in active_cities:
        try:
            forecast_url = get_forecast_hourly_url(city.lat, city.lon)

            mu_high = get_todays_projected_extreme(forecast_url, "max")
            high_rows = evaluate_series(mu_high, city.high_ticker)
            set_city_state(city.name, "high", {"projected_f": mu_high, "brackets": high_rows})

            mu_low = get_todays_projected_extreme(forecast_url, "min")
            low_rows = evaluate_series(mu_low, city.low_ticker)
            set_city_state(city.name, "low", {"projected_f": mu_low, "brackets": low_rows})

            logger.info(f"{city.name}: high={mu_high:.1f}F low={mu_low:.1f}F")
        except Exception as e:  # noqa: BLE001
            push_error(f"{city.name} cycle failed: {e}")

    balance = kalshi.get_balance()
    positions = kalshi.get_positions()
    update_state(balance_dollars=balance, positions=positions)


def poll_loop() -> None:
    while True:
        run_cycle()
        time.sleep(CONFIG.poll_interval_sec)


# =============================================================================
# Dashboard
# =============================================================================

DASHBOARD_HTML = """<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Weather x Kalshi Dashboard</title>
<style>
  body { font-family: -apple-system, Segoe UI, Roboto, sans-serif; background: #0f1117; color: #e6e6e6; margin: 0; padding: 24px; }
  h1 { font-size: 20px; margin-bottom: 4px; }
  h2 { font-size: 15px; margin: 22px 0 6px; color: #cbd5e1; }
  .sub { color: #8a8f98; font-size: 13px; margin-bottom: 16px; }
  .topcards { display: flex; gap: 16px; margin-bottom: 20px; flex-wrap: wrap; }
  .card { background: #171a21; border: 1px solid #262b36; border-radius: 8px; padding: 12px 16px; min-width: 140px; }
  .card .label { font-size: 11px; color: #8a8f98; } .card .value { font-size: 20px; font-weight: 600; margin-top: 2px; }
  table { width: 100%; border-collapse: collapse; font-size: 12.5px; margin-bottom: 6px; }
  th, td { padding: 6px 10px; text-align: right; border-bottom: 1px solid #262b36; }
  th:first-child, td:first-child { text-align: left; }
  th { color: #8a8f98; font-weight: 500; }
  .pos { color: #4ade80; } .neg { color: #f87171; }
  .badge { padding: 2px 8px; border-radius: 4px; font-size: 11px; }
  .badge.trade { background: #1e3a8a; color: #93c5fd; } .badge.none { background: #262b36; color: #8a8f98; }
  .cityblock { background: #12141b; border: 1px solid #1f232d; border-radius: 8px; padding: 14px 16px; margin-bottom: 18px; }
  .errors { margin-top: 20px; } .errors .err { color: #f87171; font-size: 12px; margin-bottom: 4px; }
  .footer { margin-top: 20px; font-size: 12px; color: #8a8f98; }
</style></head>
<body>
  <h1>Weather &rarr; Kalshi Prediction Market Dashboard</h1>
  <div class="sub" id="subheader">Loading...</div>
  <div class="topcards" id="topcards"></div>
  <div id="citiesContainer"></div>
  <div class="errors" id="errors"></div>
  <div class="footer">Auto-refreshes every 5s. Demo/paper trading only unless explicitly reconfigured.</div>

<script>
function renderTable(brackets) {
  return `<table><thead><tr><th>Bracket</th><th>Model</th><th>Market</th><th>Edge</th><th>Ticker</th><th>Status</th></tr></thead><tbody>
    ${brackets.map(b => {
      const edgeClass = b.edge == null ? '' : (b.edge > 0 ? 'pos' : 'neg');
      const badge = (b.action && b.action.startsWith('EDGE')) ? 'trade' : 'none';
      return `<tr>
        <td>${b.low}&deg;-${b.high}&deg;F</td>
        <td>${(b.model_prob*100).toFixed(2)}%</td>
        <td>${b.market_prob != null ? (b.market_prob*100).toFixed(2)+'%' : '-'}</td>
        <td class="${edgeClass}">${b.edge != null ? (b.edge*100).toFixed(2)+'%' : '-'}</td>
        <td>${b.ticker ?? '-'}</td>
        <td><span class="badge ${badge}">${b.action}</span></td>
      </tr>`;
    }).join('')}
  </tbody></table>`;
}

async function refresh() {
  const res = await fetch('/api/state');
  const s = await res.json();

  document.getElementById('subheader').innerHTML = `last updated: ${s.last_updated ?? 'never'}`;
  document.getElementById('topcards').innerHTML = `
    <div class="card"><div class="label">Demo Balance</div><div class="value">${s.balance_dollars != null ? '$' + s.balance_dollars.toFixed(2) : '-'}</div></div>
    <div class="card"><div class="label">Open Positions</div><div class="value">${s.positions.length}</div></div>
    <div class="card"><div class="label">Cities Tracked</div><div class="value">${Object.keys(s.cities).length}</div></div>
  `;

  const container = document.getElementById('citiesContainer');
  container.innerHTML = Object.entries(s.cities).map(([name, data]) => `
    <div class="cityblock">
      <h2>${name}</h2>
      <h2 style="color:#8a8f98;font-size:12px;font-weight:400;">High: ${data.high?.projected_f?.toFixed(1) ?? '-'}&deg;F</h2>
      ${renderTable(data.high?.brackets ?? [])}
      <h2 style="color:#8a8f98;font-size:12px;font-weight:400;margin-top:14px;">Low: ${data.low?.projected_f?.toFixed(1) ?? '-'}&deg;F</h2>
      ${renderTable(data.low?.brackets ?? [])}
    </div>
  `).join('');

  document.getElementById('errors').innerHTML =
    s.errors.slice(-5).map(e => `<div class="err">${e.time}: ${e.message}</div>`).join('');
}
refresh();
setInterval(refresh, 5000);
</script>
</body></html>
"""


class DashboardHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args: Any) -> None:
        pass

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/":
            self._send(200, DASHBOARD_HTML.encode("utf-8"), "text/html")
        elif path == "/api/state":
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
        f"cities={ENABLED_CITIES} host={CONFIG.kalshi_base_url}"
    )
    threading.Thread(target=poll_loop, daemon=True).start()
    server = ThreadingHTTPServer((CONFIG.dashboard_host, CONFIG.dashboard_port), DashboardHandler)
    logger.info(f"Dashboard live on port {CONFIG.dashboard_port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.shutdown()


if __name__ == "__main__":
    main()
