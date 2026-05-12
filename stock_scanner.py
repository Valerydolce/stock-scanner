#!/usr/bin/env python3
"""
stock_scanner.py — Daily stock scanner with email alerts.

Three layers of signal:

  STRICT (your original spec):
    BUY  : RSI(14) <= 24  AND  low  touches/below lower BB
    SELL : RSI(14) >= 90  AND  high touches/above upper BB
    Always alerts. Flagged ⚠ if firing against the 200-day trend or near earnings.

  CONNORS (research-backed mean reversion, Connors RSI(2) + 200-SMA filter):
    BUY  : RSI(2) <= 5   AND  close ABOVE 200-day SMA  AND  no earnings in 3d
    SELL : RSI(2) >= 95  AND  close BELOW 200-day SMA  AND  no earnings in 3d
    Filter-strict. Suggested exit: close above 5-day SMA (long) / below (short).

  HEAT MAP (new — what's actually moving today):
    Top N quality movers from S&P 500 ranked by (vol_ratio × |% change|).
    These get the same full indicator analysis as the watchlist, so you
    catch real opportunities that aren't on your fixed list.

Usage:
    python stock_scanner.py                 # watchlist + heat map, email if signals
    python stock_scanner.py --always-email  # send daily report regardless
    python stock_scanner.py --dry-run       # scan + print, no email
    python stock_scanner.py --no-heat       # watchlist only, skip heat map
    python stock_scanner.py --heat-only     # heat map only, skip watchlist
    python stock_scanner.py --top-movers 25 # change number of heat movers
    python stock_scanner.py --tickers AAPL,TSLA   # ad-hoc list override

Requires:
    pip install yfinance pandas numpy python-dotenv lxml html5lib
"""

import os, sys, json, smtplib, argparse, logging
from datetime import datetime, timezone, date
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from pathlib import Path

import yfinance as yf
import pandas as pd
import numpy as np
from dotenv import load_dotenv

import quality_universe

load_dotenv()

# ── Core watchlist (always scanned unless --heat-only) ───────────────────────
WATCHLIST = [
    # Magnificent 7
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA",
    # Other stable / high-liquidity names
    "JPM", "V", "MA", "JNJ", "PG", "KO", "WMT", "COST",
    "HD", "UNH", "BRK-B", "DIS", "NFLX", "AVGO", "ORCL",
    "XOM", "CVX", "PEP", "ADBE", "CRM",
]

# ── Strict criteria (original spec) ──────────────────────────────────────────
RSI_OVERSOLD       = 24
RSI_OVERBOUGHT     = 90
BB_TOUCH_TOLERANCE = 0.005

# ── Connors RSI(2) thresholds ────────────────────────────────────────────────
RSI2_OVERSOLD   = 5
RSI2_OVERBOUGHT = 95

# ── Trend filter ─────────────────────────────────────────────────────────────
SMA_TREND_PERIOD = 200

# ── Earnings blackout ────────────────────────────────────────────────────────
EARNINGS_BLACKOUT_DAYS = 3

# ── Near-miss thresholds (context only, not alerted) ─────────────────────────
NEAR_OVERSOLD   = 30
NEAR_OVERBOUGHT = 80

# ── Heat map ─────────────────────────────────────────────────────────────────
HEAT_TOP_N = 15   # how many top S&P 500 movers to deep-scan by default

# ── Indicator periods ────────────────────────────────────────────────────────
RSI_PERIOD  = 14
RSI2_PERIOD = 2
BB_PERIOD   = 20
BB_STD_MULT = 2.0
HISTORY_PERIOD = "13mo"

# ── Files ────────────────────────────────────────────────────────────────────
LOG_FILE   = "stock_scans.jsonl"
ALERT_FILE = "stock_alerts.jsonl"

# ── Email ────────────────────────────────────────────────────────────────────
SMTP_HOST   = os.getenv("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT   = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER   = os.getenv("SMTP_USER", "")
SMTP_PASS   = os.getenv("SMTP_PASS", "")
EMAIL_FROM  = os.getenv("EMAIL_FROM", SMTP_USER)
EMAIL_TO    = os.getenv("EMAIL_TO", "")

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Indicators
# ─────────────────────────────────────────────────────────────────────────────

def compute_rsi(prices: pd.Series, period: int) -> pd.Series:
    delta = prices.diff()
    gain  = delta.where(delta > 0, 0.0)
    loss  = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def compute_bollinger(prices: pd.Series,
                       period: int = BB_PERIOD,
                       std_mult: float = BB_STD_MULT):
    middle = prices.rolling(window=period).mean()
    std    = prices.rolling(window=period).std()
    upper  = middle + std_mult * std
    lower  = middle - std_mult * std
    return upper, middle, lower

# ─────────────────────────────────────────────────────────────────────────────
# Earnings
# ─────────────────────────────────────────────────────────────────────────────

def _to_date(x):
    if x is None:
        return None
    try:
        if hasattr(x, "date") and callable(x.date):
            return x.date()
        if isinstance(x, date):
            return x
        if isinstance(x, str):
            return datetime.fromisoformat(x.split("T")[0]).date()
    except Exception:
        return None
    return None

def get_days_to_earnings(ticker_symbol: str):
    try:
        t = yf.Ticker(ticker_symbol)
        # Try 1: calendar
        try:
            cal = t.calendar
            earnings_date = None
            if isinstance(cal, dict):
                ed = cal.get("Earnings Date")
                if isinstance(ed, list) and ed:
                    earnings_date = ed[0]
                elif ed is not None:
                    earnings_date = ed
            elif hasattr(cal, "loc"):
                try:
                    earnings_date = cal.loc["Earnings Date"].iloc[0]
                except Exception:
                    pass
            if earnings_date is not None:
                d = _to_date(earnings_date)
                if d:
                    delta = (d - date.today()).days
                    if delta >= 0:
                        return delta
        except Exception:
            pass
        # Try 2: earnings_dates
        try:
            ed = t.earnings_dates
            if ed is not None and not ed.empty:
                today = date.today()
                future = [_to_date(ix) for ix in ed.index]
                future = [d for d in future if d and d >= today]
                if future:
                    return (min(future) - today).days
        except Exception:
            pass
    except Exception:
        pass
    return None

# ─────────────────────────────────────────────────────────────────────────────
# Scan logic
# ─────────────────────────────────────────────────────────────────────────────

def scan_ticker(ticker: str, heat_info: dict = None) -> dict:
    try:
        df = yf.download(ticker, period=HISTORY_PERIOD,
                         progress=False, auto_adjust=True)
        if df.empty or len(df) < SMA_TREND_PERIOD + 5:
            return {"ticker": ticker, "error": f"insufficient data ({len(df)} bars)"}

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        close = df["Close"]
        high  = df["High"]
        low   = df["Low"]

        rsi14 = compute_rsi(close, RSI_PERIOD)
        rsi2  = compute_rsi(close, RSI2_PERIOD)
        upper, middle, lower = compute_bollinger(close)
        sma200 = close.rolling(window=SMA_TREND_PERIOD).mean()

        last_close = float(close.iloc[-1])
        last_high  = float(high.iloc[-1])
        last_low   = float(low.iloc[-1])
        last_rsi14 = float(rsi14.iloc[-1])
        last_rsi2  = float(rsi2.iloc[-1])
        last_upper = float(upper.iloc[-1])
        last_lower = float(lower.iloc[-1])
        last_sma   = float(sma200.iloc[-1])
        last_date  = df.index[-1].strftime("%Y-%m-%d")

        above_200sma = last_close > last_sma
        sma_dist_pct = (last_close / last_sma - 1) * 100

        touched_lower = last_low  <= last_lower * (1 + BB_TOUCH_TOLERANCE)
        touched_upper = last_high >= last_upper * (1 - BB_TOUCH_TOLERANCE)

        band_width = last_upper - last_lower
        pct_b = (last_close - last_lower) / band_width if band_width > 0 else 0.5

        days_to_earnings = get_days_to_earnings(ticker)
        earnings_blackout = (days_to_earnings is not None
                              and days_to_earnings <= EARNINGS_BLACKOUT_DAYS)

        strict_buy  = (last_rsi14 <= RSI_OVERSOLD)   and touched_lower
        strict_sell = (last_rsi14 >= RSI_OVERBOUGHT) and touched_upper
        against_trend = ((strict_buy and not above_200sma) or
                          (strict_sell and above_200sma))

        connors_buy  = (last_rsi2 <= RSI2_OVERSOLD
                         and above_200sma
                         and not earnings_blackout)
        connors_sell = (last_rsi2 >= RSI2_OVERBOUGHT
                         and not above_200sma
                         and not earnings_blackout)

        near_buy  = (last_rsi14 <= NEAR_OVERSOLD)
        near_sell = (last_rsi14 >= NEAR_OVERBOUGHT)

        result = {
            "ticker": ticker,
            "date": last_date,
            "close": round(last_close, 2),
            "rsi14": round(last_rsi14, 1),
            "rsi2":  round(last_rsi2, 1),
            "bb_lower": round(last_lower, 2),
            "bb_upper": round(last_upper, 2),
            "pct_b": round(pct_b, 3),
            "sma_200": round(last_sma, 2),
            "above_200sma": above_200sma,
            "sma_dist_pct": round(sma_dist_pct, 1),
            "days_to_earnings": days_to_earnings,
            "earnings_blackout": earnings_blackout,
            "touched_lower": touched_lower,
            "touched_upper": touched_upper,
            "strict_buy":  strict_buy,
            "strict_sell": strict_sell,
            "against_trend": against_trend,
            "connors_buy":  connors_buy,
            "connors_sell": connors_sell,
            "near_buy": near_buy,
            "near_sell": near_sell,
            "source": "watchlist",
        }
        # Merge heat info if this ticker came from the heat map
        if heat_info:
            result["source"]    = "heat"
            result["heat"]      = heat_info["heat"]
            result["pct_change"]= heat_info["pct_change"]
            result["vol_ratio"] = heat_info["vol_ratio"]
            result["direction"] = heat_info["direction"]
        return result
    except Exception as e:
        return {"ticker": ticker, "error": str(e)}

# ─────────────────────────────────────────────────────────────────────────────
# Email
# ─────────────────────────────────────────────────────────────────────────────

def _earnings_label(d):
    if d is None: return "?"
    if d <= EARNINGS_BLACKOUT_DAYS: return f"⚠ {d}d"
    return f"{d}d"

def _trend_label(r):
    arrow = "↑" if r["above_200sma"] else "↓"
    return f"{arrow} {r['sma_dist_pct']:+.1f}%"

def _signal_summary(r):
    parts = []
    if r.get("strict_buy"):  parts.append("STRICT BUY")
    if r.get("strict_sell"): parts.append("STRICT SELL")
    if r.get("connors_buy"): parts.append("CONNORS BUY")
    if r.get("connors_sell"):parts.append("CONNORS SELL")
    return " · ".join(parts) if parts else "—"

def build_email_html(results: list,
                      strict_alerts: list,
                      connors_alerts: list,
                      heat_results: list) -> str:
    today = datetime.now().strftime("%Y-%m-%d")
    css = ("font-family: -apple-system, Arial, sans-serif; "
           "font-size: 14px; color: #222;")
    html = f"<html><body style='{css}'>"
    html += f"<h2 style='margin-bottom:4px;'>Daily Stock Scanner — {today}</h2>"
    n_alerts = len(strict_alerts) + len(connors_alerts)
    html += (f"<p style='color:#666; font-size:12px; margin-top:0;'>"
             f"Scanned {len(results)} tickers · {n_alerts} signal(s) "
             f"({len(strict_alerts)} strict, {len(connors_alerts)} Connors)</p>")

    # Strict alerts
    if strict_alerts:
        html += ("<h3 style='color:#d9534f; margin-top:18px;'>"
                 f"Strict signals · RSI(14) extreme + BB touch ({len(strict_alerts)})</h3>")
        html += ("<table cellpadding='8' style='border-collapse:collapse; "
                 "border:1px solid #ccc;'>")
        html += ("<tr style='background:#f5f5f5;'>"
                 "<th align='left'>Ticker</th><th align='left'>Signal</th>"
                 "<th align='right'>Close</th><th align='right'>RSI(14)</th>"
                 "<th align='right'>%B</th><th align='left'>200-SMA</th>"
                 "<th align='left'>Earnings</th><th align='left'>Notes</th></tr>")
        for a in strict_alerts:
            sig   = "BUY (oversold)" if a["strict_buy"] else "SELL (overbought)"
            color = "#1f7a1f" if a["strict_buy"] else "#b22222"
            notes = []
            if a["against_trend"]:     notes.append("⚠ against 200-SMA")
            if a["earnings_blackout"]: notes.append("⚠ earnings soon")
            notes_str = ", ".join(notes) or "trend-aligned ✓"
            html += (
                f"<tr style='border-top:1px solid #eee;'>"
                f"<td><b>{a['ticker']}</b></td>"
                f"<td style='color:{color};'><b>{sig}</b></td>"
                f"<td align='right'>${a['close']}</td>"
                f"<td align='right'>{a['rsi14']}</td>"
                f"<td align='right'>{a['pct_b']}</td>"
                f"<td>{_trend_label(a)}</td>"
                f"<td>{_earnings_label(a['days_to_earnings'])}</td>"
                f"<td style='font-size:12px;'>{notes_str}</td></tr>"
            )
        html += "</table>"

    # Connors alerts
    if connors_alerts:
        html += ("<h3 style='color:#1f5f9f; margin-top:18px;'>"
                 f"Connors signals · RSI(2) + 200-SMA filter ({len(connors_alerts)})</h3>")
        html += ("<p style='color:#666; font-size:12px; margin-top:-8px;'>"
                 "Trend-aligned and earnings-clean by construction. "
                 "Suggested exit: close above 5-day SMA (longs) / below (shorts).</p>")
        html += ("<table cellpadding='8' style='border-collapse:collapse; "
                 "border:1px solid #ccc;'>")
        html += ("<tr style='background:#f0f5ff;'>"
                 "<th align='left'>Ticker</th><th align='left'>Signal</th>"
                 "<th align='right'>Close</th><th align='right'>RSI(2)</th>"
                 "<th align='left'>200-SMA</th><th align='left'>Earnings</th></tr>")
        for a in connors_alerts:
            sig   = "BUY" if a["connors_buy"] else "SELL"
            color = "#1f7a1f" if a["connors_buy"] else "#b22222"
            html += (
                f"<tr style='border-top:1px solid #eee;'>"
                f"<td><b>{a['ticker']}</b></td>"
                f"<td style='color:{color};'><b>{sig}</b></td>"
                f"<td align='right'>${a['close']}</td>"
                f"<td align='right'>{a['rsi2']}</td>"
                f"<td>{_trend_label(a)}</td>"
                f"<td>{_earnings_label(a['days_to_earnings'])}</td></tr>"
            )
        html += "</table>"

    if not (strict_alerts or connors_alerts):
        html += "<p>No signals triggered today.</p>"

    # 🔥 Heat map — top quality movers, ranked
    if heat_results:
        html += ("<h3 style='margin-top:24px;'>"
                 f"🔥 Today's Heat Map · Top {len(heat_results)} S&P 500 movers</h3>")
        html += ("<p style='color:#666; font-size:12px; margin-top:-8px;'>"
                 "Ranked by (volume vs 20-day avg) × |% change|. "
                 "Quality = S&P 500 membership only. Rows highlighted yellow "
                 "have triggered a strict or Connors signal.</p>")
        html += ("<table cellpadding='6' style='border-collapse:collapse; "
                 "border:1px solid #ccc; font-size:13px;'>")
        html += ("<tr style='background:#fff5e6;'>"
                 "<th align='left'>Ticker</th><th align='left'>Dir</th>"
                 "<th align='right'>Close</th><th align='right'>%Chg</th>"
                 "<th align='right'>Vol×</th><th align='right'>Heat</th>"
                 "<th align='right'>RSI(14)</th><th align='right'>RSI(2)</th>"
                 "<th align='right'>%B</th><th align='left'>200-SMA</th>"
                 "<th align='left'>Earnings</th><th align='left'>Signal</th></tr>")
        for r in heat_results:
            if "error" in r:
                html += (f"<tr><td>{r['ticker']}</td>"
                         f"<td colspan='11' style='color:#999;'>{r['error']}</td></tr>")
                continue
            triggered = (r.get("strict_buy") or r.get("strict_sell")
                          or r.get("connors_buy") or r.get("connors_sell"))
            bg = "background:#fff3cd;" if triggered else ""
            arrow = "↑" if r.get("direction") == "up" else "↓"
            arrow_color = "#1f7a1f" if r.get("direction") == "up" else "#b22222"
            html += (
                f"<tr style='{bg} border-top:1px solid #eee;'>"
                f"<td><b>{r['ticker']}</b></td>"
                f"<td style='color:{arrow_color}; font-weight:bold;'>{arrow}</td>"
                f"<td align='right'>${r['close']}</td>"
                f"<td align='right'>{r.get('pct_change', 0):+.2f}%</td>"
                f"<td align='right'>{r.get('vol_ratio', 0):.1f}x</td>"
                f"<td align='right'><b>{r.get('heat', 0):.1f}</b></td>"
                f"<td align='right'>{r['rsi14']}</td>"
                f"<td align='right'>{r['rsi2']}</td>"
                f"<td align='right'>{r['pct_b']}</td>"
                f"<td>{_trend_label(r)}</td>"
                f"<td>{_earnings_label(r['days_to_earnings'])}</td>"
                f"<td style='font-size:11px;'>{_signal_summary(r)}</td></tr>"
            )
        html += "</table>"

    # Watchlist snapshot
    watchlist_rows = [r for r in results if "error" not in r
                      and r.get("source") == "watchlist"]
    if watchlist_rows:
        html += "<h4 style='margin-top:24px;'>Core watchlist snapshot</h4>"
        html += ("<table cellpadding='5' style='border-collapse:collapse; "
                 "border:1px solid #ddd; font-size:12px;'>")
        html += ("<tr style='background:#fafafa;'>"
                 "<th align='left'>Ticker</th><th align='right'>Close</th>"
                 "<th align='right'>RSI(14)</th><th align='right'>RSI(2)</th>"
                 "<th align='right'>%B</th><th align='left'>200-SMA</th>"
                 "<th align='left'>Earnings</th></tr>")
        for r in watchlist_rows:
            bg = ""
            if (r["strict_buy"] or r["strict_sell"]
                or r["connors_buy"] or r["connors_sell"]):
                bg = "background:#fff3cd;"
            elif r["near_buy"] or r["near_sell"]:
                bg = "background:#f9fafb;"
            html += (f"<tr style='{bg} border-top:1px solid #eee;'>"
                     f"<td>{r['ticker']}</td>"
                     f"<td align='right'>${r['close']}</td>"
                     f"<td align='right'>{r['rsi14']}</td>"
                     f"<td align='right'>{r['rsi2']}</td>"
                     f"<td align='right'>{r['pct_b']}</td>"
                     f"<td>{_trend_label(r)}</td>"
                     f"<td>{_earnings_label(r['days_to_earnings'])}</td></tr>")
        html += "</table>"

    html += ("<p style='color:#888; font-size:11px; margin-top:20px;'>"
             "Generated by stock_scanner.py. Not investment advice. "
             "Strict: RSI(14) ≤ 24 / ≥ 90 + BB touch. "
             "Connors: RSI(2) ≤ 5 / ≥ 95 + 200-SMA filter + 3-day earnings blackout. "
             "Heat = vol-ratio × |% change|; quality universe is S&P 500.</p>")
    html += "</body></html>"
    return html

def send_email(subject: str, html: str) -> bool:
    if not (SMTP_USER and SMTP_PASS and EMAIL_TO):
        log.warning("Email creds missing — skipping. "
                    "Set SMTP_USER, SMTP_PASS, EMAIL_TO in .env")
        return False
    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = EMAIL_FROM
    msg["To"] = EMAIL_TO
    msg.attach(MIMEText(html, "html"))
    try:
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as s:
            s.starttls()
            s.login(SMTP_USER, SMTP_PASS)
            s.send_message(msg)
        log.info(f"Email sent to {EMAIL_TO}")
        return True
    except Exception as e:
        log.error(f"Email send failed: {e}")
        return False

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def log_jsonl(path: str, record: dict):
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run",       action="store_true",
                        help="Scan + print, no email")
    parser.add_argument("--no-email",      action="store_true",
                        help="Same as --dry-run")
    parser.add_argument("--always-email",  action="store_true",
                        help="Send email even if zero signals")
    parser.add_argument("--no-heat",       action="store_true",
                        help="Skip S&P 500 heat map, watchlist only")
    parser.add_argument("--heat-only",     action="store_true",
                        help="Skip watchlist, scan only S&P 500 heat movers")
    parser.add_argument("--top-movers",    type=int, default=HEAT_TOP_N,
                        help=f"How many top movers to deep-scan (default {HEAT_TOP_N})")
    parser.add_argument("--tickers", type=str, default="",
                        help="Comma-separated override list, e.g. AAPL,TSLA,NVDA")
    args = parser.parse_args()

    no_email = args.dry_run or args.no_email

    # Build ticker list
    if args.tickers:
        watch_tickers = [t.strip().upper() for t in args.tickers.split(",")
                          if t.strip()]
        heat_info_map = {}
    elif args.heat_only:
        watch_tickers = []
        log.info(f"Fetching top {args.top_movers} S&P 500 heat movers...")
        try:
            movers = quality_universe.top_movers(args.top_movers)
            heat_info_map = {m["ticker"]: m for m in movers}
        except Exception as e:
            log.warning(f"Heat map fetch failed: {e}")
            heat_info_map = {}
    else:
        watch_tickers = WATCHLIST.copy()
        heat_info_map = {}
        if not args.no_heat:
            log.info(f"Fetching top {args.top_movers} S&P 500 heat movers...")
            try:
                movers = quality_universe.top_movers(args.top_movers)
                heat_info_map = {m["ticker"]: m for m in movers}
            except Exception as e:
                log.warning(f"Heat map fetch failed: {e}")

    # Dedupe: heat movers that are already in watchlist keep watchlist as source
    all_tickers = list(watch_tickers)
    for t in heat_info_map.keys():
        if t not in all_tickers:
            all_tickers.append(t)

    log.info(f"Scanning {len(all_tickers)} tickers "
             f"({len(watch_tickers)} watchlist + {len(heat_info_map)} heat)...")

    results, strict_alerts, connors_alerts, heat_results = [], [], [], []

    for t in all_tickers:
        heat_info = heat_info_map.get(t) if t not in watch_tickers else None
        r = scan_ticker(t, heat_info=heat_info)
        results.append(r)

        # Always record heat-driven scans so the email shows them
        if t in heat_info_map:
            # Attach heat info even for watchlist overlap
            if heat_info is None and "error" not in r:
                hm = heat_info_map[t]
                r["heat"]      = hm["heat"]
                r["pct_change"]= hm["pct_change"]
                r["vol_ratio"] = hm["vol_ratio"]
                r["direction"] = hm["direction"]
            heat_results.append(r)

        if "error" in r:
            log.warning(f"  {t}: {r['error']}")
            continue

        flags = []
        if r["strict_buy"]:
            flags.append("STRICT BUY"); strict_alerts.append(r)
        elif r["strict_sell"]:
            flags.append("STRICT SELL"); strict_alerts.append(r)
        if r["connors_buy"]:
            flags.append("CONNORS BUY"); connors_alerts.append(r)
        elif r["connors_sell"]:
            flags.append("CONNORS SELL"); connors_alerts.append(r)
        if r["against_trend"]:
            flags.append("against trend")
        if r["earnings_blackout"] and (r["strict_buy"] or r["strict_sell"]):
            flags.append("earnings soon")

        suffix = ""
        if flags:
            suffix = "   <-- " + " | ".join(flags)
        elif r["near_buy"]:
            suffix = "   .. near oversold"
        elif r["near_sell"]:
            suffix = "   .. near overbought"

        ed = r["days_to_earnings"]
        ed_str = f"E={ed}d" if ed is not None else "E=?"
        trend_str = "above" if r["above_200sma"] else "below"
        tag = "🔥" if t in heat_info_map else " "
        log.info(f"  {tag} {t:7s} ${r['close']:>8.2f}  "
                 f"RSI14={r['rsi14']:>5.1f}  RSI2={r['rsi2']:>5.1f}  "
                 f"%B={r['pct_b']:.2f}  {trend_str} 200SMA  {ed_str}{suffix}")

    # Heat results sorted by heat score
    heat_results.sort(key=lambda x: x.get("heat", 0), reverse=True)

    ts = datetime.now(timezone.utc).isoformat()
    log_jsonl(LOG_FILE, {
        "ts": ts,
        "scanned": len(results),
        "strict_alerts": len(strict_alerts),
        "connors_alerts": len(connors_alerts),
        "heat_count": len(heat_results),
        "strict_tickers":  [a["ticker"] for a in strict_alerts],
        "connors_tickers": [a["ticker"] for a in connors_alerts],
        "heat_tickers":    [r["ticker"] for r in heat_results],
    })
    for a in strict_alerts + connors_alerts:
        log_jsonl(ALERT_FILE, {**a, "ts": ts})

    total = len(strict_alerts) + len(connors_alerts)
    log.info(f"Scan done. {total} signal(s) — "
             f"{len(strict_alerts)} strict, {len(connors_alerts)} connors, "
             f"{len(heat_results)} heat-scanned.")

    if no_email:
        log.info("--dry-run: skipping email")
        return
    if total == 0 and not args.always_email and not heat_results:
        log.info("No signals — email not sent. Use --always-email to override.")
        return

    today = datetime.now().strftime("%Y-%m-%d")
    if total > 0:
        subject = f"📈 Stock Scanner — {total} signal(s) — {today}"
    else:
        subject = f"Stock Scanner — heat map only — {today}"
    send_email(subject,
               build_email_html(results, strict_alerts,
                                connors_alerts, heat_results))

if __name__ == "__main__":
    main()
