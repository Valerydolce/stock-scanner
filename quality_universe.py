#!/usr/bin/env python3
"""
quality_universe.py — Maintain S&P 500 ticker list and compute "heat" scores.

Quality definition = S&P 500 membership. The index requires:
  - ~$15B+ market cap
  - 4 consecutive profitable quarters
  - Audited financials
  - U.S.-domiciled, traded on NYSE/Nasdaq

This filters out the speculative names (OPEN, MARA, RDW, EOSE...) that
dominate retail "most-bought" heat maps and blow up accounts.

Heat score = (today's volume / 20-day avg volume) × abs(% change today).
Direction (up / down) is reported separately — both can be tradeable.

Source resilience:
  1. GitHub CSV (datasets/s-and-p-500-companies) — most reliable
  2. Wikipedia with browser User-Agent — fallback
  3. Stale local cache — if both above fail
  4. Hardcoded list of ~150 large caps — absolute last resort

CLI:
    python quality_universe.py            # print top 20 movers right now
    python quality_universe.py --refresh  # force-refresh the S&P 500 cache
"""

import json, time, logging, argparse, io
from pathlib import Path
from datetime import datetime

import pandas as pd
import yfinance as yf
import requests

log = logging.getLogger(__name__)

SP500_CACHE = Path("sp500.json")
SP500_CACHE_DAYS = 7

# Browser-like User-Agent so Wikipedia / GitHub don't 403 us
USER_AGENT = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
              "AppleWebKit/537.36 (KHTML, like Gecko) "
              "Chrome/120.0.0.0 Safari/537.36")

# Tried in order — first one that works wins
SP500_SOURCES = [
    {
        "name": "GitHub (datasets/s-and-p-500-companies)",
        "url":  "https://raw.githubusercontent.com/datasets/"
                 "s-and-p-500-companies/master/data/constituents.csv",
        "method": "csv",
    },
    {
        "name": "Wikipedia",
        "url":  "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
        "method": "html",
    },
]

# Last-resort hardcoded fallback — ~150 of the largest, most stable S&P 500
# names. Used only if all network sources fail AND no cache exists. This is
# intentionally not the full 500: when offline, narrower-but-stable is safer.
SP500_FALLBACK = [
    # Top 50
    "AAPL","MSFT","NVDA","GOOGL","GOOG","AMZN","META","BRK-B","LLY","AVGO",
    "TSLA","JPM","V","WMT","XOM","MA","UNH","ORCL","COST","PG",
    "JNJ","HD","NFLX","BAC","CRM","ABBV","KO","CVX","AMD","MRK",
    "PEP","ADBE","TMO","LIN","CSCO","ACN","ABT","MCD","IBM","NOW",
    "GE","AXP","WFC","MS","DIS","INTU","PM","GS","T","ISRG",
    # Next 50
    "CAT","QCOM","VZ","BKNG","AMGN","SPGI","DHR","UBER","BLK","PFE",
    "HON","LOW","COP","C","AMAT","UNP","ANET","SYK","LMT","ETN",
    "NKE","TJX","BSX","MDT","ELV","GILD","MU","SCHW","ADP","PANW",
    "BA","VRTX","MMC","CB","KLAC","REGN","INTC","SBUX","ADI","DE",
    "LRCX","FI","ICE","MO","CMG","BMY","AMT","PYPL","EQIX","USB",
    # Next 50
    "PGR","ENB","ZTS","AON","NOC","MDLZ","FCX","CL","TT","SO",
    "DUK","GD","TMUS","APH","SHW","GM","CME","EMR","MMM","MCK",
    "ITW","CSX","NSC","FDX","PSX","SRE","COF","MAR","CTAS","NXPI",
    "EOG","KMI","SLB","F","ORLY","AJG","DXCM","CDNS","MET","AIG",
    "PNC","ALL","ROP","MSI","BK","CTSH","AFL","KMB","HUM","NEM",
]


def _fetch_from_csv(url: str) -> list:
    resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=15)
    resp.raise_for_status()
    df = pd.read_csv(io.StringIO(resp.text))
    col = "Symbol" if "Symbol" in df.columns else df.columns[0]
    return (df[col].astype(str)
                   .str.replace(".", "-", regex=False)
                   .tolist())


def _fetch_from_html(url: str) -> list:
    resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=15)
    resp.raise_for_status()
    tables = pd.read_html(io.StringIO(resp.text))
    df = tables[0]
    col = "Symbol" if "Symbol" in df.columns else df.columns[0]
    return (df[col].astype(str)
                   .str.replace(".", "-", regex=False)
                   .tolist())


def _sanitize(tickers: list) -> list:
    return [t for t in tickers
            if t and len(t) <= 6 and t.replace("-", "").isalpha()]


def get_sp500_tickers(force_refresh: bool = False) -> list:
    """Return list of S&P 500 tickers, cached for 7 days, with multi-source
    fallback. Never raises — returns the hardcoded fallback in the worst case."""

    # 1. Recent cache
    if not force_refresh and SP500_CACHE.exists():
        age_days = (time.time() - SP500_CACHE.stat().st_mtime) / 86400
        if age_days < SP500_CACHE_DAYS:
            with open(SP500_CACHE) as f:
                tickers = json.load(f)
            log.info(f"Using cached S&P 500 list ({len(tickers)} tickers, "
                     f"{age_days:.1f}d old)")
            return tickers

    # 2. Try each network source
    for src in SP500_SOURCES:
        try:
            log.info(f"Fetching S&P 500 from {src['name']}...")
            if src["method"] == "csv":
                tickers = _fetch_from_csv(src["url"])
            else:
                tickers = _fetch_from_html(src["url"])

            tickers = _sanitize(tickers)
            if len(tickers) < 400:
                log.warning(f"  {src['name']} returned only {len(tickers)} "
                            f"tickers — looks wrong, skipping")
                continue

            with open(SP500_CACHE, "w") as f:
                json.dump(tickers, f)
            log.info(f"  ok — cached {len(tickers)} tickers to {SP500_CACHE}")
            return tickers
        except Exception as e:
            log.warning(f"  {src['name']} failed: {e}")
            continue

    # 3. Stale cache
    if SP500_CACHE.exists():
        log.warning("All network sources failed — using stale cache")
        with open(SP500_CACHE) as f:
            return json.load(f)

    # 4. Hardcoded fallback
    log.warning(f"All network sources failed and no cache — using hardcoded "
                f"fallback ({len(SP500_FALLBACK)} large-cap names)")
    return SP500_FALLBACK


def compute_heat_scores(tickers: list, lookback_days: int = 20) -> list:
    """
    Batch-fetch ~2 months of data and compute heat scores.

    Returns a list of dicts sorted by heat (highest first):
        { ticker, close, pct_change, vol_ratio, heat, direction }
    """
    if not tickers:
        return []

    log.info(f"Fetching ~2mo of data for {len(tickers)} tickers (batched)...")
    df = yf.download(
        " ".join(tickers),
        period="2mo",
        progress=False,
        auto_adjust=True,
        group_by="ticker",
        threads=True,
    )

    results = []
    top_level = (set(df.columns.get_level_values(0))
                 if isinstance(df.columns, pd.MultiIndex) else set())

    for t in tickers:
        try:
            if t not in top_level:
                continue
            tdf = df[t]
            close = tdf["Close"].dropna()
            vol   = tdf["Volume"].dropna()
            if len(close) < lookback_days + 2 or len(vol) < lookback_days + 2:
                continue

            last_close = float(close.iloc[-1])
            prev_close = float(close.iloc[-2])
            pct_change = (last_close / prev_close - 1) * 100

            vol_today = float(vol.iloc[-1])
            vol_avg   = float(vol.iloc[-lookback_days-1:-1].mean())
            vol_ratio = vol_today / vol_avg if vol_avg > 0 else 1.0

            heat = vol_ratio * abs(pct_change)

            results.append({
                "ticker": t,
                "close": round(last_close, 2),
                "pct_change": round(pct_change, 2),
                "vol_ratio": round(vol_ratio, 2),
                "heat": round(heat, 2),
                "direction": "up" if pct_change > 0 else "down",
            })
        except Exception:
            continue

    results.sort(key=lambda x: x["heat"], reverse=True)
    return results


def top_movers(n: int = 15, min_price: float = 5.0) -> list:
    """Convenience: get top N heat movers from S&P 500, above min_price."""
    tickers = get_sp500_tickers()
    if not tickers:
        return []
    scores = compute_heat_scores(tickers)
    scores = [s for s in scores if s["close"] >= min_price]
    return scores[:n]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh", action="store_true",
                        help="Force-refresh the S&P 500 cache")
    parser.add_argument("--top", type=int, default=20,
                        help="How many top movers to show")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(message)s",
                        datefmt="%H:%M:%S")

    if args.refresh:
        get_sp500_tickers(force_refresh=True)

    movers = top_movers(args.top)
    if not movers:
        print("No movers found — check S&P 500 cache and network.")
        return

    print(f"\n  Top {len(movers)} S&P 500 heat movers:\n")
    print(f"  {'Ticker':<8} {'Close':>10} {'%Chg':>8} {'VolRatio':>10} "
          f"{'Heat':>8}  Dir")
    print(f"  {'-'*8} {'-'*10} {'-'*8} {'-'*10} {'-'*8}  {'-'*3}")
    for m in movers:
        arrow = "↑" if m["direction"] == "up" else "↓"
        print(f"  {m['ticker']:<8} ${m['close']:>9.2f} "
              f"{m['pct_change']:>+7.2f}% {m['vol_ratio']:>9.2f}x "
              f"{m['heat']:>7.2f}  {arrow}")
    print()


if __name__ == "__main__":
    main()