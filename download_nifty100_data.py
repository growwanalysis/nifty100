"""
Download Nifty 100 daily OHLCV data from Yahoo Finance.

What it does
------------
- Reads tickers from `nifty100_symbols.csv` (must have a `Symbol` column,
  or a single column of symbols).
- Appends the `.NS` suffix that Yahoo Finance uses for NSE listings.
- Downloads daily OHLCV from 2020-01-01 up to today.
- Writes one CSV per symbol into `./data/<SYMBOL>.csv`.
- Incremental: if a CSV already exists, only fetches rows after the last
  date already saved, then appends and de-duplicates. Re-running is cheap.
- Parallelised with a thread pool, with retries and a per-symbol log.

Usage
-----
    pip install yfinance pandas
    python download_nifty100.py

Outputs
-------
    data/<SYMBOL>.csv          one file per ticker
    download_log.txt           run log (successes + failures)
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf

# ---------------------------------------------------------------------------
# Configuration — tweak these if you need to
# ---------------------------------------------------------------------------
SYMBOLS_FILE = Path("nifty100_symbols.csv")
DATA_DIR = Path("data")
LOG_FILE = Path("download_log.txt")

START_DATE = "2020-01-01"
# yfinance's `end` is exclusive, so add a day to include today's bar
# once it's published.
END_DATE = (date.today() + timedelta(days=1)).isoformat()

# NSE symbols on Yahoo Finance carry a ".NS" suffix. Use ".BO" for BSE.
YF_SUFFIX = ".NS"

# Yahoo tolerates some parallelism; 5–8 workers is a safe sweet spot.
MAX_WORKERS = 6
RETRIES = 3
RETRY_BACKOFF_SECS = 3

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(LOG_FILE, mode="w"), logging.StreamHandler()],
)
log = logging.getLogger("nifty100")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_symbols(path: Path) -> list[str]:
    """Read tickers from the CSV. Accepts a `Symbol` column or a single column."""
    df = pd.read_csv(path)
    col = "Symbol" if "Symbol" in df.columns else df.columns[0]
    symbols = [str(s).strip() for s in df[col].dropna() if str(s).strip()]
    # Deduplicate while preserving order.
    seen, out = set(), []
    for s in symbols:
        if s not in seen:
            seen.add(s)
            out.append(s)
    log.info("Loaded %d symbols from %s", len(out), path)
    return out


def output_path(symbol: str) -> Path:
    return DATA_DIR / f"{symbol}.csv"


def last_date_in_csv(path: Path) -> date | None:
    """Return the latest `Date` in an existing CSV, or None on any issue."""
    try:
        existing = pd.read_csv(path, parse_dates=["Date"])
        if existing.empty:
            return None
        return existing["Date"].max().date()
    except Exception:
        return None


def fetch_one(symbol: str) -> tuple[str, str, int]:
    """
    Download (or incrementally update) a single symbol.
    Returns (symbol, status, rows_in_file_after_save).
    """
    yf_symbol = symbol + YF_SUFFIX
    out = output_path(symbol)

    last = last_date_in_csv(out) if out.exists() else None
    start = (last + timedelta(days=1)).isoformat() if last else START_DATE

    # Already current — skip the network call.
    if last and start >= END_DATE:
        return symbol, "up-to-date", 0

    last_err: Exception | None = None
    for attempt in range(1, RETRIES + 1):
        try:
            df = yf.download(
                yf_symbol,
                start=start,
                end=END_DATE,
                interval="1d",
                auto_adjust=False,
                progress=False,
                threads=False,
            )
            if df is None or df.empty:
                return symbol, "no-data", 0

            # yfinance sometimes returns a MultiIndex column header even for a
            # single ticker — flatten it so the CSV is clean.
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df.index.name = "Date"
            df = df.reset_index()

            # Merge with existing data on incremental runs.
            if last and out.exists():
                old = pd.read_csv(out, parse_dates=["Date"])
                df = (
                    pd.concat([old, df], ignore_index=True)
                    .drop_duplicates(subset=["Date"], keep="last")
                    .sort_values("Date")
                    .reset_index(drop=True)
                )

            DATA_DIR.mkdir(parents=True, exist_ok=True)
            df.to_csv(out, index=False)
            return symbol, "updated" if last else "fetched", len(df)

        except Exception as e:  # network / Yahoo hiccups — retry
            last_err = e
            if attempt < RETRIES:
                time.sleep(RETRY_BACKOFF_SECS * attempt)

    return symbol, f"failed: {last_err}", 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    if not SYMBOLS_FILE.exists():
        raise SystemExit(
            f"Symbols file not found: {SYMBOLS_FILE.resolve()}. "
            "Place nifty100_symbols.csv next to this script."
        )

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    symbols = load_symbols(SYMBOLS_FILE)

    log.info(
        "Daily download window %s → %s | %d symbols | suffix '%s' | %d workers",
        START_DATE, END_DATE, len(symbols), YF_SUFFIX, MAX_WORKERS,
    )

    fetched, updated, up_to_date, no_data, failures = [], [], [], [], []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(fetch_one, s): s for s in symbols}
        for i, fut in enumerate(as_completed(futures), 1):
            sym, status, rows = fut.result()
            tag = "OK " if status in {"fetched", "updated", "up-to-date"} else "ERR"
            log.info("[%3d/%d] %s %-15s %-12s rows=%d",
                     i, len(symbols), tag, sym, status, rows)

            if status == "fetched":
                fetched.append(sym)
            elif status == "updated":
                updated.append(sym)
            elif status == "up-to-date":
                up_to_date.append(sym)
            elif status == "no-data":
                no_data.append(sym)
            else:
                failures.append((sym, status))

    log.info("=" * 70)
    log.info(
        "Done — fetched=%d, updated=%d, up-to-date=%d, no-data=%d, failed=%d",
        len(fetched), len(updated), len(up_to_date), len(no_data), len(failures),
    )
    if no_data:
        log.warning("No data returned: %s", ", ".join(no_data))
    if failures:
        log.warning("Failed symbols:")
        for sym, reason in failures:
            log.warning("  %s — %s", sym, reason)


if __name__ == "__main__":
    main()