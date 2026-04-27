"""
Backtest a long-only trend-following / breakout strategy on Nifty 100 daily
data downloaded from Yahoo Finance.

Strategy
--------
Filter (all must hold on the signal day's CLOSE):
  1. SMA(150)  > EMA(220)
  2. Close     > SMA(50)
  3. SMA(50)   > SMA(150)
  4. Close     > 1.25 × 52-week low (Low)
  5. Low has dipped below EMA(220) at least once in the past 90 trading days

Entry trigger (in addition to the filter):
  - Close is a new 52-week high  (Close > previous 252-day max of Close)

Execution model:
  - Signals generated on day T's close.
  - Trades (entries AND exits) execute at day T+1's open.
  - Equal-weighted: 10% of current equity per position, max 10 positions.
  - Starting capital ₹100,000. No leverage, no shorting, integer-share sizing.

Exits (whichever fires first on a close):
  - Close < EMA(220)            -> reason "ema_break"
  - Close <= 0.85 × entry_price -> reason "stop_loss"

Outputs (in ./results):
  - trades.csv         every closed trade
  - equity_curve.csv   daily equity, cash, position count
  - metrics.json       summary stats
  - equity_curve.png   plot (if matplotlib is available)

Usage:
    python backtest_strategy.py
    python backtest_strategy.py --data-dir data --start 2020-01-01
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Strategy parameters — tweak here
# ---------------------------------------------------------------------------
INITIAL_CAPITAL = 100_000.0
MAX_POSITIONS = 10
ALLOC_PCT = 0.10           # 10% of current equity per position
STOP_LOSS_PCT = 0.85       # exit at -15% from entry

SMA_FAST = 50
SMA_MID = 150
EMA_SLOW = 220
LOOKBACK_52W = 252         # trading days
LOOKBACK_DIP = 90          # trading days
MIN_PCT_FROM_LOW = 1.25    # close > 1.25 × 52-week low

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("backtest")


# ===========================================================================
# Data loading and indicator computation
# ===========================================================================
def load_prices(data_dir: Path) -> Dict[str, pd.DataFrame]:
    """Read every CSV in data_dir, keyed by symbol (filename stem)."""
    prices: Dict[str, pd.DataFrame] = {}
    for path in sorted(data_dir.glob("*.csv")):
        sym = path.stem
        try:
            df = pd.read_csv(path, parse_dates=["Date"])
        except Exception as e:
            log.warning("Skipping %s (read error): %s", sym, e)
            continue
        if df.empty or "Close" not in df.columns:
            log.warning("Skipping %s (empty / no Close column)", sym)
            continue
        df = (df.sort_values("Date")
                .drop_duplicates(subset=["Date"])
                .set_index("Date"))
        for c in ("Open", "High", "Low", "Close", "Adj Close", "Volume"):
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna(subset=["Open", "High", "Low", "Close"])
        if df.empty:
            continue
        prices[sym] = df
    log.info("Loaded %d symbols", len(prices))
    return prices


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add SMAs, EMA, 52-week high/low, and the 'recently dipped' flag."""
    df = df.copy()
    df["SMA50"]  = df["Close"].rolling(SMA_FAST).mean()
    df["SMA150"] = df["Close"].rolling(SMA_MID).mean()
    df["EMA220"] = df["Close"].ewm(span=EMA_SLOW, adjust=False).mean()

    df["Low52W"]      = df["Low"].rolling(LOOKBACK_52W).min()
    df["PrevHigh52W"] = df["Close"].shift(1).rolling(LOOKBACK_52W).max()

    dipped = (df["Low"] < df["EMA220"]).astype(int)
    df["DippedRecently"] = (
        dipped.rolling(LOOKBACK_DIP).max().fillna(0).astype(bool)
    )
    return df


def compute_signals(df: pd.DataFrame) -> pd.Series:
    """Combined filter + breakout — True on rows where we'd open a position."""
    cond = (
        (df["SMA150"] > df["EMA220"])
        & (df["Close"]  > df["SMA50"])
        & (df["SMA50"]  > df["SMA150"])
        & (df["Close"]  > MIN_PCT_FROM_LOW * df["Low52W"])
        & df["DippedRecently"]
    )
    breakout = df["Close"] > df["PrevHigh52W"]
    return (cond & breakout).fillna(False)


# ===========================================================================
# Backtest engine
# ===========================================================================
@dataclass
class Position:
    symbol: str
    entry_date: pd.Timestamp
    entry_price: float
    shares: int


@dataclass
class Trade:
    symbol: str
    entry_date: pd.Timestamp
    entry_price: float
    exit_date: pd.Timestamp
    exit_price: float
    shares: int
    exit_reason: str

    @property
    def pnl(self) -> float:
        return (self.exit_price - self.entry_price) * self.shares

    @property
    def return_pct(self) -> float:
        return self.exit_price / self.entry_price - 1.0

    @property
    def days_held(self) -> int:
        return (self.exit_date - self.entry_date).days


def _safe_at(df: pd.DataFrame, date: pd.Timestamp, col: str) -> Optional[float]:
    """Return df.at[date, col] or None if missing/NaN."""
    if date not in df.index:
        return None
    val = df.at[date, col]
    if pd.isna(val):
        return None
    return float(val)


def run_backtest(
    prices: Dict[str, pd.DataFrame],
    start_date: Optional[pd.Timestamp] = None,
) -> Tuple[pd.DataFrame, List[Trade]]:
    """Walk the timeline day-by-day. Returns (equity_curve_df, trades)."""

    # Indicators + signals per symbol
    indicators = {sym: add_indicators(df) for sym, df in prices.items()}
    signals = {sym: compute_signals(df) for sym, df in indicators.items()}

    # Master timeline (union of all symbol dates)
    all_dates = pd.DatetimeIndex(
        sorted(set().union(*(df.index for df in indicators.values())))
    )
    if start_date is not None:
        all_dates = all_dates[all_dates >= pd.Timestamp(start_date)]

    cash = INITIAL_CAPITAL
    positions: Dict[str, Position] = {}
    trades: List[Trade] = []
    equity_rows: List[dict] = []
    pending_entries: List[str] = []           # signaled yesterday
    pending_exits: List[Tuple[str, str]] = [] # (symbol, reason)

    for today in all_dates:

        # 1) Execute pending exits at today's open
        for sym, reason in pending_exits:
            if sym not in positions:
                continue
            pos = positions[sym]
            df = indicators.get(sym)
            if df is None:
                continue
            open_px = _safe_at(df, today, "Open")
            if open_px is None:
                continue  # symbol didn't trade — try again next day
            cash += pos.shares * open_px
            trades.append(Trade(
                symbol=sym,
                entry_date=pos.entry_date,
                entry_price=pos.entry_price,
                exit_date=today,
                exit_price=open_px,
                shares=pos.shares,
                exit_reason=reason,
            ))
            del positions[sym]
        pending_exits = [
            (s, r) for s, r in pending_exits if s in positions
        ]  # keep ones we couldn't execute today

        # 2) Execute pending entries at today's open
        # Compute current equity first (for sizing) — value held positions at today's open
        held_value_at_open = 0.0
        for sym, pos in positions.items():
            df = indicators[sym]
            px = _safe_at(df, today, "Open") or pos.entry_price
            held_value_at_open += pos.shares * px
        equity_for_sizing = cash + held_value_at_open

        for sym in pending_entries:
            if sym in positions or len(positions) >= MAX_POSITIONS:
                continue
            df = indicators.get(sym)
            if df is None:
                continue
            open_px = _safe_at(df, today, "Open")
            if open_px is None or open_px <= 0:
                continue
            alloc = equity_for_sizing * ALLOC_PCT
            shares = int(alloc // open_px)
            cost = shares * open_px
            if shares <= 0 or cost > cash:
                continue
            cash -= cost
            positions[sym] = Position(
                symbol=sym,
                entry_date=today,
                entry_price=open_px,
                shares=shares,
            )
        pending_entries = []

        # 3) Mark-to-market at today's close → record equity
        position_value = 0.0
        for sym, pos in positions.items():
            df = indicators[sym]
            close_px = _safe_at(df, today, "Close") or pos.entry_price
            position_value += pos.shares * close_px
        equity = cash + position_value
        equity_rows.append({
            "Date": today,
            "Equity": equity,
            "Cash": cash,
            "Positions": len(positions),
        })

        # 4) Generate exit signals on today's close (execute next day's open)
        for sym, pos in list(positions.items()):
            df = indicators[sym]
            close_px = _safe_at(df, today, "Close")
            ema220 = _safe_at(df, today, "EMA220")
            if close_px is None or ema220 is None:
                continue
            reason: Optional[str] = None
            if close_px <= pos.entry_price * STOP_LOSS_PCT:
                reason = "stop_loss"
            elif close_px < ema220:
                reason = "ema_break"
            if reason and not any(s == sym for s, _ in pending_exits):
                pending_exits.append((sym, reason))

        # 5) Generate entry signals on today's close
        exiting_today = {s for s, _ in pending_exits}
        candidates: List[Tuple[str, float]] = []
        for sym, sig in signals.items():
            if sym in positions or sym in exiting_today:
                continue  # no same-day exit-and-reenter
            if today not in sig.index or not sig.at[today]:
                continue
            df = indicators[sym]
            close_px = _safe_at(df, today, "Close")
            low52w   = _safe_at(df, today, "Low52W")
            if close_px is None or low52w is None or low52w <= 0:
                continue
            score = close_px / low52w  # strength of move off lows
            candidates.append((sym, score))

        # Limit by available slots (accounting for upcoming exits)
        slots_left = MAX_POSITIONS - len(positions) + len(pending_exits)
        if slots_left > 0 and candidates:
            candidates.sort(key=lambda x: -x[1])
            for sym, _ in candidates[:slots_left]:
                pending_entries.append(sym)

    equity_df = pd.DataFrame(equity_rows).set_index("Date")
    return equity_df, trades


# ===========================================================================
# Performance metrics
# ===========================================================================
def compute_metrics(equity: pd.DataFrame, trades: List[Trade]) -> dict:
    eq = equity["Equity"]
    if len(eq) < 2:
        return {"error": "not enough equity points"}

    total_return = eq.iloc[-1] / eq.iloc[0] - 1
    days = (eq.index[-1] - eq.index[0]).days
    years = max(days / 365.25, 1e-9)
    cagr = (eq.iloc[-1] / eq.iloc[0]) ** (1 / years) - 1

    rolling_max = eq.cummax()
    drawdown = eq / rolling_max - 1
    max_dd = drawdown.min()

    daily_ret = eq.pct_change().dropna()
    sharpe = (
        np.sqrt(252) * daily_ret.mean() / daily_ret.std()
        if daily_ret.std() > 0 else 0.0
    )

    n = len(trades)
    if n:
        wins   = [t for t in trades if t.pnl > 0]
        losses = [t for t in trades if t.pnl <= 0]
        win_rate = len(wins) / n
        avg_win  = float(np.mean([t.return_pct for t in wins]))   if wins   else 0.0
        avg_loss = float(np.mean([t.return_pct for t in losses])) if losses else 0.0
        avg_held = float(np.mean([t.days_held  for t in trades]))
        # Profit factor = sum of wins / |sum of losses|
        gross_win  = sum(t.pnl for t in wins)
        gross_loss = -sum(t.pnl for t in losses)
        profit_factor = gross_win / gross_loss if gross_loss > 0 else float("inf")
    else:
        win_rate = avg_win = avg_loss = avg_held = 0.0
        profit_factor = 0.0

    return {
        "InitialEquity":  round(float(eq.iloc[0]), 2),
        "FinalEquity":    round(float(eq.iloc[-1]), 2),
        "TotalReturn%":   round(total_return * 100, 2),
        "CAGR%":          round(cagr * 100, 2),
        "MaxDrawdown%":   round(max_dd * 100, 2),
        "Sharpe":         round(float(sharpe), 2),
        "Trades":         n,
        "WinRate%":       round(win_rate * 100, 2),
        "AvgWin%":        round(avg_win * 100, 2),
        "AvgLoss%":       round(avg_loss * 100, 2),
        "ProfitFactor":   round(profit_factor, 2) if profit_factor != float("inf") else "inf",
        "AvgDaysHeld":    round(avg_held, 1),
        "PeriodStart":    str(eq.index[0].date()),
        "PeriodEnd":      str(eq.index[-1].date()),
    }


# ===========================================================================
# Output writing
# ===========================================================================
def save_outputs(
    out_dir: Path,
    equity: pd.DataFrame,
    trades: List[Trade],
    metrics: dict,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    equity.to_csv(out_dir / "equity_curve.csv")

    trade_rows = [{
        "Symbol":      t.symbol,
        "EntryDate":   t.entry_date.date(),
        "EntryPrice":  round(t.entry_price, 2),
        "ExitDate":    t.exit_date.date(),
        "ExitPrice":   round(t.exit_price, 2),
        "Shares":      t.shares,
        "PnL":         round(t.pnl, 2),
        "Return%":     round(t.return_pct * 100, 2),
        "DaysHeld":    t.days_held,
        "ExitReason":  t.exit_reason,
    } for t in trades]
    pd.DataFrame(trade_rows).to_csv(out_dir / "trades.csv", index=False)

    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, default=str)

    # Optional plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(11, 7), sharex=True,
            gridspec_kw={"height_ratios": [3, 1]},
        )
        ax1.plot(equity.index, equity["Equity"], color="#1f77b4", linewidth=1.5)
        ax1.set_title("Equity curve")
        ax1.set_ylabel("Equity (₹)")
        ax1.grid(alpha=0.3)

        rolling_max = equity["Equity"].cummax()
        dd = (equity["Equity"] / rolling_max - 1) * 100
        ax2.fill_between(equity.index, dd, 0, color="#d62728", alpha=0.4)
        ax2.set_ylabel("Drawdown (%)")
        ax2.grid(alpha=0.3)

        fig.tight_layout()
        fig.savefig(out_dir / "equity_curve.png", dpi=120)
        plt.close(fig)
    except ImportError:
        log.info("matplotlib not installed — skipping plot")


# ===========================================================================
# Entry point
# ===========================================================================
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"),
                        help="Folder with per-symbol CSVs (default: data)")
    parser.add_argument("--results-dir", type=Path, default=Path("results"),
                        help="Where to write outputs (default: results)")
    parser.add_argument("--start", type=str, default=None,
                        help="Start the backtest on or after this date (YYYY-MM-DD)")
    args = parser.parse_args()

    if not args.data_dir.exists():
        raise SystemExit(f"Data directory not found: {args.data_dir.resolve()}")

    prices = load_prices(args.data_dir)
    if not prices:
        raise SystemExit("No usable CSVs found.")

    start = pd.Timestamp(args.start) if args.start else None
    log.info("Running backtest…")
    equity, trades = run_backtest(prices, start_date=start)

    metrics = compute_metrics(equity, trades)
    save_outputs(args.results_dir, equity, trades, metrics)

    log.info("=" * 60)
    log.info("Backtest complete. Summary:")
    for k, v in metrics.items():
        log.info("  %-16s %s", k, v)
    log.info("Outputs written to %s/", args.results_dir)


if __name__ == "__main__":
    main()