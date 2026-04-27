"""
End-to-end smoke test: generate synthetic trending stocks, write them to a
temp data folder, run the backtest, assert it produces sensible output.
"""
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "/home/claude")
import backtest_strategy as bt


def make_synth_stock(seed: int, n_days: int = 1500, start_price: float = 100.0) -> pd.DataFrame:
    """A stock with multiple shallow dips during an overall uptrend, ending in
    a sharp drop. Designed so multiple breakouts AND exits will fire."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2018-01-02", periods=n_days)

    # Mild positive drift + Gaussian noise for the bulk of the series
    drift_per_day = 0.0006   # ~15% annualised
    daily_returns = rng.normal(drift_per_day, 0.012, n_days)

    # Inject 4 short, shallow pullbacks (~8% over ~25 days) at known points.
    # Each is small enough that recovery to new highs within 90 days is feasible.
    pullback_starts = [350, 700, 1050, 1350]
    for s in pullback_starts:
        if s + 25 < n_days:
            daily_returns[s:s + 25] += np.linspace(-0.005, -0.001, 25)
            daily_returns[s + 25:s + 50] += np.linspace(0.002, 0.0, 25)

    # Final crash to force exits
    daily_returns[-60:] = rng.normal(-0.010, 0.012, 60)

    close = start_price * np.exp(np.cumsum(daily_returns))
    close = np.maximum(close, 5.0)

    open_ = close * (1 + rng.normal(0, 0.003, n_days))
    high  = np.maximum(close, open_) * (1 + np.abs(rng.normal(0, 0.005, n_days)))
    low   = np.minimum(close, open_) * (1 - np.abs(rng.normal(0, 0.005, n_days)))
    vol   = rng.integers(1_000_000, 10_000_000, n_days)

    return pd.DataFrame({
        "Date":   dates,
        "Open":   open_.round(2),
        "High":   high.round(2),
        "Low":    low.round(2),
        "Close":  close.round(2),
        "Adj Close": close.round(2),
        "Volume": vol,
    })


def main() -> None:
    tmp = Path(tempfile.mkdtemp())
    data_dir = tmp / "data"; data_dir.mkdir()
    results_dir = tmp / "results"

    # 8 synthetic stocks, different seeds → different paths
    for i in range(8):
        df = make_synth_stock(seed=42 + i)
        df.to_csv(data_dir / f"SYN{i:02d}.csv", index=False)

    prices = bt.load_prices(data_dir)
    print(f"Loaded synthetic stocks: {len(prices)}")

    equity, trades = bt.run_backtest(prices)
    metrics = bt.compute_metrics(equity, trades)
    bt.save_outputs(results_dir, equity, trades, metrics)

    print("\nMetrics on synthetic data:")
    for k, v in metrics.items():
        print(f"  {k:<16} {v}")

    # Sanity assertions
    assert len(equity) > 0, "no equity points"
    assert metrics["FinalEquity"] > 0, "negative or zero final equity"
    assert isinstance(metrics["Trades"], int), "bad trade count"
    print(f"\nFiles written: {sorted(p.name for p in results_dir.iterdir())}")

    if trades:
        print(f"\nFirst 5 trades:")
        first5 = pd.read_csv(results_dir / "trades.csv").head()
        print(first5.to_string(index=False))

    shutil.rmtree(tmp)
    print("\nSmoke test passed ✓")


if __name__ == "__main__":
    main()