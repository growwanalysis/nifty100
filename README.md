<div align="center">

# 📈 Nifty 100 Strategy Scanner

**A live trend-following & breakout scanner + backtester for India's top 100 NSE stocks**

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://nifty100.streamlit.app)
&nbsp;
![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=flat&logo=plotly&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=flat)

</div>

---

## 🌐 Live App

> **Try it now →** [nifty100.streamlit.app](https://nifty100.streamlit.app)

No setup required. The app scans all 100 Nifty stocks in real time and identifies breakout candidates using a rules-based trend-following strategy.

---

## 🧠 The Strategy

This is a **long-only trend-following + breakout** system. A stock must satisfy **all 5 filters** on the signal day's close, then trigger a breakout entry.

### ✅ Entry Filters (all must hold)

| # | Filter | Logic |
|---|--------|-------|
| F1 | Trend aligned | `SMA(150) > EMA(220)` |
| F2 | Price above fast MA | `Close > SMA(50)` |
| F3 | MAs stacked | `SMA(50) > SMA(150)` |
| F4 | Distance from lows | `Close > 1.25 × 52-week Low` |
| F5 | Recent pullback | `Low dipped below EMA(220)` at least once in the past **90 trading days** |

### 🎯 Entry Trigger

```
Close > previous 252-day max of Close   →   new 52-week closing high
```

Signals fire on **day T's close**. Trades execute at **day T+1's open**.

### 🚪 Exit Rules (whichever fires first)

| Exit | Condition | Reason |
|------|-----------|--------|
| Trend break | `Close < EMA(220)` | `ema_break` |
| Stop loss | `Close ≤ 0.85 × Entry price` | `stop_loss` (-15%) |

### 💼 Position Sizing

- **Equal-weight**: 10% of current equity per position
- **Max open positions**: 10 simultaneously
- **Starting capital**: ₹1,00,000
- **No leverage, no shorting, integer share sizing**

---

## 📊 Backtest Results (2020 – Apr 2026)

> Backtested on 100 Nifty stocks using local CSV data via `backtest_strategy.py`

| Metric | Value |
|--------|-------|
| 📅 Period | Jan 2020 – Apr 2026 |
| 💰 Initial Equity | ₹1,00,000 |
| 💵 Final Equity | ₹3,03,125 |
| 📈 Total Return | **+203.13%** |
| 🚀 CAGR | **19.21%** |
| 📉 Max Drawdown | -19.84% |
| ⚖️ Sharpe Ratio | **1.22** |
| 🔁 Total Trades | 48 |
| 🏆 Win Rate | 50.0% |
| 💹 Avg Win | +67.73% |
| 🛑 Avg Loss | -8.44% |
| ⚡ Profit Factor | **5.0** |
| 🕐 Avg Days Held | 316.4 days |

> ₹1 lakh grew to ₹3 lakh+ with a **profit factor of 5** — meaning for every ₹1 lost, the strategy made ₹5 in winners.

---

## 🖥️ Dashboard Features

The Streamlit dashboard has **4 tabs**:

### 🎯 Tab 1 — Active Signals
Stocks passing **all 5 filters** AND making a **new 52-week closing high today**.
These are actionable: enter at tomorrow's open per the strategy rules.
Includes a one-click **CSV download** of signals with date stamp.

### 👁️ Tab 2 — Watchlist
Stocks passing **all 5 filters** but **not yet at a new high**.
Sorted by closeness to breakout (`%ToBreakout`) — top of the list may trigger soon.

### 🪜 Tab 3 — Filter Funnel
An interactive **Plotly funnel chart** showing how many stocks survive each successive filter.
Also shows a per-filter pass-rate table. Useful for reading the market regime:
- Strong bull market → F1–F3 pass wide
- Choppy/bear → F4 & F5 tighten the cone significantly

### 🔍 Tab 4 — Stock Detail
Pick any Nifty 100 stock to see:
- **Candlestick chart** (green/red) with configurable lookback (120 / 252 / 504 bars)
- **SMA50** (amber), **SMA150** (blue), **EMA220** (purple) overlaid
- **Previous 52W high** dotted line (the breakout trigger level)
- Live metrics: Close, `%Chg`, `%FromLow52W`, `%FromHigh52W`, filters passed badge

---

## 📂 Project Structure

```
nifty100/
├── dashboard.py               # Streamlit live scanner app
├── backtest_strategy.py       # Historical backtest engine
├── download_nifty100_data.py  # Bulk data downloader (yfinance → CSV)
├── nifty100_symbols.csv       # 100 NSE ticker symbols
├── requirements.txt           # Python dependencies
│
├── data/                      # Pre-downloaded OHLCV CSVs (100 stocks)
│   ├── RELIANCE.csv
│   ├── HDFCBANK.csv
│   ├── TCS.csv
│   └── ... (97 more)
│
└── results/                   # Backtest outputs
    ├── trades.csv             # Every closed trade
    ├── equity_curve.csv       # Daily equity, cash, position count
    ├── equity_curve.png       # Equity curve chart
    └── metrics.json           # Summary stats
```

---

## 🚀 Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/growwanalysis/nifty100.git
cd nifty100
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

> Requirements: `streamlit`, `yfinance`, `pandas`, `numpy`, `plotly`

### 3. (Optional) Refresh stock data

```bash
python download_nifty100_data.py
```

Data is cached as CSVs in `data/`. The dashboard also fetches live via yfinance (cached for 10 min).

### 4. Launch the dashboard

```bash
streamlit run dashboard.py
```

Opens at `http://localhost:8501`. Use the sidebar to switch between NSE (`.NS`) and BSE (`.BO`) suffixes, set history window, and force-refresh data.

### 5. Run the backtest

```bash
python backtest_strategy.py
# Or with custom date range:
python backtest_strategy.py --data-dir data --start 2020-01-01
```

Results saved to `results/` — trades, equity curve CSV/PNG, and metrics JSON.

---

## ⚙️ Sidebar Settings

| Setting | Default | Description |
|---------|---------|-------------|
| Symbols CSV | `nifty100_symbols.csv` | Path to the ticker list |
| Exchange suffix | `.NS` | `.NS` = NSE, `.BO` = BSE |
| History window | `2y` | Must cover 252-day warm-up period |
| 🔄 Refresh | — | Clears cache and re-fetches all data |

Data is cached for **10 minutes** by default to avoid Yahoo Finance rate limits.

---

## ⚠️ Disclaimer

> This tool is built **for educational and research purposes only**. It is **not financial advice**. Past backtest performance does not guarantee future results. Always do your own research before making investment decisions.

---

## 🙌 Contributing

1. Fork this repo
2. Create a branch: `git checkout -b feature/my-feature`
3. Commit: `git commit -m "Add my feature"`
4. Push & open a Pull Request

---

<div align="center">

Made with ❤️ by [growwanalysis](https://github.com/growwanalysis)

⭐ **Star this repo if it helped you!**

</div>
