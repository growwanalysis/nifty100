"""
Live Nifty 100 strategy scanner — Streamlit dashboard.

Same strategy filters as the backtest:
  1. SMA(150)  > EMA(220)
  2. Close     > SMA(50)
  3. SMA(50)   > SMA(150)
  4. Close     > 1.25 × 52-week low
  5. Low has dipped below EMA(220) at least once in the past 90 trading days
  + Breakout trigger: Close > previous 252-day max of Close (new 52w high)

Run:
    pip install streamlit yfinance pandas numpy plotly
    streamlit run dashboard.py

Open the URL it prints (usually http://localhost:8501).
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

# ---------------------------------------------------------------------------
# Strategy constants — mirror the backtest
# ---------------------------------------------------------------------------
SMA_FAST = 50
SMA_MID = 150
EMA_SLOW = 220
LOOKBACK_52W = 252
LOOKBACK_DIP = 90
MIN_PCT_FROM_LOW = 1.25

DEFAULT_SYMBOLS_FILE = "nifty100_symbols.csv"
HISTORY_PERIOD = "2y"   # need ~12 months for 252-day high + warm-up
CACHE_TTL_SECS = 600    # 10 minutes — balance freshness vs. Yahoo rate limits

IST = ZoneInfo("Asia/Kolkata")

# ---------------------------------------------------------------------------
# Page config + light styling
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Nifty 100 Strategy Scanner",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        .block-container {padding-top: 2rem;}
        [data-testid="stMetricValue"] {font-size: 1.8rem;}
        .small-caption {color: #888; font-size: 0.85rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


# ===========================================================================
# Data loading
# ===========================================================================
@st.cache_data(ttl=CACHE_TTL_SECS, show_spinner=False)
def load_symbols(path: str) -> list[str]:
    df = pd.read_csv(path)
    col = "Symbol" if "Symbol" in df.columns else df.columns[0]
    seen, out = set(), []
    for s in df[col].dropna().astype(str):
        s = s.strip()
        if s and s not in seen:
            seen.add(s); out.append(s)
    return out


@st.cache_data(ttl=CACHE_TTL_SECS, show_spinner=False)
def fetch_bulk(symbols: tuple[str, ...], suffix: str, period: str) -> dict[str, pd.DataFrame]:
    """One bulk download → dict[symbol] = OHLCV DataFrame.

    Bulk is dramatically faster than per-symbol calls: ~5–10 seconds for 100
    tickers vs. several minutes sequentially. Tuple input so the cache key is
    hashable.
    """
    yf_symbols = [s + suffix for s in symbols]
    raw = yf.download(
        tickers=yf_symbols,
        period=period,
        interval="1d",
        group_by="ticker",
        auto_adjust=False,
        progress=False,
        threads=True,
    )

    out: dict[str, pd.DataFrame] = {}
    for sym in symbols:
        ys = sym + suffix
        try:
            df = raw[ys].copy() if isinstance(raw.columns, pd.MultiIndex) else raw.copy()
        except KeyError:
            continue
        df = df.dropna(subset=["Close"])
        if df.empty:
            continue
        for c in ("Open", "High", "Low", "Close", "Volume"):
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna(subset=["Open", "High", "Low", "Close"])
        if not df.empty:
            out[sym] = df
    return out


# ===========================================================================
# Indicators + per-stock evaluation
# ===========================================================================
def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["SMA50"]       = df["Close"].rolling(SMA_FAST).mean()
    df["SMA150"]      = df["Close"].rolling(SMA_MID).mean()
    df["EMA220"]      = df["Close"].ewm(span=EMA_SLOW, adjust=False).mean()
    df["High52W"]     = df["Close"].rolling(LOOKBACK_52W).max()
    df["Low52W"]      = df["Low"].rolling(LOOKBACK_52W).min()
    df["PrevHigh52W"] = df["Close"].shift(1).rolling(LOOKBACK_52W).max()

    dipped = (df["Low"] < df["EMA220"]).astype(int)
    df["DippedRecently"] = (
        dipped.rolling(LOOKBACK_DIP).max().fillna(0).astype(bool)
    )
    return df


def evaluate_latest(sym: str, df: pd.DataFrame) -> dict | None:
    """Return a row of stats + filter outcomes for the most recent bar."""
    if len(df) < LOOKBACK_52W + 5:
        return None  # not enough history for the longest indicator
    df = add_indicators(df)
    row = df.iloc[-1]
    prev = df.iloc[-2] if len(df) >= 2 else row

    needed = ["SMA50", "SMA150", "EMA220", "High52W", "Low52W", "PrevHigh52W"]
    if any(pd.isna(row[c]) for c in needed):
        return None

    f1 = row["SMA150"] > row["EMA220"]
    f2 = row["Close"]  > row["SMA50"]
    f3 = row["SMA50"]  > row["SMA150"]
    f4 = row["Close"]  > MIN_PCT_FROM_LOW * row["Low52W"]
    f5 = bool(row["DippedRecently"])
    breakout = row["Close"] > row["PrevHigh52W"]
    all_filters = bool(f1 and f2 and f3 and f4 and f5)

    pct_change = (row["Close"] / prev["Close"] - 1) * 100 if prev["Close"] else 0.0

    return {
        "Symbol":       sym,
        "Date":         df.index[-1],
        "Close":        float(row["Close"]),
        "Open":         float(row["Open"]),
        "High":         float(row["High"]),
        "Low":          float(row["Low"]),
        "Volume":       int(row["Volume"]) if pd.notna(row["Volume"]) else 0,
        "%Chg":         float(pct_change),
        "SMA50":        float(row["SMA50"]),
        "SMA150":       float(row["SMA150"]),
        "EMA220":       float(row["EMA220"]),
        "High52W":      float(row["High52W"]),
        "Low52W":       float(row["Low52W"]),
        "PrevHigh52W":  float(row["PrevHigh52W"]),
        "%FromLow52W":  (row["Close"] / row["Low52W"] - 1) * 100,
        "%FromHigh52W": (row["Close"] / row["High52W"] - 1) * 100,
        "%ToBreakout":  (row["PrevHigh52W"] / row["Close"] - 1) * 100,
        "F1": bool(f1), "F2": bool(f2), "F3": bool(f3),
        "F4": bool(f4), "F5": bool(f5),
        "Breakout":     bool(breakout),
        "AllFilters":   all_filters,
        "FullSignal":   all_filters and bool(breakout),
        "FilterCount":  int(f1) + int(f2) + int(f3) + int(f4) + int(f5),
    }


def build_scan(prices: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = [r for sym, df in prices.items() if (r := evaluate_latest(sym, df))]
    return pd.DataFrame(rows)


# ===========================================================================
# UI helpers
# ===========================================================================
def fmt_inr(x: float) -> str:
    return f"₹{x:,.2f}"


def candidate_table(df: pd.DataFrame, columns: list[str]) -> None:
    if df.empty:
        st.info("No stocks match this view right now.")
        return
    show = df[columns].copy()
    # Round the floats for display
    for c in show.select_dtypes(include="float").columns:
        show[c] = show[c].round(2)
    st.dataframe(
        show,
        use_container_width=True,
        hide_index=True,
        height=min(60 + 35 * len(show), 600),
    )


def filter_funnel_chart(df: pd.DataFrame) -> go.Figure:
    """Cumulative funnel — how many stocks survive each successive filter."""
    flags = df[["F1", "F2", "F3", "F4", "F5"]].astype(bool)
    counts = [
        ("Universe",                len(df)),
        ("F1: SMA150>EMA220",        int(flags["F1"].sum())),
        ("F1+F2: Close>SMA50",       int((flags["F1"] & flags["F2"]).sum())),
        ("F1-3: SMA50>SMA150",       int((flags[["F1","F2","F3"]].all(1)).sum())),
        ("F1-4: Close>1.25×Low52W",  int((flags[["F1","F2","F3","F4"]].all(1)).sum())),
        ("F1-5: All filters",        int(flags.all(1).sum())),
        ("+ Breakout",               int((flags.all(1) & df["Breakout"]).sum())),
    ]
    fig = go.Figure(go.Funnel(
        y=[c[0] for c in counts],
        x=[c[1] for c in counts],
        textposition="inside",
        textinfo="value+percent initial",
        marker={"color": ["#2E86AB", "#3F8FCC", "#5298E0", "#65A1EE",
                          "#7AAAEE", "#8FB3EE", "#22C55E"]},
    ))
    fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=380)
    return fig


def detail_chart(symbol: str, df: pd.DataFrame, lookback: int = 252) -> go.Figure:
    df = add_indicators(df).iloc[-lookback:]
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"], low=df["Low"],
        close=df["Close"], name=symbol,
        increasing_line_color="#22C55E", decreasing_line_color="#EF4444",
    ))
    for col, color in [("SMA50", "#F59E0B"), ("SMA150", "#3B82F6"), ("EMA220", "#A855F7")]:
        fig.add_trace(go.Scatter(
            x=df.index, y=df[col], mode="lines", name=col,
            line=dict(color=color, width=1.4),
        ))
    fig.add_trace(go.Scatter(
        x=df.index, y=df["PrevHigh52W"], mode="lines", name="Prev 52W High",
        line=dict(color="#94a3b8", width=1, dash="dot"),
    ))
    fig.update_layout(
        height=520, margin=dict(l=10, r=10, t=30, b=10),
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", y=1.05, x=0),
        hovermode="x unified",
    )
    return fig


# ===========================================================================
# Main
# ===========================================================================
def main() -> None:
    # ---- Sidebar -----------------------------------------------------------
    with st.sidebar:
        st.markdown("### Settings")
        symbols_path = st.text_input("Symbols CSV", DEFAULT_SYMBOLS_FILE)
        suffix = st.selectbox("Exchange suffix", [".NS", ".BO"], index=0,
                              help=".NS = NSE, .BO = BSE")
        period = st.selectbox("History window", ["1y", "2y", "3y", "5y"], index=1,
                              help="2y is enough for the 252-day 52w window + warm-up")
        st.markdown("---")
        if st.button("🔄 Refresh now", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
        st.caption(f"Data cached for {CACHE_TTL_SECS // 60} min — refresh forces a re-fetch.")

        st.markdown("---")
        st.markdown("**Strategy**")
        st.caption(
            "Filters: SMA150>EMA220 · Close>SMA50 · SMA50>SMA150 · "
            "Close>1.25×Low52W · Dipped below EMA220 in last 90d. "
            "Trigger: new 52-week closing high."
        )

    # ---- Header ------------------------------------------------------------
    st.title("📈 Nifty 100 Strategy Scanner")

    # ---- Load + scan -------------------------------------------------------
    if not Path(symbols_path).exists():
        st.error(f"Symbols file not found: `{symbols_path}`. Place "
                 "`nifty100_symbols.csv` next to this script or update the path "
                 "in the sidebar.")
        return

    symbols = load_symbols(symbols_path)
    with st.spinner(f"Fetching {len(symbols)} symbols from Yahoo Finance…"):
        prices = fetch_bulk(tuple(symbols), suffix, period)

    if not prices:
        st.error("Yahoo Finance returned no data. Try clicking Refresh, or "
                 "check whether you've hit a rate limit.")
        return

    scan = build_scan(prices)
    if scan.empty:
        st.warning("No symbols had enough history to evaluate (need ≥ 252 trading days).")
        return

    # ---- KPIs --------------------------------------------------------------
    latest_data_date = scan["Date"].max()
    pass_all = int(scan["AllFilters"].sum())
    full_signals = int(scan["FullSignal"].sum())
    fetched_at = datetime.now(IST).strftime("%Y-%m-%d %H:%M IST")

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Symbols scanned", len(scan))
    k2.metric("Pass all 5 filters", pass_all,
              delta=f"{pass_all/len(scan)*100:.0f}% of universe")
    k3.metric("🎯 Active signals", full_signals,
              delta="passing all + breakout today" if full_signals else None)
    k4.metric("Latest bar", latest_data_date.strftime("%Y-%m-%d"))
    k5.metric("Fetched", fetched_at.split(" ")[1])

    st.markdown(
        f"<span class='small-caption'>Bulk-downloaded {len(prices)} of "
        f"{len(symbols)} symbols · last bar {latest_data_date.date()} · "
        f"fetched {fetched_at}</span>",
        unsafe_allow_html=True,
    )

    # ---- Tabs --------------------------------------------------------------
    tab1, tab2, tab3, tab4 = st.tabs([
        f"🎯 Active signals ({full_signals})",
        f"👁  Watchlist ({pass_all - full_signals})",
        "🪜 Filter funnel",
        "🔍 Stock detail",
    ])

    with tab1:
        st.markdown(
            "Stocks passing **all 5 filters** AND making a **new 52-week "
            "closing high today**. Per the strategy, you'd enter these at "
            "tomorrow's open."
        )
        signals = scan[scan["FullSignal"]].sort_values("%FromLow52W", ascending=False)
        candidate_table(
            signals,
            ["Symbol", "Close", "%Chg", "Volume",
             "%FromLow52W", "%FromHigh52W",
             "SMA50", "SMA150", "EMA220"],
        )
        if not signals.empty:
            st.download_button(
                "Download signals as CSV",
                signals.to_csv(index=False).encode("utf-8"),
                file_name=f"signals_{latest_data_date.date()}.csv",
                mime="text/csv",
            )

    with tab2:
        st.markdown(
            "Stocks passing **all 5 filters** but **not yet** at a new 52-week "
            "high. Sorted by closeness to breakout — the top of the list could "
            "trigger soon."
        )
        watch = scan[scan["AllFilters"] & ~scan["FullSignal"]].sort_values("%ToBreakout")
        candidate_table(
            watch,
            ["Symbol", "Close", "%Chg", "%ToBreakout",
             "PrevHigh52W", "%FromLow52W",
             "SMA50", "SMA150", "EMA220"],
        )

    with tab3:
        st.markdown(
            "How aggressively each successive filter narrows the universe. "
            "Useful for sensing market regime — in a strong bull, F1–3 pass "
            "wide; in a chop, F4 (25% off lows) and F5 (recent dip) tighten "
            "the cone."
        )
        st.plotly_chart(filter_funnel_chart(scan), use_container_width=True)

        st.markdown("##### Per-filter pass rates")
        per_filter = pd.DataFrame({
            "Filter": ["F1: SMA150 > EMA220", "F2: Close > SMA50",
                      "F3: SMA50 > SMA150",  "F4: Close > 1.25×Low52W",
                      "F5: Dipped recently",  "Breakout: new 52w high"],
            "Passing": [int(scan[c].sum()) for c in
                        ["F1", "F2", "F3", "F4", "F5", "Breakout"]],
        })
        per_filter["% of universe"] = (per_filter["Passing"] / len(scan) * 100).round(1)
        st.dataframe(per_filter, hide_index=True, use_container_width=True)

    with tab4:
        # Default to the strongest active signal if any, else first symbol
        default_idx = 0
        symbol_options = sorted(scan["Symbol"].tolist())
        if full_signals:
            top = scan[scan["FullSignal"]].sort_values("%FromLow52W", ascending=False).iloc[0]["Symbol"]
            if top in symbol_options:
                default_idx = symbol_options.index(top)

        col_sel, col_lookback = st.columns([2, 1])
        sym = col_sel.selectbox("Symbol", symbol_options, index=default_idx)
        lookback = col_lookback.selectbox("Lookback", [120, 252, 504], index=1,
                                          format_func=lambda x: f"{x} bars (~{x//21}m)")

        # Mini summary row
        row = scan[scan["Symbol"] == sym].iloc[0]
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Close", fmt_inr(row["Close"]),
                  delta=f"{row['%Chg']:+.2f}%")
        m2.metric("From 52W low", f"{row['%FromLow52W']:+.1f}%")
        m3.metric("From 52W high", f"{row['%FromHigh52W']:+.1f}%")
        m4.metric("Filters passed", f"{row['FilterCount']}/5",
                  delta="🎯 BREAKOUT" if row["FullSignal"] else None)

        # Filter status badges
        status = " ".join(
            ("✅" if row[f] else "❌") + f" {label}"
            for f, label in [
                ("F1", "SMA150>EMA220"), ("F2", "Close>SMA50"),
                ("F3", "SMA50>SMA150"),  ("F4", "Close>1.25×Low52W"),
                ("F5", "Dipped<EMA220 in 90d"),
                ("Breakout", "New 52W high"),
            ]
        )
        st.markdown(f"<div style='font-size:0.95rem'>{status}</div>",
                    unsafe_allow_html=True)

        st.plotly_chart(detail_chart(sym, prices[sym], lookback),
                        use_container_width=True)


if __name__ == "__main__":
    main()