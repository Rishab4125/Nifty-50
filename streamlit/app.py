from datetime import date, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf


NIFTY_50_SYMBOLS: Dict[str, str] = {
    "RELIANCE": "RELIANCE.NS",
    "HDFCBANK": "HDFCBANK.NS",
    "ICICIBANK": "ICICIBANK.NS",
    "INFY": "INFY.NS",
    "TCS": "TCS.NS",
    "ITC": "ITC.NS",
    "LT": "LT.NS",
    "SBIN": "SBIN.NS",
    "KOTAKBANK": "KOTAKBANK.NS",
    "AXISBANK": "AXISBANK.NS",
    "HINDUNILVR": "HINDUNILVR.NS",
    "BHARTIARTL": "BHARTIARTL.NS",
    "BAJFINANCE": "BAJFINANCE.NS",
    "ASIANPAINT": "ASIANPAINT.NS",
    "MARUTI": "MARUTI.NS",
    "SUNPHARMA": "SUNPHARMA.NS",
    "TITAN": "TITAN.NS",
    "ULTRACEMCO": "ULTRACEMCO.NS",
    "NESTLEIND": "NESTLEIND.NS",
    "HCLTECH": "HCLTECH.NS",
    "WIPRO": "WIPRO.NS",
    "ONGC": "ONGC.NS",
    "TMPV": "TMPV.NS",
    "COALINDIA": "COALINDIA.NS",
    "BAJAJ-AUTO": "BAJAJ-AUTO.NS",
    "TATASTEEL": "TATASTEEL.NS",
    "JSWSTEEL": "JSWSTEEL.NS",
    "POWERGRID": "POWERGRID.NS",
    "NTPC": "NTPC.NS",
    "ADANIENT": "ADANIENT.NS",
    "ADANIPORTS": "ADANIPORTS.NS",
    "DIVISLAB": "DIVISLAB.NS",
    "CIPLA": "CIPLA.NS",
    "DRREDDY": "DRREDDY.NS",
    "GRASIM": "GRASIM.NS",
    "HDFCLIFE": "HDFCLIFE.NS",
    "SBILIFE": "SBILIFE.NS",
    "BAJAJFINSV": "BAJAJFINSV.NS",
    "BRITANNIA": "BRITANNIA.NS",
    "HEROMOTOCO": "HEROMOTOCO.NS",
    "EICHERMOT": "EICHERMOT.NS",
    "TECHM": "TECHM.NS",
    "UPL": "UPL.NS",
    "SHREECEM": "SHREECEM.NS",
    "M&M": "M&M.NS",
    "TATACONSUM": "TATACONSUM.NS",
    "BPCL": "BPCL.NS",
    "IOC": "IOC.NS",
    "HINDALCO": "HINDALCO.NS",
}


def years_ago(as_of: date, years: int) -> date:
    try:
        return as_of.replace(year=as_of.year - years)
    except ValueError:
        return as_of.replace(month=2, day=28, year=as_of.year - years)


@st.cache_data(show_spinner=False)
def get_history(ticker: str, start: date, end: date) -> pd.DataFrame:
    yf_ticker = yf.Ticker(ticker)
    df = yf_ticker.history(start=start, end=end + timedelta(days=1), auto_adjust=False)
    if df.empty:
        return df
    df.index = pd.to_datetime(df.index)
    try:
        df.index = df.index.tz_localize(None)
    except (TypeError, AttributeError):
        pass
    return df


def compute_horizon_returns_with_dividends(
    df: pd.DataFrame, as_of: date, years: int, include_dividends: bool
) -> Tuple[Optional[float], Optional[float]]:
    if df.empty:
        return None, None

    cols = df.columns
    has_close = "Close" in cols
    has_adj = "Adj Close" in cols

    if not (has_close and has_adj):
        return None, None

    idx = df.index
    if len(idx) == 0:
        return None, None

    as_of_ts = pd.to_datetime(as_of)
    allowed_end = idx[idx <= as_of_ts]
    if len(allowed_end) == 0:
        return None, None
    end_ts = allowed_end[-1]

    start_target = years_ago(as_of, years)
    start_ts_candidates = idx[idx >= pd.to_datetime(start_target)]
    start_ts = start_ts_candidates[0] if len(start_ts_candidates) else idx[0]

    start_close = float(df.loc[start_ts, "Close"])
    end_close = float(df.loc[end_ts, "Close"])
    start_adj = float(df.loc[start_ts, "Adj Close"])
    end_adj = float(df.loc[end_ts, "Adj Close"])

    if start_close <= 0 or start_adj <= 0:
        return None, None

    price_return = end_close / start_close - 1.0
    total_return = end_adj / start_adj - 1.0
    dividend_return = total_return - price_return

    if include_dividends:
        return total_return, dividend_return
    else:
        return price_return, dividend_return


def compute_for_symbol(symbol: str, yf_ticker: str, as_of: date, include_dividends: bool):
    start_for_5y = years_ago(as_of, 5) - timedelta(days=7)
    df = get_history(yf_ticker, start_for_5y, as_of)

    one, one_div = compute_horizon_returns_with_dividends(df, as_of, 1, include_dividends)
    three, three_div = compute_horizon_returns_with_dividends(df, as_of, 3, include_dividends)
    five, five_div = compute_horizon_returns_with_dividends(df, as_of, 5, include_dividends)

    return {
        "Symbol": symbol,
        "1Y": one,
        "3Y": three,
        "5Y": five,
        "1Y Div": one_div,
        "3Y Div": three_div,
        "5Y Div": five_div,
    }


def format_pct(v: Optional[float]) -> str:
    if v is None or np.isnan(v):
        return "—"
    return f"{v * 100:.1f}%"


def main():
    st.set_page_config(
        page_title="Nifty 50 Return Tracker",
        layout="wide",
    )

    st.title("Nifty 50 Return Tracker")
    st.caption(
        "Exact 1 / 3 / 5 year returns for current Nifty 50 constituents as of a chosen date. "
        "Splits/bonus handled via Yahoo prices, with an optional dividend toggle and separate dividend contribution columns."
    )

    col1, col2, col3 = st.columns([2, 2, 3])
    with col1:
        as_of = st.date_input("As-of date", value=date.today(), max_value=date.today())
    with col2:
        include_dividends = st.checkbox("Include dividends in total return", value=False)
    with col3:
        sort_key = st.selectbox(
            "Sort by",
            options=[
                "Symbol",
                "1Y",
                "3Y",
                "5Y",
                "1Y Div",
                "3Y Div",
                "5Y Div",
            ],
            index=3,
        )
        sort_desc = st.checkbox("Sort descending", value=True)

    st.divider()

    run_btn = st.button("Compute returns")

    if run_btn:
        rows: List[Dict] = []
        progress = st.progress(0.0)
        total = len(NIFTY_50_SYMBOLS)

        for i, (symbol, yf_ticker) in enumerate(NIFTY_50_SYMBOLS.items(), start=1):
            try:
                row = compute_for_symbol(symbol, yf_ticker, as_of, include_dividends)
                rows.append(row)
            except Exception:
                rows.append(
                    {
                        "Symbol": symbol,
                        "1Y": None,
                        "3Y": None,
                        "5Y": None,
                        "1Y Div": None,
                        "3Y Div": None,
                        "5Y Div": None,
                    }
                )
            progress.progress(i / total)

        df = pd.DataFrame(rows)

        key_map = {
            "Symbol": "Symbol",
            "1Y": "1Y",
            "3Y": "3Y",
            "5Y": "5Y",
            "1Y Div": "1Y Div",
            "3Y Div": "3Y Div",
            "5Y Div": "5Y Div",
        }
        sort_col = key_map.get(sort_key, "5Y")

        if sort_col in df.columns:
            df = df.sort_values(by=sort_col, ascending=not sort_desc, na_position="last")

        styled = df.copy()
        for col in ["1Y", "3Y", "5Y", "1Y Div", "3Y Div", "5Y Div"]:
            if col in styled.columns:
                styled[col] = styled[col].apply(format_pct)

        st.subheader("Returns table")
        st.write(
            f"As of **{as_of}** · Dividends **{'included' if include_dividends else 'excluded'}** in main return columns."
        )
        st.dataframe(
            styled,
            use_container_width=True,
        )

        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download CSV",
            data=csv,
            file_name=f"nifty50-returns-{as_of}{'-with-dividends' if include_dividends else '-price-only'}.csv",
            mime="text/csv",
        )


if __name__ == "__main__":
    main()

