from datetime import date, timedelta
from typing import List, Optional, Dict

import numpy as np
import pandas as pd
import yfinance as yf
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


class StockReturn(BaseModel):
    symbol: str
    name: Optional[str]
    one_year: Optional[float]
    three_year: Optional[float]
    five_year: Optional[float]


class ReturnsResponse(BaseModel):
    as_of_date: date
    include_dividends: bool
    stocks: List[StockReturn]


class ReturnsQuery(BaseModel):
    as_of_date: date
    include_dividends: bool = False


app = FastAPI(title="Nifty 50 Returns API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


NIFTY_50_SYMBOLS: Dict[str, str] = {
    # Mapping: NSE symbol -> Yahoo Finance ticker
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
    "TATAMOTORS": "TATAMOTORS.NS",
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


def _years_ago(as_of: date, years: int) -> date:
    """
    Move back exactly `years` calendar years, adjusting for leap years.
    E.g. 2024-02-29 -> 2019-02-28 for 5 years ago.
    """
    try:
        return as_of.replace(year=as_of.year - years)
    except ValueError:
        # Handles dates like Feb 29 which don't exist in the target year.
        return as_of.replace(month=2, day=28, year=as_of.year - years)


def _get_history_with_actions(ticker: str, start: date, end: date) -> pd.DataFrame:
    """Download price history, stock splits, and dividends for a ticker."""
    yf_ticker = yf.Ticker(ticker)
    df = yf_ticker.history(start=start, end=end + timedelta(days=1), auto_adjust=False)
    if df.empty:
        return df
    df = df[["Close", "Dividends", "Stock Splits"]]
    df.index = pd.to_datetime(df.index)
    return df


def _apply_split_adjustment(df: pd.DataFrame) -> pd.DataFrame:
    """Create a Close price series adjusted for splits/bonuses but not dividends."""
    if df.empty:
        return df

    df = df.copy()
    df["split_factor"] = 1.0

    factors = []
    cumulative = 1.0
    for split in df["Stock Splits"].fillna(0):
        if split and split > 0:
            cumulative *= split
        factors.append(cumulative)
    df["split_factor"] = factors

    df["Close_adj_split"] = df["Close"] / df["split_factor"].replace(0, np.nan)
    return df


def _nearest_trading_day(df: pd.DataFrame, target: date) -> Optional[pd.Timestamp]:
    if df.empty:
        return None
    ts = pd.to_datetime(target)
    before = df.index[df.index <= ts]
    after = df.index[df.index >= ts]
    candidates = []
    if len(before) > 0:
        candidates.append(before[-1])
    if len(after) > 0:
        candidates.append(after[0])
    if not candidates:
        return None
    return min(candidates, key=lambda d: abs(d - ts))


def _compute_horizon_return(
    df: pd.DataFrame, as_of: date, years: int, include_dividends: bool
) -> Optional[float]:
    if df.empty:
        return None

    end_ts = _nearest_trading_day(df, as_of)
    if end_ts is None:
        return None

    start_date = _years_ago(as_of, years)
    start_ts = _nearest_trading_day(df, start_date)
    if start_ts is None:
        return None

    df = df.loc[min(start_ts, end_ts) : max(start_ts, end_ts)].copy()
    df = _apply_split_adjustment(df)

    start_price = float(df.loc[start_ts, "Close_adj_split"])
    end_price = float(df.loc[end_ts, "Close_adj_split"])

    if start_price <= 0:
        return None

    if include_dividends:
        dividends = float(df.loc[start_ts:end_ts, "Dividends"].sum())
        total_return = (end_price + dividends) / start_price - 1.0
    else:
        total_return = end_price / start_price - 1.0

    return total_return


def _compute_stock_returns(
    symbol: str, ticker: str, as_of: date, include_dividends: bool
) -> StockReturn:
    # Fetch enough history to cover 5Y lookback plus a small buffer for holidays.
    start_for_5y = _years_ago(as_of, 5) - timedelta(days=7)
    df = _get_history_with_actions(ticker, start_for_5y, as_of)

    one = _compute_horizon_return(df, as_of, 1, include_dividends)
    three = _compute_horizon_return(df, as_of, 3, include_dividends)
    five = _compute_horizon_return(df, as_of, 5, include_dividends)

    name = None
    try:
        info = yf.Ticker(ticker).fast_info
        name = getattr(info, "shortName", None) or getattr(info, "longName", None)
    except Exception:
        name = None

    return StockReturn(
        symbol=symbol,
        name=name,
        one_year=one,
        three_year=three,
        five_year=five,
    )


@app.post("/api/returns", response_model=ReturnsResponse)
def get_nifty_returns(query: ReturnsQuery) -> ReturnsResponse:
    as_of = query.as_of_date
    if as_of > date.today():
        raise HTTPException(status_code=400, detail="as_of_date cannot be in the future")

    stocks: List[StockReturn] = []
    for symbol, ticker in NIFTY_50_SYMBOLS.items():
        try:
            sr = _compute_stock_returns(symbol, ticker, as_of, query.include_dividends)
            stocks.append(sr)
        except Exception:
            stocks.append(
                StockReturn(
                    symbol=symbol,
                    name=None,
                    one_year=None,
                    three_year=None,
                    five_year=None,
                )
            )

    return ReturnsResponse(as_of_date=as_of, include_dividends=query.include_dividends, stocks=stocks)


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}

