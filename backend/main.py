from datetime import date, timedelta
from typing import List, Optional, Dict

import pandas as pd
import yfinance as yf
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests


session = requests.Session()
session.headers.update({"User-Agent": "Mozilla/5.0 (compatible; Nifty50ReturnsBot/1.0)"})
yf.utils.requests = lambda : session

class StockReturn(BaseModel):
    symbol: str
    name: Optional[str]
    one_year: Optional[float]
    three_year: Optional[float]
    five_year: Optional[float]
    one_year_dividend: Optional[float] = None
    three_year_dividend: Optional[float] = None
    five_year_dividend: Optional[float] = None


class ReturnsResponse(BaseModel):
    as_of_date: date
    include_dividends: bool
    stocks: List[StockReturn]


class DebugResponse(BaseModel):
    symbol: str
    rows: int
    start_date: Optional[date]
    end_date: Optional[date]
    approx_5y_return: Optional[float]


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
    "Bharat Electronics": "BEL.NS",
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


def _get_history(ticker: str, start: date, end: date) -> pd.DataFrame:
    """
    Download raw price history for a ticker, in the same spirit as test_yfinance.py.
    """
    yf_ticker = yf.Ticker(ticker)
    df = yf_ticker.history(start=start, end=end + timedelta(days=1), auto_adjust=False)
    if df.empty:
        return df
    # Normalise index and ensure we have expected columns if present.
    df.index = pd.to_datetime(df.index)
    # yfinance can return timezone-aware indices; make them tz-naive so that
    # comparisons with plain dates (as_of_date) don't raise TypeError.
    try:
        df.index = df.index.tz_localize(None)
    except (TypeError, AttributeError):
        pass
    return df


def _compute_horizon_return(
    df: pd.DataFrame, as_of: date, years: int, include_dividends: bool
) -> Optional[float]:
    if df.empty:
        return None

    # Choose price series:
    # - If include_dividends: use Adj Close (Yahoo total-return style).
    # - If exclude_dividends: use Close (price-only; splits handled by Yahoo).
    price_col = "Adj Close" if include_dividends and "Adj Close" in df.columns else "Close"
    if price_col not in df.columns:
        return None

    idx = df.index
    if len(idx) == 0:
        return None

    as_of_ts = pd.to_datetime(as_of)
    allowed_end = idx[idx <= as_of_ts]
    if len(allowed_end) == 0:
        return None
    end_ts = allowed_end[-1]

    start_target = _years_ago(as_of, years)
    start_ts_candidates = idx[idx >= pd.to_datetime(start_target)]
    if len(start_ts_candidates) == 0:
        start_ts = idx[0]
    else:
        start_ts = start_ts_candidates[0]

    # Slice and compute simple total return.
    start_price = float(df.loc[start_ts, price_col])
    end_price = float(df.loc[end_ts, price_col])
    if start_price <= 0:
        return None

    return end_price / start_price - 1.0


def _compute_horizon_returns_with_dividends(
    df: pd.DataFrame, as_of: date, years: int, include_dividends: bool
) -> (Optional[float], Optional[float]):
    """
    Compute price-only return and dividend contribution separately when possible.
    Returns (total_return_used, dividend_return).
    """
    if df.empty:
        return None, None

    cols = df.columns
    has_close = "Close" in cols
    has_adj = "Adj Close" in cols

    # Fallback to the simpler logic if we don't have both series.
    if not (has_close and has_adj):
        base = _compute_horizon_return(df, as_of, years, include_dividends)
        return base, None

    idx = df.index
    if len(idx) == 0:
        return None, None

    as_of_ts = pd.to_datetime(as_of)
    allowed_end = idx[idx <= as_of_ts]
    if len(allowed_end) == 0:
        return None, None
    end_ts = allowed_end[-1]

    start_target = _years_ago(as_of, years)
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


def _compute_stock_returns(
    symbol: str, ticker: str, as_of: date, include_dividends: bool
) -> StockReturn:
    # Fetch enough history to cover 5Y lookback plus a small buffer for holidays.
    start_for_5y = _years_ago(as_of, 5) - timedelta(days=7)
    df = _get_history(ticker, start_for_5y, as_of)

    one, one_div = _compute_horizon_returns_with_dividends(
        df, as_of, 1, include_dividends
    )
    three, three_div = _compute_horizon_returns_with_dividends(
        df, as_of, 3, include_dividends
    )
    five, five_div = _compute_horizon_returns_with_dividends(
        df, as_of, 5, include_dividends
    )

    # We skip name lookups to avoid extra Yahoo calls that can fail.
    name = None

    return StockReturn(
        symbol=symbol,
        name=name,
        one_year=one,
        three_year=three,
        five_year=five,
        one_year_dividend=one_div,
        three_year_dividend=three_div,
        five_year_dividend=five_div,
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


@app.get("/debug/{ticker}", response_model=DebugResponse)
def debug_ticker(ticker: str) -> DebugResponse:
    """
    Minimal debug endpoint that mirrors test_yfinance.py behaviour for a single ticker.
    Helps verify that yfinance + history() work inside the FastAPI process.
    """
    as_of = date.today()
    start = _years_ago(as_of, 5) - timedelta(days=7)
    df = _get_history(ticker, start, as_of)

    if df.empty:
        return DebugResponse(
            symbol=ticker,
            rows=0,
            start_date=None,
            end_date=None,
            approx_5y_return=None,
        )

    first_idx = df.index[0]
    last_idx = df.index[-1]

    price_col = "Adj Close" if "Adj Close" in df.columns else "Close"
    first_close = float(df[price_col].iloc[0])
    last_close = float(df[price_col].iloc[-1])

    approx = None
    if first_close > 0:
        approx = last_close / first_close - 1.0

    return DebugResponse(
        symbol=ticker,
        rows=len(df),
        start_date=first_idx.date(),
        end_date=last_idx.date(),
        approx_5y_return=approx,
    )

