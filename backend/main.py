from datetime import date, timedelta
from typing import List, Optional, Dict

import pandas as pd
import yfinance as yf
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests
import os
import json
import math
from dotenv import load_dotenv

def _clean_float(val) -> Optional[float]:
    if val is None:
        return None
    try:
        f = float(val)
        if math.isnan(f) or math.isinf(f):
            return None
        return f
    except Exception:
        return None

load_dotenv()

# session = requests.Session()
# session.headers.update({"User-Agent": "Mozilla/5.0 (compatible; Nifty50ReturnsBot/1.0)"})
# session.headers.update({
#     "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
#     "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
# })
# yf.utils.requests = lambda : session

from google import genai
from google.genai import types

class StockReturn(BaseModel):
    symbol: str
    name: Optional[str]
    one_year: Optional[float]
    three_year: Optional[float]
    five_year: Optional[float]
    one_year_dividend: Optional[float] = None
    three_year_dividend: Optional[float] = None
    five_year_dividend: Optional[float] = None
    price: Optional[float] = None
    week52_low: Optional[float] = None
    week52_high: Optional[float] = None


class ReturnsResponse(BaseModel):
    as_of_date: date
    include_dividends: bool
    stocks: List[StockReturn]
    portfolio: Optional[StockReturn] = None


class DebugResponse(BaseModel):
    symbol: str
    rows: int
    start_date: Optional[date]
    end_date: Optional[date]
    approx_5y_return: Optional[float]


class ReturnsQuery(BaseModel):
    as_of_date: date
    include_dividends: bool = False
    symbols: Optional[List[str]] = None
    search_query: Optional[str] = None


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
    # "TATAMOTORS": "TATAMOTORS.NS",
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
    "Bharat Electronics": "BEL.NS",
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

    # 52W range + as-of price, computed from raw history.
    price = None
    week52_low = None
    week52_high = None
    if not df.empty and len(df.index) > 0 and "Close" in df.columns:
        as_of_ts = pd.to_datetime(as_of)
        allowed_end = df.index[df.index <= as_of_ts]
        if len(allowed_end) > 0:
            end_ts = allowed_end[-1]
            try:
                price = float(df.loc[end_ts, "Close"])
            except Exception:
                price = None

            # Use trailing 52 weeks ending at end_ts.
            window_start = end_ts - pd.Timedelta(days=365)
            window_df = df.loc[window_start:end_ts]
            if not window_df.empty:
                # Prefer intraday High/Low if present; fall back to Close.
                if "Low" in window_df.columns and "High" in window_df.columns:
                    try:
                        week52_low = float(window_df["Low"].min())
                        week52_high = float(window_df["High"].max())
                    except Exception:
                        week52_low, week52_high = None, None
                else:
                    try:
                        week52_low = float(window_df["Close"].min())
                        week52_high = float(window_df["Close"].max())
                    except Exception:
                        week52_low, week52_high = None, None

    return StockReturn(
        symbol=symbol,
        name=name,
        one_year=_clean_float(one),
        three_year=_clean_float(three),
        five_year=_clean_float(five),
        one_year_dividend=_clean_float(one_div),
        three_year_dividend=_clean_float(three_div),
        five_year_dividend=_clean_float(five_div),
        price=_clean_float(price),
        week52_low=_clean_float(week52_low),
        week52_high=_clean_float(week52_high),
    )


def _resolve_search_query_to_symbols(q: str) -> Dict[str, str]:
    """Call the LLM to resolve a search query into {symbol: ticker}. Raises ValueError if q is blank or LLM fails."""
    if not q or not q.strip():
        raise ValueError("Search query must not be empty.")

    prompt = f"""
    You are a financial assistant for Indian stocks. The user searched for: "{q}".
    If the query is a sector or theme (e.g. "IT", "Pharma", "Banking"), return the top 15 Indian stocks in that sector.
    If the query is a specific stock (e.g. "Tata Motors", "RELIANCE"), return that exact stock AND its top 5 closest Indian peers (6 total).
    Criteria for best matches include: relevance to the query, market cap, performance, RSI, Moving Averages, whether if its cheap or expensive as per current price in market and popularity among Indian investors.
    Respond ONLY with a valid JSON array of objects. Do not include markdown formatting or backticks.
    Each object must have exactly two keys: "symbol" (the NSE symbol without .NS, e.g. "TCS") and "ticker" (the Yahoo Finance ticker, e.g. "TCS.NS").
    """
    try:
        client = genai.Client()
        response = client.models.generate_content(
            # model='gemini-2.5-flash',
            model='gemini-3.1-flash-lite-preview',
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.0,
                response_mime_type="application/json",
            )
        )
        raw_text = response.text.strip()
        if raw_text.startswith("```json"):
            raw_text = raw_text[7:]
        elif raw_text.startswith("```"):
            raw_text = raw_text[3:]
        if raw_text.endswith("```"):
            raw_text = raw_text[:-3]
        
        items = json.loads(raw_text.strip())
        if isinstance(items, list) and len(items) > 0:
            return {item["symbol"]: item["ticker"] for item in items if "symbol" in item and "ticker" in item}
        
        raise ValueError("LLM returned empty or invalid json format.")
    except Exception as e:
        print("Error fetching tickers from LLM:", e)
        raise ValueError(f"LLM AI search failed: {str(e)}")


@app.get("/api/tickers")
def get_tickers(q: Optional[str] = None) -> List[Dict[str, str]]:
    """Return the list of stock symbols. If q is provided, use AI to generate the list. Otherwise return defaults."""
    if not q or not q.strip():
        return [{"symbol": k, "ticker": v} for k, v in NIFTY_50_SYMBOLS.items()]
    try:
        resolved = _resolve_search_query_to_symbols(q)
        return [{"symbol": k, "ticker": v} for k, v in resolved.items()]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/returns", response_model=ReturnsResponse)
def get_nifty_returns(query: ReturnsQuery) -> ReturnsResponse:
    as_of = query.as_of_date
    if as_of > date.today():
        raise HTTPException(status_code=400, detail="as_of_date cannot be in the future")

    stocks: List[StockReturn] = []
    
    target_symbols = NIFTY_50_SYMBOLS
    if query.search_query and query.search_query.strip():
        try:
            target_symbols = _resolve_search_query_to_symbols(query.search_query.strip())
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    elif query.symbols is not None:
        target_symbols = {s: NIFTY_50_SYMBOLS.get(s, f"{s}.NS") for s in query.symbols}

    for symbol, ticker in target_symbols.items():
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

    portfolio = None
    try:
        # Calculate returns for the Nifty 50 Index itself as the portfolio benchmark
        portfolio = _compute_stock_returns("NIFTY 50", "^NSEI", as_of, query.include_dividends)
    except Exception:
        pass

    return ReturnsResponse(
        as_of_date=as_of,
        include_dividends=query.include_dividends,
        stocks=stocks,
        portfolio=portfolio
    )


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
        approx_5y_return=_clean_float(approx),
    )
