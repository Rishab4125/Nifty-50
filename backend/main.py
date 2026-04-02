from datetime import date, timedelta
from typing import List, Optional, Dict
import time
import random
import logging

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
from cachetools import TTLCache

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

from google import genai
from google.genai import types

load_dotenv()

# ── Enhanced session to mimic real browser traffic ──────────────────────────
_USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_4_1) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4.1 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:125.0) Gecko/20100101 Firefox/125.0",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36 Edg/123.0.0.0",
]

session = requests.Session()
session.headers.update({
    "User-Agent": random.choice(_USER_AGENTS),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Referer": "https://finance.yahoo.com/",
    "DNT": "1",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "same-origin",
    "Sec-Fetch-User": "?1",
    "Cache-Control": "max-age=0",
})

# ── In-memory caches (2-hour TTL) ───────────────────────────────────────────
# Cache for computed StockReturn objects: key = (symbol, as_of_date, include_dividends)
_returns_cache: TTLCache = TTLCache(maxsize=2048, ttl=2 * 60 * 60)  # 2 hours
# Cache for raw price DataFrames: key = (ticker, start_date, end_date)
_history_cache: TTLCache = TTLCache(maxsize=512, ttl=2 * 60 * 60)   # 2 hours

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


def _get_history(ticker: str, start: date, end: date, _retry_count: int = 3) -> pd.DataFrame:
    """
    Download raw price history for a ticker with caching and retry logic.
    Retries up to `_retry_count` times with exponential backoff on failure.
    """
    # ── Check history cache first ────────────────────────────────────────
    cache_key = (ticker, str(start), str(end))
    cached = _history_cache.get(cache_key)
    if cached is not None:
        logger.info(f"  [CACHE HIT] history for {ticker}")
        return cached

    # ── Fetch with retry + exponential backoff ───────────────────────────
    last_error = None
    for attempt in range(1, _retry_count + 1):
        try:
            # Rotate User-Agent on each attempt
            session.headers["User-Agent"] = random.choice(_USER_AGENTS)

            yf_ticker = yf.Ticker(ticker, session=session)
            df = yf_ticker.history(
                start=start, end=end + timedelta(days=1), auto_adjust=False
            )
            if df.empty:
                logger.warning(f"  [EMPTY] yfinance returned no data for {ticker} (attempt {attempt})")
                # Don't retry on empty — ticker may just have no data
                _history_cache[cache_key] = df
                return df

            # Normalise index
            df.index = pd.to_datetime(df.index)
            try:
                df.index = df.index.tz_localize(None)
            except (TypeError, AttributeError):
                pass

            # Cache and return
            _history_cache[cache_key] = df
            logger.info(f"  [FETCHED] {ticker}: {len(df)} rows (attempt {attempt})")
            return df

        except Exception as e:
            last_error = e
            wait = (2 ** (attempt - 1)) + random.uniform(0, 1)  # 1-2s, 2-3s, 4-5s
            logger.warning(
                f"  [RETRY] {ticker} attempt {attempt}/{_retry_count} failed: {e}. "
                f"Waiting {wait:.1f}s before retry..."
            )
            if attempt < _retry_count:
                time.sleep(wait)

    # All retries exhausted — return empty DataFrame
    logger.error(f"  [FAILED] {ticker} after {_retry_count} attempts: {last_error}")
    empty_df = pd.DataFrame()
    _history_cache[cache_key] = empty_df
    return empty_df


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
    # ── Check returns cache first ────────────────────────────────────────
    cache_key = (symbol, str(as_of), include_dividends)
    cached = _returns_cache.get(cache_key)
    if cached is not None:
        logger.info(f"[CACHE HIT] returns for {symbol}")
        return cached

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

    result = StockReturn(
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

    # Cache the result
    _returns_cache[cache_key] = result
    return result


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

    # ── Batch pre-warm: try yf.download() for all tickers at once ─────
    all_tickers = list(target_symbols.values())
    start_for_5y = _years_ago(as_of, 5) - timedelta(days=7)

    # Check how many are already cached
    uncached_tickers = [
        t for s, t in target_symbols.items()
        if _returns_cache.get((s, str(as_of), query.include_dividends)) is None
    ]

    if uncached_tickers:
        logger.info(f"[BATCH] Pre-warming {len(uncached_tickers)} uncached tickers via yf.download()")
        try:
            session.headers["User-Agent"] = random.choice(_USER_AGENTS)
            batch_df = yf.download(
                tickers=uncached_tickers,
                start=start_for_5y,
                end=as_of + timedelta(days=1),
                auto_adjust=False,
                group_by="ticker",
                threads=False,  # sequential to avoid rate-limit spikes
                session=session,
            )
            # Pre-populate the history cache from batch results
            if not batch_df.empty:
                for ticker in uncached_tickers:
                    try:
                        if len(uncached_tickers) == 1:
                            ticker_df = batch_df.copy()
                        else:
                            ticker_df = batch_df[ticker].dropna(how="all")
                        if not ticker_df.empty:
                            ticker_df.index = pd.to_datetime(ticker_df.index)
                            try:
                                ticker_df.index = ticker_df.index.tz_localize(None)
                            except (TypeError, AttributeError):
                                pass
                            hcache_key = (ticker, str(start_for_5y), str(as_of))
                            _history_cache[hcache_key] = ticker_df
                    except Exception:
                        pass  # individual ticker extraction failed, will retry individually
                logger.info(f"[BATCH] Pre-warmed history cache with batch download")
        except Exception as e:
            logger.warning(f"[BATCH] Batch download failed: {e}. Will fetch individually.")
    else:
        logger.info(f"[CACHE] All {len(target_symbols)} tickers already cached")

    # ── Compute returns per stock (will use cache from batch above) ──────
    for i, (symbol, ticker) in enumerate(target_symbols.items()):
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
        # Small random delay between individual fetches to avoid rate-limiting
        if i < len(target_symbols) - 1 and uncached_tickers:
            time.sleep(random.uniform(0.1, 0.3))

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
    return {
        "status": "ok",
        "returns_cache_size": len(_returns_cache),
        "history_cache_size": len(_history_cache),
    }


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
