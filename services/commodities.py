import yfinance as yf
import pandas as pd
from datetime import datetime

# Mapping user-friendly commodity names to Yahoo Finance tickers (common futures)
DEFAULT_TICKER_MAP = {
    'Soybean': 'ZS=F',       # CBOT Soybeans
    'Corn': 'ZC=F',          # CBOT Corn (Maize)
    'Wheat': 'ZW=F',         # CBOT Wheat
    'Soybean Oil': 'ZL=F',   # Soybean Oil futures (where available)
    'Soybean Meal': 'ZM=F'   # Soybean Meal futures
}


def fetch_commodity_history(ticker: str, period: str = '90d', interval: str = '1d') -> pd.DataFrame:
    """Fetch historical price data for a ticker using yfinance.

    Returns a DataFrame with Datetime index and columns including 'Open','High','Low','Close','Volume'.
    """
    if not ticker:
        return pd.DataFrame()
    try:
        t = yf.Ticker(ticker)
        hist = t.history(period=period, interval=interval, actions=False)
        if hist is None or hist.empty:
            return pd.DataFrame()
        hist = hist.reset_index()
        hist['Datetime'] = pd.to_datetime(hist['Date']) if 'Date' in hist.columns else pd.to_datetime(hist['Datetime'])
        # Normalize column names
        if 'Close' in hist.columns:
            hist = hist[['Datetime', 'Close']]
            hist = hist.rename(columns={'Close': 'close'})
        else:
            hist = hist.rename(columns={hist.columns[-1]: 'close'})
            hist = hist[['Datetime', 'close']]
        return hist
    except Exception:
        return pd.DataFrame()


def fetch_current_price(ticker: str):
    """Fetch current market price for a ticker. Returns float or None."""
    if not ticker:
        return None
    try:
        t = yf.Ticker(ticker)
        info = t.info
        # prefer regularMarketPrice, fallback to previousClose
        price = info.get('regularMarketPrice') or info.get('previousClose') or info.get('last_price')
        if price is None:
            # as a last resort try fast history
            hist = t.history(period='2d', interval='1m')
            if hist is not None and not hist.empty:
                price = float(hist['Close'].iloc[-1])
        return float(price) if price is not None else None
    except Exception:
        return None


def get_commodities_data(names, period='90d', interval='1d'):
    """Return a dict mapping commodity name to dict {ticker, current, history(DataFrame)}."""
    out = {}
    for name in names:
        ticker = DEFAULT_TICKER_MAP.get(name)
        if not ticker:
            out[name] = {'ticker': None, 'current': None, 'history': pd.DataFrame()}
            continue
        hist = fetch_commodity_history(ticker, period=period, interval=interval)
        current = fetch_current_price(ticker)
        out[name] = {'ticker': ticker, 'current': current, 'history': hist}
    return out
