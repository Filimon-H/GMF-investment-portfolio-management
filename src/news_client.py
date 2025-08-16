import os
import requests
from datetime import date, timedelta
from typing import List, Dict, Any, Tuple

def _date_range(days_back: int) -> Tuple[str, str]:
    to_d = date.today()
    from_d = to_d - timedelta(days=days_back)
    return from_d.isoformat(), to_d.isoformat()

def _symbol_for_finnhub(symbol: str) -> str:
    # Pass-through for US tickers/ETFs (TSLA, SPY, BND)
    return symbol.upper()

def _normalize_item(raw: Dict[str, Any], symbol: str) -> Dict[str, Any]:
    # Finnhub company-news fields: headline, summary, url, source, datetime (unix)
    ts = raw.get("datetime")
    return {
        "symbol": symbol,
        "headline": raw.get("headline", "").strip(),
        "summary": raw.get("summary", "").strip(),
        "url": raw.get("url"),
        "source": raw.get("source"),
        "time": ts,
        "image": raw.get("image"),
    }

def _sentiment_tag(text: str) -> str:
    """Very light heuristic sentiment (no heavy deps)."""
    if not text:
        return "→"
    t = text.lower()
    pos = any(k in t for k in ["beats", "surge", "rally", "record", "upgrades", "growth", "soars", "gain"])
    neg = any(k in t for k in ["misses", "falls", "plunge", "downgrade", "loss", "cuts", "drop", "slump"])
    if pos and not neg: return "↑"
    if neg and not pos: return "↓"
    return "→"

def fetch_news_finnhub(api_key: str, symbols: List[str], days_back: int = 7, limit_per_symbol: int = 8) -> List[Dict[str, Any]]:
    """
    Fetch latest headlines per symbol from Finnhub company-news and return a merged, recent-first list.
    """
    from_d, to_d = _date_range(days_back)
    base = "https://finnhub.io/api/v1/company-news"
    out: List[Dict[str, Any]] = []
    for sym in symbols:
        fsym = _symbol_for_finnhub(sym)
        params = {"symbol": fsym, "from": from_d, "to": to_d, "token": api_key}
        try:
            r = requests.get(base, params=params, timeout=10)
            r.raise_for_status()
            items = r.json()
            # normalize and keep only headline/url items
            norm = [_normalize_item(x, sym) for x in items if x.get("headline") and x.get("url")]
            # sort desc by time and trim
            norm.sort(key=lambda x: x.get("time", 0), reverse=True)
            out.extend(norm[:limit_per_symbol])
        except Exception:
            # Skip symbol on error; continue others
            continue
    # global sort by time desc
    out.sort(key=lambda x: x.get("time", 0), reverse=True)
    # attach a tiny sentiment tag
    for it in out:
        it["sentiment"] = _sentiment_tag(it["headline"])
    return out
