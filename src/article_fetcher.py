# src/article_fetcher.py
from __future__ import annotations
import trafilatura
from trafilatura.settings import use_config
from functools import lru_cache

# Build a fast, polite config (short timeout; no sitemaps; minimal extras)
_CFG = use_config()
_CFG.set("DEFAULT", "user_agent", "GMF-Dashboard/1.0 (+https://example.com)")
_CFG.set("DEFAULT", "timeout", "6")
_CFG.set("DEFAULT", "favor_precision", "true")
_CFG.set("DEFAULT", "include_formatting", "false")
_CFG.set("DEFAULT", "only_with_metadata", "false")

@lru_cache(maxsize=512)
def fetch_article_text(url: str) -> str:
    """
    Fetch & extract full article text from a URL using trafilatura.
    Returns a clean string or "" if extraction fails.
    Cached to avoid re-downloading.
    """
    if not url or not url.startswith(("http://", "https://")):
        return ""
    try:
        raw = trafilatura.fetch_url(url, config=_CFG)
        if not raw:
            return ""
        text = trafilatura.extract(
            raw,
            include_comments=False,
            include_tables=False,
            no_fallback=False,
            config=_CFG,
        )
        if not text:
            return ""
        # Trim overly long texts (LLM cost/latency control)
        text = text.strip()
        return text
    except Exception:
        return ""

def choose_text_for_summary(headline: str, finnhub_summary: str, article_text: str) -> str:
    """
    Pick the most informative text for LLM summarization.
    Preference: full article > publisher summary > headline.
    """
    article_text = (article_text or "").strip()
    fin_sum = (finnhub_summary or "").strip()
    head = (headline or "").strip()

    # If we got a decent article body, use it
    if len(article_text) >= 600:    # ~2–3 paragraphs
        return article_text

    # Otherwise use the Finnhub publisher summary if non-trivial
    if len(fin_sum) >= 120:
        return f"{head}\n\n{fin_sum}" if head else fin_sum

    # Fallback to headline
    return head
