# src/news/summarizer.py
from dataclasses import dataclass
from typing import List, Optional
import json
import os

try:
    from groq import Groq
except Exception:
    Groq = None  # handle missing package gracefully








@dataclass
class SummaryResult:
    summary: str
    sentiment: str       # "Good", "Bad", "Neutral"
    sentiment_icon: str  # "📈", "📉", "→"
    bullets: List[str]

def _icon_for(sent: str) -> str:
    s = (sent or "").lower()
    if s.startswith("good") or s.startswith("pos"):
        return "📈"
    if s.startswith("bad") or s.startswith("neg"):
        return "📉"
    return "→"

def summarize_with_groq(text: str, symbol: str, *, max_tokens: int = 520) -> SummaryResult:
    """
    Summarize a news item with structured output.
    Prefers multi-bullet summary and a clear Good/Bad/Neutral impact tag.
    """
    # Fallback if SDK not installed or key missing
    api_key = os.getenv("GROQ_API_KEY")
    if (Groq is None) or (not api_key):
        t = (text or "").strip()
        # basic fallback
        t = (t[:400] + "…") if len(t) > 400 else t
        return SummaryResult(summary=t, sentiment="Neutral", sentiment_icon="→", bullets=[])

    client = Groq(api_key=api_key)

    system = (
        "You are a financial news assistant. Read the news text and produce:\n"
        "1) 3–5 concise bullet points (facts only, no hype),\n"
        "2) a 2-sentence takeaway for investors,\n"
        "3) an overall impact tag as one of: Good, Bad, Neutral (for the stock mentioned).\n"
        "Return strict JSON with keys: bullets (list of strings), takeaway (string), impact (string)."
    )
    user = f"Ticker: {symbol}\n\nNews text:\n{text}"

    try:
        resp = client.chat.completions.create(
            model="llama3-70b-8192",           # larger model for better summaries
            temperature=0.2,                   # stable outputs
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        content = resp.choices[0].message.content.strip()
        # Try to extract JSON (model often returns valid JSON when asked explicitly)
        data = json.loads(content)
        bullets = [b for b in (data.get("bullets") or []) if isinstance(b, str)]
        takeaway = (data.get("takeaway") or "").strip()
        impact = (data.get("impact") or "Neutral").strip()
        icon = _icon_for(impact)
        # Join bullets + takeaway into one summary block for display
        joined = "\n".join([f"• {b}" for b in bullets]) + (f"\n\n**Takeaway:** {takeaway}" if takeaway else "")
        return SummaryResult(summary=joined.strip(), sentiment=impact, sentiment_icon=icon, bullets=bullets)
    except Exception:
        # Robust fallback if parsing fails
        raw = resp.choices[0].message.content.strip()
        return SummaryResult(summary=raw, sentiment="Neutral", sentiment_icon="→", bullets=[])
