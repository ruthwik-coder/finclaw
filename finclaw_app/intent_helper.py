"""
FinClaw Intent & Budget Extraction Helper
=========================================
Extracts purchase items, costs, and categories from conversational messages,
then queries db.sqlite to produce the live financial ledger.
"""

import re
from typing import Dict, Any, Tuple
from fetch_actual_budget import fetch_finclaw_ledger, resolve_category_name, CATEGORY_KEYWORD_MAP

def extract_purchase_intent(text: str) -> Tuple[str, str, float]:
    """
    Returns (category_hint, item_name, cost)
    """
    cleaned = re.sub(r'(\d),(\d)', r'\1\2', text)

    # 1. Cost detection ($ or ₹ or number)
    cost = 0.0
    m_cost = re.search(r'[\$₹]\s*(\d+(?:\.\d+)?)|(?:rs\.?\s*)(\d+(?:\.\d+)?)|(\d+(?:\.\d+)?)\s*(?:dollars|bucks|usd|rs|inr)', cleaned, re.I)
    if m_cost:
        for g in m_cost.groups():
            if g is not None:
                cost = float(g)
                break
    if cost == 0.0:
        m_alt = re.search(r'(?:costs?|for|costing|at|priced at|worth|spending|paying|about|around|of)\s*(?:[₹$]|rs\.?\s*)?(\d+(?:\.\d+)?)', cleaned, re.I)
        if m_alt:
            cost = float(m_alt.group(1))

    # 2. Category hint & Item name
    category_hint = "General"
    item_name = "Discretionary Expense"

    lower_text = text.lower()
    # Sort keywords by descending length so multi-word matches take precedence
    for kw in sorted(CATEGORY_KEYWORD_MAP.keys(), key=len, reverse=True):
        if re.search(r'\b' + re.escape(kw) + r'\b', lower_text):
            category_hint = CATEGORY_KEYWORD_MAP[kw]
            item_name = kw.capitalize()
            break

    # Look for item nouns after 'buy', 'get', 'ordering', 'craving', 'planning to buy', etc.
    m_noun = re.search(r'(?:buy|buying|get|ordering|craving|purchase|planning to buy|planning on buying|subscribe to|subscribing to)\s+(?:a|an)?\s*([a-zA-Z\s]+?)(?:\s+for|\s+worth|\s+tonight|\s+today|\s+costing|\.|\?|$)', cleaned, re.I)
    if m_noun:
        found_item = m_noun.group(1).strip()
        if len(found_item.split()) <= 5 and found_item.lower() not in ['it', 'something', 'this', 'that']:
            item_name = found_item.capitalize()

    return category_hint, item_name, cost


def get_live_ledger_for_message(
    db_path: str,
    user_text: str,
    active_ledger: Dict[str, Any] = None,
    engine: Any = None
) -> Dict[str, Any]:
    """
    Checks if a new purchase is mentioned; if so, queries db.sqlite.
    Uses Cascading Hybrid resolution:
      1. Fast Regex/Dictionary (<0.05ms)
      2. If category is 'General' but a purchase is detected, invokes LLM classifier fallback (~150ms)
    """
    cat_hint, item_name, cost = extract_purchase_intent(user_text)

    # Cascading Fallback: If regex didn't find a specific category, invoke LLM classifier
    if cat_hint == "General" and (cost > 0.0 or any(w in user_text.lower() for w in ["buy", "purchase", "order", "get", "subscribe"])) and engine is not None:
        try:
            llm_cat = engine.classify_category(user_text)
            if llm_cat and llm_cat != "General":
                cat_hint = llm_cat
        except Exception as e:
            print(f"[Classifier] LLM fallback error: {e}")

    if cost > 0.0 or cat_hint != "General":
        # New purchase decision detected!
        return fetch_finclaw_ledger(
            db_path=db_path,
            category_query=cat_hint,
            item_name=item_name,
            item_cost=cost,
            month_str="202609"
        )

    elif active_ledger:
        # Retain current ledger for follow-up conversational turns
        return active_ledger
    else:
        # Default baseline ledger
        return fetch_finclaw_ledger(
            db_path=db_path,
            category_query="General",
            item_name="General Discussion",
            item_cost=0.0,
            month_str="202609"
        )
