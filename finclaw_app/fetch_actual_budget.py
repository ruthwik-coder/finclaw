"""
FinClaw SQLite Budget Extractor — Actual Budget Integration
============================================================
Queries Actual Budget's SQLite database (db.sqlite) and converts the live budget balances
into the exact FinClaw JSON financial_state schema for LLM context injection.
"""

import sqlite3
import json
import datetime
import calendar
import re

CATEGORY_KEYWORD_MAP = {
    # Entertainment / Streaming / Subscriptions / Games / Outings
    "netflix": "Entertainment",
    "spotify": "Entertainment",
    "disney": "Entertainment",
    "disney+": "Entertainment",
    "hulu": "Entertainment",
    "prime video": "Entertainment",
    "amazon prime": "Entertainment",
    "hbo": "Entertainment",
    "max": "Entertainment",
    "apple tv": "Entertainment",
    "youtube": "Entertainment",
    "youtube premium": "Entertainment",
    "peacock": "Entertainment",
    "paramount": "Entertainment",
    "crunchyroll": "Entertainment",
    "audible": "Entertainment",
    "patreon": "Entertainment",
    "twitch": "Entertainment",
    "discord": "Entertainment",
    "subscription": "Entertainment",
    "subscribe": "Entertainment",
    "streaming": "Entertainment",
    "stream": "Entertainment",
    "membership": "Entertainment",
    "entertainment": "Entertainment",
    "game": "Entertainment",
    "gaming": "Entertainment",
    "videogame": "Entertainment",
    "video game": "Entertainment",
    "steam": "Entertainment",
    "playstation": "Entertainment",
    "ps5": "Entertainment",
    "ps4": "Entertainment",
    "xbox": "Entertainment",
    "switch": "Entertainment",
    "nintendo": "Entertainment",
    "movie": "Entertainment",
    "cinema": "Entertainment",
    "film": "Entertainment",
    "theater": "Entertainment",
    "theatre": "Entertainment",
    "concert": "Entertainment",
    "show": "Entertainment",
    "festival": "Entertainment",
    "ticket": "Entertainment",
    "tickets": "Entertainment",
    "drinks": "Entertainment",
    "club": "Entertainment",
    "bar": "Entertainment",
    "pub": "Entertainment",
    "party": "Entertainment",
    "outing": "Entertainment",
    "karaoke": "Entertainment",
    "bowling": "Entertainment",
    "board game": "Entertainment",
    "comic": "Entertainment",
    "manga": "Entertainment",
    "novel": "Entertainment",
    "toy": "Entertainment",
    "toys": "Entertainment",

    # Food / Dining / Groceries / Delivery
    "food": "Food",
    "dinner": "Food",
    "lunch": "Food",
    "breakfast": "Food",
    "brunch": "Food",
    "takeout": "Food",
    "delivery": "Food",
    "doordash": "Food",
    "uber eats": "Food",
    "ubereats": "Food",
    "grubhub": "Food",
    "swiggy": "Food",
    "zomato": "Food",
    "groceries": "Food",
    "grocery": "Food",
    "supermarket": "Food",
    "dining": "Food",
    "restaurant": "Food",
    "cafe": "Food",
    "sushi": "Food",
    "pizza": "Food",
    "burger": "Food",
    "coffee": "Food",
    "starbucks": "Food",
    "eating out": "Food",
    "meal": "Food",
    "snack": "Food",
    "snacks": "Food",
    "dessert": "Food",

    # Personal Care / Self-care / Apparel / Fitness
    "personal care": "Personal Care",
    "personal": "Personal Care",
    "massage": "Personal Care",
    "spa": "Personal Care",
    "haircut": "Personal Care",
    "salon": "Personal Care",
    "barber": "Personal Care",
    "skincare": "Personal Care",
    "makeup": "Personal Care",
    "cosmetics": "Personal Care",
    "perfume": "Personal Care",
    "shoes": "Personal Care",
    "sneakers": "Personal Care",
    "boots": "Personal Care",
    "clothes": "Personal Care",
    "clothing": "Personal Care",
    "apparel": "Personal Care",
    "jacket": "Personal Care",
    "dress": "Personal Care",
    "shirt": "Personal Care",
    "jeans": "Personal Care",
    "gym": "Personal Care",
    "fitness": "Personal Care",
    "wellness": "Personal Care",
    "yoga": "Personal Care",
    "workout": "Personal Care",

    # Fixed obligations & Utilities
    "bills": "Bills",
    "bill": "Bills",
    "utilities": "Bills (Flexible)",
    "electricity": "Bills (Flexible)",
    "electric": "Bills (Flexible)",
    "power": "Bills (Flexible)",
    "internet": "Bills (Flexible)",
    "wifi": "Bills (Flexible)",
    "broadband": "Bills (Flexible)",
    "water": "Bills (Flexible)",
    "phone": "Bills (Flexible)",
    "mobile": "Bills (Flexible)",
    "insurance": "Bills (Flexible)",
    "rent": "Rent/housing",
    "housing": "Rent/housing",
    "apartment": "Rent/housing",
    "lease": "Rent/housing",
    "mortgage": "Rent/housing",

    # Savings / Goals
    "savings": "Savings",
    "emergency fund": "Emergency Savings",
    "investment": "Savings",

    # General / Shopping
    "general": "General",
    "shopping": "General",
    "stuff": "General",
    "amazon": "General",
    "gadget": "General",
    "accessories": "General"
}


def resolve_category_name(query: str, available_categories: list) -> str:
    """
    Resolves free-form keyword or category string to one of the exact categories in db.sqlite.
    """
    clean = query.lower().strip()
    
    # 1. Exact match
    for cat in available_categories:
        if clean == cat.lower():
            return cat
            
    # 2. Keyword dictionary lookup
    for kw, mapped_cat in CATEGORY_KEYWORD_MAP.items():
        if kw in clean:
            for cat in available_categories:
                if mapped_cat.lower() == cat.lower():
                    return cat

    # 3. Partial substring match in available categories
    for cat in available_categories:
        if clean in cat.lower() or cat.lower() in clean:
            return cat

    # Default fallback
    return "General" if "General" in available_categories else available_categories[0]


def fetch_finclaw_ledger(db_path="db.sqlite", category_query="Food", item_name="Item", item_cost=0.0, month_str="202609"):
    """
    Queries Actual Budget db.sqlite and formats the deterministic financial ledger.
    """
    month_int = int(month_str)
    year = int(month_str[:4])
    month = int(month_str[4:])

    # Calculate days remaining in that month
    days_in_month = calendar.monthrange(year, month)[1]
    today = datetime.datetime.now()
    if today.year == year and today.month == month:
        period_days_remaining = max(1, days_in_month - today.day)
    else:
        period_days_remaining = max(1, days_in_month - 27)  # default for late September

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # 1. Fetch liquid checking balance & emergency savings balance from accounts
    cursor.execute("""
        SELECT a.name, SUM(t.amount) 
        FROM accounts a
        LEFT JOIN transactions t ON a.id = t.acct AND t.tombstone = 0
        WHERE a.closed = 0
        GROUP BY a.id
    """)
    
    checking_balance = 0.0
    savings_balance = 0.0

    for acct_name, total_cents in cursor.fetchall():
        total_dollars = (total_cents or 0) / 100.0
        if "Checking" in acct_name or "checking" in acct_name.lower():
            checking_balance += total_dollars
        elif "Savings" in acct_name or "savings" in acct_name.lower() or "emergency" in acct_name.lower():
            savings_balance += total_dollars

    # 2. Fetch available categories
    cursor.execute("SELECT id, name FROM categories WHERE name NOT IN ('Starting Balances', 'Income')")
    cat_rows = cursor.fetchall()
    cat_map = {row[1]: row[0] for row in cat_rows}
    available_cats = list(cat_map.keys())

    # Resolve category
    resolved_cat_name = resolve_category_name(category_query, available_cats)
    cat_id = cat_map[resolved_cat_name]

    # 3. Fetch allocated limit for current month from zero_budgets
    cursor.execute("SELECT amount FROM zero_budgets WHERE category = ? AND month = ?", (cat_id, month_int))
    budget_row = cursor.fetchone()
    allocated_limit = (budget_row[0] / 100.0) if budget_row else 0.0

    # 4. Fetch spent to date for this category and month from transactions
    start_date = month_int * 100 + 1
    end_date = month_int * 100 + days_in_month
    cursor.execute("""
        SELECT SUM(amount) 
        FROM transactions 
        WHERE category = ? AND tombstone = 0 AND amount < 0 AND date >= ? AND date <= ?
    """, (cat_id, start_date, end_date))
    spent_row = cursor.fetchone()
    spent_to_date = abs((spent_row[0] or 0) / 100.0)

    conn.close()

    # 5. Compute derived financial metrics
    remaining_balance = round(allocated_limit - spent_to_date, 2)
    post_purchase_balance = round(remaining_balance - item_cost, 2)
    exceeds_category_by = max(0.0, round(item_cost - remaining_balance, 2))

    financial_state = {
        "currency": "USD",
        "budget_period": "monthly",
        "period_days_remaining": period_days_remaining,
        "accounts": {
            "liquid_checking_balance": round(checking_balance, 2),
            "savings_emergency_balance": round(savings_balance, 2)
        },
        "category_envelope": {
            "category_name": resolved_cat_name.lower().replace(" ", "_").replace("/", "_"),
            "allocated_limit": allocated_limit,
            "spent_to_date": spent_to_date,
            "remaining_balance": remaining_balance
        },
        "decision_transaction": {
            "item_name": item_name,
            "estimated_cost": item_cost,
            "post_purchase_category_balance": post_purchase_balance,
            "exceeds_category_by": exceeds_category_by
        }
    }

    return financial_state


def fetch_budget_summary(db_path="db.sqlite", month_str="202609") -> dict:
    """
    Returns an overview of all active budget categories for the month.
    """
    month_int = int(month_str)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT c.name, 
               COALESCE(b.amount, 0) / 100.0 AS budgeted,
               COALESCE(ABS(SUM(CASE WHEN t.amount < 0 AND t.tombstone = 0 THEN t.amount ELSE 0 END)), 0) / 100.0 AS spent
        FROM categories c
        LEFT JOIN zero_budgets b ON c.id = b.category AND b.month = ?
        LEFT JOIN transactions t ON c.id = t.category AND t.date >= ? AND t.date <= ?
        WHERE c.name NOT IN ('Starting Balances', 'Income')
        GROUP BY c.id
        ORDER BY budgeted DESC
    """, (month_int, month_int * 100 + 1, month_int * 100 + 31))

    summary = []
    for name, budgeted, spent in cursor.fetchall():
        balance = round(budgeted - spent, 2)
        summary.append({
            "category": name,
            "budgeted": budgeted,
            "spent": spent,
            "balance": balance
        })

    conn.close()
    return summary


if __name__ == "__main__":
    print("=" * 65)
    print("LIVE BUDGET SNAPSHOT FROM ACTUAL BUDGET SQLITE FILE (db.sqlite)")
    print("=" * 65)
    
    summary = fetch_budget_summary("db.sqlite", "202609")
    for row in summary:
        print(f" • {row['category']:<20}: Budgeted=${row['budgeted']:<7.2f} Spent=${row['spent']:<7.2f} Balance=${row['balance']:<7.2f}")
        
    print("\n" + "=" * 65)
    print("SAMPLE LEDGER EXTRACTION FOR A $60 ENTERTAINMENT EXPENSE:")
    print("=" * 65)
    ledger = fetch_finclaw_ledger(db_path="db.sqlite", category_query="gaming", item_name="video game", item_cost=60.0, month_str="202609")
    print(json.dumps(ledger, indent=2))
