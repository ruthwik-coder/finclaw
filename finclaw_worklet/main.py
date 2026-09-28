"""
FinClaw Main Application Backend — Samsung PRISM
================================================
Binds together:
1. Live SQLite database budget fetcher (fetch_actual_budget.py)
2. Fine-Tuned FinClaw Model running on Ollama

Usage:
  python main.py
"""

import json
import requests
from fetch_actual_budget import fetch_finclaw_ledger

# Ollama local endpoint URL (default is http://localhost:11434/api/generate)
OLLAMA_API_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "finclaw"  # Name of your fine-tuned model in Ollama

def ask_finclaw(user_message: str, category_query: str, item_name: str, item_cost: float):
    print(f"\n[1/3] Querying live SQLite database (db.sqlite) for category '{category_query}'...")
    
    # 1. Fetch live SQLite budget
    try:
        ledger = fetch_finclaw_ledger(
            db_path="db.sqlite", 
            category_query=category_query, 
            item_name=item_name, 
            item_cost=item_cost,
            month_str="202609"
        )
        print("  ✓ Successfully retrieved live financial state from SQLite.")
    except Exception as e:
        print(f"  ❌ Error fetching from SQLite: {e}")
        return

    # 2. Construct System Prompt (Exact format used during fine-tuning)
    system_prompt = f"""You are FinClaw, an emotionally-aware financial coaching bot.
Each sample links a psychological/emotional scenario to an exact financial ledger state and a multi-turn coaching conversation.

CURRENT USER FINANCIAL LEDGER:
{json.dumps(ledger, indent=2)}

Provide empathetic emotional validation, analyze their financial ledger state, and guide their financial decision."""

    full_prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_message}<|im_end|>\n<|im_start|>assistant\n"

    print(f"[2/3] Passing injected SQLite prompt to Fine-Tuned FinClaw Model on Ollama...")

    # 3. Call Fine-Tuned Model on Ollama
    payload = {
        "model": MODEL_NAME,
        "prompt": full_prompt,
        "stream": False,
        "options": {
            "temperature": 0.7
        }
    }

    try:
        response = requests.post(OLLAMA_API_URL, json=payload)
        res_json = response.json()
        model_reply = res_json.get("response", "")
        
        print("\n" + "=" * 65)
        print("FINCLAW COACHING RESPONSE (LIVE)")
        print("=" * 65)
        print(model_reply)
        print("=" * 65)
        return model_reply
        
    except requests.exceptions.ConnectionError:
        print("\n❌ Could not connect to Ollama. Make sure Ollama is running (`ollama run finclaw`).")


if __name__ == "__main__":
    # Test query
    user_msg = "I've had a super stressed day at work and I'm really craving a special dinner tonight for $50, but I don't know if I should spend it."
    ask_finclaw(user_message=user_msg, category_query="Food", item_name="Special Dinner", item_cost=50.0)
