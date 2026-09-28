"""
FinClaw Backend Helper & SQLite Integration Engine
===================================================
1. Automatically detects purchase intent, item name, cost, and target category.
2. Directly queries Actual Budget SQLite database (db.sqlite) to retrieve:
   - Liquid Checking Balance ($1,080.00)
   - Emergency Savings Balance ($5,000.00)
   - Target Category Limit & Spent-to-Date
   - Computed Remaining Balance & Exceeds-By Metric
3. Injects this deterministic financial ledger into the locked FinClaw training system prompt.
4. Generates empathetic, mathematically grounded financial coaching advice.
"""

import os
import sys
import re
import json

sys.stdout.reconfigure(encoding="utf-8")
os.environ["HF_HOME"] = os.path.abspath(os.path.join(os.path.dirname(__file__), ".cache", "huggingface"))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from fetch_actual_budget import fetch_finclaw_ledger, fetch_budget_summary, CATEGORY_KEYWORD_MAP

ADAPTER_DIR = "./finclaw_finetuned_adapter"
BASE_MODEL_ID = "unsloth/Llama-3.2-1B-Instruct-bnb-4bit"
DB_PATH = "db.sqlite"


def extract_transaction_intent(text: str):
    """
    Parses conversational text to extract:
    - Item name
    - Proposed cost
    - Target budget category in db.sqlite
    """
    cleaned = re.sub(r'(\d),(\d)', r'\1\2', text)
    
    # 1. Cost extraction ($50, ₹8500, 50 dollars, 50 bucks, etc.)
    cost = 0.0
    m_cost = re.search(r'(?:[\$₹]|rs\.?\s*)(\d+(?:\.\d+)?)|(\d+(?:\.\d+)?)\s*(?:dollars|bucks|usd|inr|rs)', cleaned, re.I)
    if m_cost:
        cost = float(m_cost.group(1) or m_cost.group(2))
    else:
        m_num = re.search(r'(?:for|costs?|costing|spend|spending|buy|buying|pay|worth)\s+(?:about|around)?\s*(\d+(?:\.\d+)?)', cleaned, re.I)
        if m_num:
            cost = float(m_num.group(1))

    # 2. Category matching via keywords
    detected_cat = "General"
    item_name = "item"
    text_lower = cleaned.lower()

    for kw, mapped_cat in CATEGORY_KEYWORD_MAP.items():
        if re.search(r'\b' + re.escape(kw) + r'\b', text_lower):
            detected_cat = mapped_cat
            item_name = kw
            break

    # Look for specific item phrases (e.g., 'craving a special dinner', 'buy an $80 video game')
    m_item = re.search(r'(?:craving|buy|buying|purchase|ordering|order|get)\s+(?:a|an)?\s*(?:[\$₹]?\d+\s*)?([a-zA-Z\s]{3,25}?)(?:\s+(?:for|tonight|at|on|because|since)|$)', cleaned, re.I)
    if m_item:
        cand = m_item.group(1).strip()
        if len(cand) > 2 and cand.lower() not in ['it', 'them', 'that', 'this', 'something']:
            item_name = cand

    return item_name, cost, detected_cat


def fallback_extract_ledger(text: str) -> dict:
    """
    Fallback regex parser when db.sqlite is unavailable.
    """
    cleaned = re.sub(r'(\d),(\d)', r'\1\2', text)
    m_bal = re.search(r'(?:have|left|account|balance|checking)\s+(?:around|about|approx)?\s*(?:[₹$]|rs\.?\s*)?(\d+(?:\.\d+)?)', cleaned, re.I)
    liquid_balance = float(m_bal.group(1)) if m_bal else 1000.0
    item_name, cost, cat = extract_transaction_intent(text)

    return {
        "currency": "USD",
        "budget_period": "monthly",
        "period_days_remaining": 3,
        "accounts": {
            "liquid_checking_balance": liquid_balance,
            "savings_emergency_balance": 5000.0
        },
        "category_envelope": {
            "category_name": cat.lower().replace(" ", "_"),
            "allocated_limit": round(liquid_balance * 0.3, 2),
            "spent_to_date": 0.0,
            "remaining_balance": round(liquid_balance * 0.3, 2)
        },
        "decision_transaction": {
            "item_name": item_name,
            "estimated_cost": cost,
            "post_purchase_category_balance": round(liquid_balance * 0.3 - cost, 2),
            "exceeds_category_by": max(0.0, round(cost - liquid_balance * 0.3, 2))
        }
    }


class FinClawCoTEngine:
    def __init__(self):
        print("=" * 65)
        print("  🐾 INITIALIZING FINCLAW COACHING ENGINE")
        print("=" * 65)
        print(f"Loading adapter weights from {ADAPTER_DIR} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(ADAPTER_DIR)
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True
        )
        device = "cuda" if torch.cuda.is_available() else "cpu"
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_ID,
            torch_dtype=torch.bfloat16,
            quantization_config=bnb_config,
            device_map={"": device} if device == "cpu" else "auto"
        )
        self.model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)
        self.model.eval()
        self.device = device
        print(f"  ✓ FinClaw model loaded successfully on device: {self.device}")

    def consult(self, user_query: str) -> dict:
        item_name, cost, category = extract_transaction_intent(user_query)

        # 1. Fetch live ground truth from SQLite db.sqlite
        source = "heuristic fallback"
        if os.path.exists(DB_PATH):
            try:
                ledger = fetch_finclaw_ledger(
                    db_path=DB_PATH,
                    category_query=category,
                    item_name=item_name,
                    item_cost=cost,
                    month_str="202609"
                )
                source = f"SQLite ({DB_PATH}) -> Category: '{category}'"
            except Exception as e:
                print(f"  [Warning: SQLite lookup fallback: {e}]")
                ledger = fallback_extract_ledger(user_query)
        else:
            ledger = fallback_extract_ledger(user_query)

        # 2. Inject live ledger into locked training system prompt layout
        system_prompt = f"""You are FinClaw, an emotionally-aware financial coaching bot.
Each sample links a psychological/emotional scenario to an exact financial ledger state and a multi-turn coaching conversation.

CURRENT USER FINANCIAL LEDGER:
{json.dumps(ledger, indent=2)}

Provide empathetic emotional validation, analyze their financial ledger state, and guide their financial decision."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query}
        ]

        prompt_text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt_text, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=300,
                temperature=0.3,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )

        response = self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

        return {
            "source": source,
            "extracted_ledger": ledger,
            "response": response.strip(),
            "category": category,
            "cost": cost
        }


if __name__ == "__main__":
    # Display the current live budget summary from db.sqlite
    if os.path.exists(DB_PATH):
        print("\n" + "=" * 65)
        print("  📊 LIVE BUDGET PORTFOLIO (September 2026 from db.sqlite)")
        print("=" * 65)
        summary = fetch_budget_summary(DB_PATH, "202609")
        for row in summary:
            print(f"  • {row['category']:<20}: Budgeted=${row['budgeted']:<7.2f} Spent=${row['spent']:<7.2f} Balance=${row['balance']:<7.2f}")
        print("=" * 65)

    engine = FinClawCoTEngine()

    # Pre-flight demonstration query using real budget
    demo_query = "I've had a super stressed day at work and I'm really craving a special dinner tonight for $50, but I don't know if I should spend it."
    print("\n" + "=" * 65)
    print(f"DEMO USER QUERY: {demo_query}")
    print("=" * 65)

    result = engine.consult(demo_query)
    print(f"\n[DATA SOURCE]: {result['source']}")
    print("\n[INJECTED SQLITE FINANCIAL LEDGER]:")
    print(json.dumps(result["extracted_ledger"], indent=2))
    print("\n[FINCLAW COACHING ADVICE]:")
    print(result["response"])
    print("=" * 65)

    # Interactive Live Chat Loop
    print("\n" + "=" * 65)
    print("  💬 FINCLAW INTERACTIVE BUDGET COACHING")
    print("  Ask any spending question (or press Enter to exit)")
    print("=" * 65)

    while True:
        try:
            user_input = input("\nYou: ").strip()
            if not user_input:
                print("Exiting interactive mode. Goodbye!")
                break

            result = engine.consult(user_input)
            print(f"\n[DATA SOURCE]: {result['source']}")
            print("\n[FINCLAW LEDGER]:")
            print(json.dumps(result["extracted_ledger"], indent=2))
            print("\n[FINCLAW RESPONSE]:")
            print(result["response"])
            print("-" * 50)
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break
