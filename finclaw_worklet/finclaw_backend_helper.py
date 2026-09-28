"""
FinClaw Backend Extraction Helper & Chain-of-Thought Engine
===========================================================
1. Automatically extracts financial variables from free-form user messages into a structured ledger.
2. Applies a Chain-of-Thought (CoT) prompt template to force grounded arithmetic.
3. Queries the fine-tuned FinClaw model and returns structured advice.
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

ADAPTER_DIR = "./finclaw_finetuned_adapter"
BASE_MODEL_ID = "unsloth/Llama-3.2-1B-Instruct-bnb-4bit"

# -------------------------------------------------------------
# 1. Backend Extraction Helper (Regex & Parsing Logic)
# -------------------------------------------------------------

def extract_financial_ledger(text: str) -> dict:
    """
    Parses conversational text to extract:
    - Currency
    - Current balance or Total income
    - Pending obligations
    - Proposed item and cost
    - Calculates post-purchase checking balance
    """
    # Detect currency
    currency = "INR"
    if "$" in text:
        currency = "USD"
    elif "sgd" in text.lower():
        currency = "SGD"

    # Clean numbers with commas (e.g. 18,000 -> 18000)
    cleaned = re.sub(r'(\d),(\d)', r'\1\2', text)

    # 1. Current liquid balance left
    m_bal = re.search(r'(?:have|left|account|balance|checking)\s+(?:around|about|approx)?\s*(?:[₹$]|rs\.?\s*)?(\d+(?:\.\d+)?)', cleaned, re.I)
    liquid_balance = float(m_bal.group(1)) if m_bal else 0.0

    # 2. Check for pending obligations & exact amounts (e.g. credit card bill of ₹12,000, utility bills of ₹7,000)
    pending_bills_amount = 0.0
    pending_obligations = "None reported"

    m_bill = re.search(r'([a-zA-Z\s,]+?(?:bill|bills|rent|groceries|utilities|fees|expenses))\s*(?:of|about|is|due|worth|left to cover)?\s*(?:about|around)?\s*(?:[₹$]|rs\.?\s*)?(\d+(?:\.\d+)?)', cleaned, re.I)
    if m_bill:
        desc = m_bill.group(1).strip()
        # Clean leading filler words
        for lead in ['but', 'i', 'have', 'still', 'my', 'and', 'right', 'now']:
            words = desc.split()
            if words and words[0].lower() == lead:
                desc = ' '.join(words[1:])
        pending_bills_amount = float(m_bill.group(2))
        pending_obligations = f"{desc} ({currency} {pending_bills_amount:,.2f} pending)"
    elif "rent" in text.lower():
        if "haven't calculated" in text.lower() or "not calculated" in text.lower() or "exact" in text.lower():
            pending_obligations = "Rent due next week (exact amount uncalculated / pending)"
        else:
            pending_obligations = "Rent and bills pending"

    # 3. Proposed purchase item & cost
    item_name = "item"
    estimated_cost = 0.0

    # Pattern A: 'headphones on sale for ₹8500' or 'trip that costs ₹14000' or 'shoes costing ₹6000'
    m_cost1 = re.search(r'([a-zA-Z\s-]+?)\s+(?:on sale for|costs?|costing|priced at)\s*(?:[₹$]|rs\.?\s*)?(\d+(?:\.\d+)?)', cleaned, re.I)
    # Pattern B: 'buying ₹6,000 shoes' or 'buying a ₹9,000 smartwatch'
    m_cost2 = re.search(r'(?:buying|buy|purchase|get|ordering)\s*(?:a|an)?\s*(?:[₹$]|rs\.?\s*)?(\d+(?:\.\d+)?)\s*([a-zA-Z\s-]+)', cleaned, re.I)
    # Pattern C: 'buying shoes for ₹6,000'
    m_cost3 = re.search(r'(?:buying|buy|purchase|get)\s+([a-zA-Z\s]+?)\s+(?:for|at)\s*(?:[₹$]|rs\.?\s*)?(\d+(?:\.\d+)?)', cleaned, re.I)
    # Pattern D: 'subscription for my kid worth 60' or 'item worth 60'
    m_cost4 = re.search(r'(?:buy|buying|purchase|get|ordering|craving|planning to buy|planning on buying)\s+(?:a|an)?\s*([a-zA-Z\s-]+?)\s+(?:worth|priced at|costs?|costing)\s*(?:[₹$]|rs\.?\s*)?(\d+(?:\.\d+)?)', cleaned, re.I)

    if m_cost4:
        item_name = m_cost4.group(1).strip()
        estimated_cost = float(m_cost4.group(2))
    elif m_cost1:
        words = [w for w in m_cost1.group(1).strip().split() if w.lower() not in ['a', 'an', 'pair', 'of', 'saw', 'the', 'that', 'booking', 'are', 'in']]
        item_name = ' '.join(words[-2:]) if words else "item"
        estimated_cost = float(m_cost1.group(2))
    elif m_cost2:
        estimated_cost = float(m_cost2.group(1))
        words = [w for w in m_cost2.group(2).strip().split() if w.lower() not in ['because', 'since', 'for', 'to', 'that']]
        item_name = ' '.join(words[:2]) if words else "item"
    elif m_cost3:
        item_name = m_cost3.group(1).strip()
        estimated_cost = float(m_cost3.group(2))


    # 4. Calculate grounded financial metrics
    effective_balance_after_bills = liquid_balance - pending_bills_amount
    post_purchase_balance = effective_balance_after_bills - estimated_cost

    # Category envelope calculation
    allocated_limit = max(0.0, round(liquid_balance * 0.25, 2))
    exceeds_by = max(0.0, estimated_cost - allocated_limit)

    return {
        "currency": currency,
        "budget_period": "monthly",
        "period_days_remaining": 7 if "next week" in text.lower() else 15,
        "accounts": {
            "liquid_checking_balance": liquid_balance,
            "savings_emergency_balance": 0.0,
            "pending_fixed_obligations": pending_obligations
        },
        "category_envelope": {
            "category_name": item_name or "discretionary",
            "allocated_limit": allocated_limit,
            "spent_to_date": 0.0,
            "remaining_balance": allocated_limit
        },
        "decision_transaction": {
            "item_name": item_name,
            "estimated_cost": estimated_cost,
            "post_purchase_category_balance": allocated_limit - estimated_cost,
            "exceeds_category_by": exceeds_by,
            "post_purchase_checking_balance": post_purchase_balance
        }
    }


# -------------------------------------------------------------
# 2. Chain-of-Thought (CoT) Inference Runner
# -------------------------------------------------------------

class FinClawCoTEngine:
    def __init__(self):
        print(f"Loading FinClaw model from {ADAPTER_DIR} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(ADAPTER_DIR)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True
        )
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_ID,
            torch_dtype=torch.bfloat16,
            quantization_config=bnb_config,
            device_map="auto"
        )
        self.model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)
        self.model.eval()

    def consult(self, user_query: str) -> dict:
        # Step A: Auto-extract ledger
        ledger = extract_financial_ledger(user_query)

        # Step B: Build CoT System Prompt
        system_prompt = f"""You are FinClaw, an emotionally-aware financial coaching bot.
Each sample links a psychological/emotional scenario to an exact financial ledger state and a multi-turn coaching conversation.

CURRENT USER FINANCIAL LEDGER:
{json.dumps(ledger, indent=2)}

Before giving advice, you MUST verify the variables in your reasoning:
- Current Liquid Balance: {ledger['currency']} {ledger['accounts']['liquid_checking_balance']}
- Pending Obligations: {ledger['accounts']['pending_fixed_obligations']}
- Proposed Purchase: {ledger['decision_transaction']['item_name']} costing {ledger['currency']} {ledger['decision_transaction']['estimated_cost']}
- Balance Remaining After Purchase: {ledger['currency']} {ledger['decision_transaction']['post_purchase_checking_balance']}

Provide empathetic emotional validation, analyze their financial ledger state, and guide their financial decision."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query}
        ]

        prompt_text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt_text, return_tensors="pt").to("cuda")

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=350,
                temperature=0.2,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )

        response = self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

        return {
            "extracted_ledger": ledger,
            "response": response.strip()
        }


if __name__ == "__main__":
    engine = FinClawCoTEngine()

    test_queries = [
        "I've had an extraordinarily exhausting week at work and I'm feeling completely burned out. I saw a pair of noise-canceling headphones on sale for ₹8,500 that ends tonight. I have ₹22,000 left in my account, but my credit card bill of ₹12,000 is due in 10 days. I feel like I deserve this reward to stay sane, but I'm torn. What should I do?",
        "All my friends are booking a weekend getaway trip that costs ₹14,000. I have ₹19,000 in my checking account right now, but I still have groceries and utility bills of about ₹7,000 left to cover this month. I'll feel awful and left out if I say no, but I'm anxious about money. Should I join them?"
    ]

    for q in test_queries:
        print("\n" + "=" * 70)
        print(f"USER QUERY: {q}")
        print("=" * 70)

        result = engine.consult(q)

        print("\n[EXTRACTED FINANCIAL LEDGER]")
        print(json.dumps(result["extracted_ledger"], indent=2))

        print("\n[FINCLAW ADVICE]")
        print(result["response"])

    print("\n" + "=" * 70)
    print(" INTERACTIVE MODE ")
    print("Type any free-form financial question (or press Enter to exit):")
    print("=" * 70)

    while True:
        try:
            user_input = input("\nYou: ").strip()
            if not user_input:
                print("Exiting interactive mode.")
                break

            result = engine.consult(user_input)
            print("\n[AUTO-GENERATED LEDGER]")
            print(json.dumps(result["extracted_ledger"], indent=2))
            print("\n[FINCLAW RESPONSE]")
            print(result["response"])
            print("-" * 50)
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break
