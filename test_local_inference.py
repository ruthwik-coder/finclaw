"""
FinClaw Local Inference Engine
==============================
Interacts with the fine-tuned FinClaw model using the trained Financial Ledger format.
"""

import os
import sys
import json

sys.stdout.reconfigure(encoding="utf-8")
os.environ["HF_HOME"] = os.path.abspath(os.path.join(os.path.dirname(__file__), ".cache", "huggingface"))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

BASE_MODEL_ID = "unsloth/Llama-3.2-1B-Instruct-bnb-4bit"
ADAPTER_DIR = "./finclaw_finetuned_adapter"

print("=" * 60)
print(" Loading FinClaw Local Coaching Model ")
print("=" * 60)

tokenizer = AutoTokenizer.from_pretrained(ADAPTER_DIR)

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

model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)
model.eval()

def generate_finclaw_advice(user_message: str, ledger: dict) -> str:
    """
    Formats the prompt using the exact financial ledger schema FinClaw was trained on.
    """
    system_prompt = f"""You are FinClaw, an emotionally-aware financial coaching bot.
Each sample links a psychological/emotional scenario to an exact financial ledger state and a multi-turn coaching conversation.

CURRENT USER FINANCIAL LEDGER:
{json.dumps(ledger, indent=2)}

Provide empathetic emotional validation, analyze their financial ledger state, and guide their financial decision."""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message}
    ]

    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt_text, return_tensors="pt").to("cuda")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=350,
            temperature=0.3,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return response.strip()

if __name__ == "__main__":
    # Test case: User's smartwatch dilemma with structured financial ledger
    user_query = (
        "I received ₹35,000 this month and currently have around ₹18,000 left. "
        "I've already spent ₹7,000 on shopping and ₹4,500 on eating out. "
        "My rent and bills for the month are still pending, and I'm thinking of buying a "
        "₹9,000 smartwatch because there's a limited-time discount. "
        "I really want it, but I'm worried I might regret spending the money later. "
        "What should I consider before buying it?"
    )

    ledger_data = {
        "currency": "INR",
        "budget_period": "monthly",
        "period_days_remaining": 15,
        "accounts": {
            "total_income": 35000.0,
            "liquid_checking_balance": 18000.0,
            "pending_obligations": "rent and utility bills still unpaid"
        },
        "category_envelope": {
            "shopping_spent": 7000.0,
            "dining_out_spent": 4500.0,
            "gadgets_allocated_limit": 3000.0,
            "gadgets_spent_to_date": 0.0,
            "gadgets_remaining": 3000.0
        },
        "decision_transaction": {
            "item_name": "smartwatch",
            "estimated_cost": 9000.0,
            "post_purchase_category_balance": -6000.0,
            "exceeds_category_by": 6000.0,
            "post_purchase_checking_balance": 9000.0,
            "pct_of_remaining_checking_spent": "50%"
        }
    }

    print("\n" + "=" * 60)
    print("User Scenario:")
    print(user_query)
    print("\nProvided Financial Ledger:")
    print(json.dumps(ledger_data, indent=2))
    print("=" * 60)

    print("\nGenerating FinClaw Coaching Advice...\n")
    advice = generate_finclaw_advice(user_query, ledger_data)
    print(advice)
