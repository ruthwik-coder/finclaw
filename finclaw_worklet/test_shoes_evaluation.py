import os
import sys
import json

sys.stdout.reconfigure(encoding="utf-8")
os.environ["HF_HOME"] = os.path.abspath(os.path.join(os.path.dirname(__file__), ".cache", "huggingface"))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

ADAPTER_DIR = "./finclaw_finetuned_adapter"
BASE_MODEL_ID = "unsloth/Llama-3.2-1B-Instruct-bnb-4bit"

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

# The evaluation test query from the user's image
user_query = "I have ₹12,000 left from my salary. I need to pay rent next week, but I haven't calculated the exact rent amount yet. I'm thinking of buying ₹6,000 shoes. Should I buy them?"

# Grounded Financial Ledger Representation
ledger = {
  "currency": "INR",
  "budget_period": "monthly",
  "period_days_remaining": 7,
  "accounts": {
    "liquid_checking_balance": 12000.0,
    "savings_emergency_balance": 0.0,
    "pending_fixed_obligations": "rent due next week (amount uncalculated / unknown)"
  },
  "category_envelope": {
    "category_name": "apparel",
    "allocated_limit": 2000.0,
    "spent_to_date": 0.0,
    "remaining_balance": 2000.0
  },
  "decision_transaction": {
    "item_name": "shoes",
    "estimated_cost": 6000.0,
    "post_purchase_category_balance": -4000.0,
    "exceeds_category_by": 4000.0,
    "post_purchase_checking_balance": 6000.0
  }
}

system_prompt = f"""You are FinClaw, an emotionally-aware financial coaching bot.
Each sample links a psychological/emotional scenario to an exact financial ledger state and a multi-turn coaching conversation.

CURRENT USER FINANCIAL LEDGER:
{json.dumps(ledger, indent=2)}

Provide empathetic emotional validation, analyze their financial ledger state, and guide their financial decision."""

messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": user_query}
]

prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
inputs = tokenizer(prompt_text, return_tensors="pt").to(device)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=350,
        temperature=0.2,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )

response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
print("=" * 60)
print("INPUT QUERY:")
print(user_query)
print("\nJSON LEDGER:")
print(json.dumps(ledger, indent=2))
print("=" * 60)
print("MODEL RESPONSE:")
print(response.strip())
print("=" * 60)
