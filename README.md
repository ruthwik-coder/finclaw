# 🐾 FinClaw: Emotionally-Aware Financial Coaching AI

[![Model](https://img.shields.io/badge/Model-Llama--3.2--1B--Instruct-blue.svg)](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct)
[![Fine-Tuning](https://img.shields.io/badge/Method-QLoRA%20(4--bit)-green.svg)](https://github.com/huggingface/peft)
[![Precision](https://img.shields.io/badge/Precision-Native%20BF16-orange.svg)]()
[![License](https://img.shields.io/badge/License-Apache%202.0-lightgrey.svg)]()

FinClaw is a specialized, fine-tuned financial coaching AI built on **Meta's Llama-3.2-1B-Instruct** using 4-bit QLoRA. It links emotional/psychological user scenarios (impulse spending, stress, burnout, FOMO) with exact financial ledger states to provide **grounded, empathetic, and responsible financial guidance**.

---

## 🚀 Quick Start for Developers & AI Agents

### 1. Installation
```bash
pip install -r requirements.txt
```

### 2. Integrate the Model in 3 Lines of Code
The fine-tuned adapter is contained in [`finclaw_finetuned_adapter/`](./finclaw_finetuned_adapter):

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# 1. Load the 4-bit base model
base_model = AutoModelForCausalLM.from_pretrained(
    "unsloth/Llama-3.2-1B-Instruct-bnb-4bit",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("./finclaw_finetuned_adapter")

# 2. Attach the fine-tuned FinClaw adapter
model = PeftModel.from_pretrained(base_model, "./finclaw_finetuned_adapter")
model.eval()
```

---

## 🧠 The Input Contract (Financial Ledger)

FinClaw expects a structured JSON ledger in the **System Prompt**. In production applications, your backend database should supply these calculated figures:

```json
CURRENT USER FINANCIAL LEDGER:
{
  "currency": "INR",
  "budget_period": "monthly",
  "period_days_remaining": 15,
  "accounts": {
    "liquid_checking_balance": 18000.0,
    "savings_emergency_balance": 0.0,
    "pending_fixed_obligations": "rent and bills pending"
  },
  "category_envelope": {
    "category_name": "gadgets",
    "allocated_limit": 3000.0,
    "spent_to_date": 0.0,
    "remaining_balance": 3000.0
  },
  "decision_transaction": {
    "item_name": "smartwatch",
    "estimated_cost": 9000.0,
    "post_purchase_category_balance": -6000.0,
    "exceeds_category_by": 6000.0,
    "post_purchase_checking_balance": 9000.0
  }
}
```

---

## 🛠️ Extra Feature: Heuristic Natural-Language Parser

> [!NOTE]
> **Optional Testing Tool:**  
> The core deliverable is the fine-tuned adapter weights. To test the model with free-form English without writing JSON manually, we built [`finclaw_backend_helper.py`](./finclaw_backend_helper.py).  
> *Disclaimer:* This parser uses regex heuristics and is intended as a developer testing prototype, not a replacement for a live banking database feed.

Run the interactive chat interface:
```bash
python finclaw_backend_helper.py
```

---

## 📊 Training Metrics & History

The model was fine-tuned locally on an **NVIDIA GeForce RTX 3050 Laptop GPU (4 GB VRAM)** using native **BF16** mixed precision and Paged AdamW 8-bit optimizer.

* **Dataset:** [`finclaw_colab_dataset.jsonl`](./finclaw_colab_dataset.jsonl) (758 multi-turn conversations).
* **Training Iterations:**
  * **Run 1 (Baseline):** 100 steps (~0.5 epochs) | Runtime: 5.8m | Final loss: `0.799`.
  * **Run 2 (Deep Grounding):** 350 steps (~1.85 epochs) | Runtime: 20.6m | Final loss: **`0.676`** | Accuracy: **`82.72%`**.
* **Key Strengths:**
  * Resists hallucinating unknown obligations (e.g. uncalculated rent).
  * Accurately factors in category deficits and remaining liquid balances.
  * Recommends cooling-off periods rather than impulsive approvals.

For complete technical logs, hardware adaptations, and evaluation scorecards, see **[FINCLAW_PROJECT_REPORT.md](./FINCLAW_PROJECT_REPORT.md)**.

---

## 📂 Repository Structure

| File / Folder | Purpose |
| :--- | :--- |
| [`finclaw_finetuned_adapter/`](./finclaw_finetuned_adapter) | **The trained LoRA adapter weights** (`adapter_model.safetensors`, tokenizer, configs). |
| [`finclaw_finetuned_adapter.zip`](./finclaw_finetuned_adapter.zip) | Portable 19.3 MB zip archive of the model weights for instant sharing. |
| [`finclaw_backend_helper.py`](./finclaw_backend_helper.py) | Optional natural-language parser + interactive chat engine. |
| [`train_local.py`](./train_local.py) | Standalone local training script with SFTTrainer & BF16. |
| [`Fine_Tuning_FinClaw.ipynb`](./Fine_Tuning_FinClaw.ipynb) | End-to-end Jupyter Notebook pipeline for training & inference. |
| [`test_shoes_evaluation.py`](./test_shoes_evaluation.py) | Grounded reasoning benchmark script (rent/shoes scenario). |
| [`test_local_inference.py`](./test_local_inference.py) | Clean testing script for custom JSON ledger scenarios. |
| [`FINCLAW_PROJECT_REPORT.md`](./FINCLAW_PROJECT_REPORT.md) | In-depth engineering changelog and training report. |
| [`finclaw_colab_dataset.jsonl`](./finclaw_colab_dataset.jsonl) | The 758-sample synthetic training dataset. |

---

## 📜 License
Apache-2.0 License. Base model weights are governed by the Meta Llama 3.2 Community License Agreement.
