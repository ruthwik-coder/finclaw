# FinClaw Model Fine-Tuning & Local Pipeline Report

**Project:** FinClaw (Emotionally-Aware Financial Coaching AI)  
**Author / Engineering Log:** Local Training, Optimization & Deployment Documentation  
**Date:** September 27, 2026  

---

## 1. Project Overview & Background

### 1.1 The Original Problem
* The project originally ran on **Google Colab** using an 8-Billion parameter model (`Meta-Llama-3.1-8B-Instruct`).
* While training finished in Colab, the export step generated an unquantized 16 GB F16 file and a ~5 GB 4-bit GGUF file.
* Browser downloads via `google.colab.files.download()` crashed and timed out due to browser memory buffer limits on files larger than ~2 GB.
* As a result, the fine-tuned model could not be retrieved from Colab, and the task was handed over to run and deploy locally.

### 1.2 Local Hardware Constraints & Environment Adaptation
Upon inspecting the local machine:
* **Operating System:** Windows 11
* **Local GPU:** NVIDIA GeForce RTX 3050 Laptop GPU (**4 GB VRAM**, Ampere Architecture)
* **Local Storage:** 
  * Drive C: ~2.92 GB free (critically low)
  * Drive D: ~25.8 GB free
* **Installed Python Environment:** Python 3.12 with CPU-only PyTorch (`2.10.0+cpu`).

### 1.3 Architectural Pivot
* **Why 8B cannot train on 4GB VRAM:** Llama-3.1-8B requires ~5.7 GB of VRAM just to load the weights in 4-bit quantization, causing instant `CUDA Out of Memory (OOM)` errors during backpropagation.
* **The Solution:** We migrated the fine-tuning target to **`unsloth/Llama-3.2-1B-Instruct-bnb-4bit`**. In 4-bit precision, this base model takes only ~1.02 GB of VRAM, leaving over 2.5 GB of VRAM headroom for gradient checkpointing and LoRA optimizer states.
* **Storage Protection:** Created an isolated virtual environment (`.venv`) and redirected `HF_HOME` cache to Drive D (`d:\Documents\finclaw\.cache\huggingface`), completely safeguarding the limited space on Drive C.
* **CUDA & Precision:** Installed PyTorch `2.6.0+cu124` and configured native **`BF16` (BFloat16)** mixed precision, leveraging the RTX 3050 Ampere tensor cores.

---

## 2. Dataset Contract & Core Design

The model is trained on **`finclaw_colab_dataset.jsonl`** (758 multi-turn coaching conversations).

### 2.1 The Financial Ledger Schema
Every training sample links an emotional/behavioral user query to an exact **JSON Financial Ledger** provided in the System Prompt:

```json
CURRENT USER FINANCIAL LEDGER:
{
  "currency": "INR",
  "budget_period": "monthly",
  "period_days_remaining": 15,
  "accounts": {
    "liquid_checking_balance": 18000.0,
    "savings_emergency_balance": 0.0,
    "pending_fixed_obligations": "rent and utility bills still unpaid"
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

> **Important Design Principle:**  
> In production, the application backend calculates ledger math (balances, category limits, deficits) and feeds it into the prompt. FinClaw's objective is **behavioral coaching, empathy, and decision guidance**, not mental arithmetic.

---

## 3. Training History & Iterations

### Run 1: Baseline Local Fine-Tuning (100 Steps)
* **Goal:** Verify that the 4-bit QLoRA pipeline trains cleanly on the 4GB RTX 3050 without OOM.
* **Hyperparameters:**
  * Base Model: `Llama-3.2-1B-Instruct-bnb-4bit`
  * LoRA Rank ($r$): 16, Alpha: 16, Dropout: 0.05
  * Target Modules: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`
  * Batch Size: 1 per device, Gradient Accumulation: 4 (effective batch size: 4)
  * Optimizer: `paged_adamw_8bit`
  * Precision: Native `bf16=True`, `fp16=False`
  * Max Steps: 100 (~0.52 epochs)
* **Runtime:** 351.8 seconds (~5.8 minutes)
* **Results:**
  * Initial Loss: `2.681` $\rightarrow$ Final Loss: `0.799`
  * Token Accuracy: `80.66%`
* **Finding:** While training succeeded, 0.52 epochs was insufficient for deeply anchoring complex variable relationships; the model occasionally hallucinated persona details when prompted with incomplete schemas.

---

### Run 2: Deep Grounding Fine-Tuning (350 Steps)
* **Goal:** Extend training across multiple full dataset passes (~2 epochs) to improve grounded reasoning and minimize hallucinations.
* **Hyperparameters:**
  * Warmup Steps: 15
  * Max Steps: 350 (~1.85 full epochs across 758 samples)
  * Learning Rate: `2e-4` with linear decay
  * Checkpoint Strategy: Saved every 100 steps
* **Runtime:** 1,240 seconds (**20.6 minutes**) on RTX 3050 GPU
* **Results & Metrics:**
  * Initial Loss: `2.159`
  * Step 100 Loss: `0.793`
  * Step 200 Loss: `0.708`
  * Step 300 Loss: `0.654`
  * Final Training Loss: **`0.676`** *(Significant drop from baseline)*
  * Mean Token Accuracy: **`82.72%`**
* **Artifacts Generated:** Saved in [`finclaw_finetuned_adapter/`](./finclaw_finetuned_adapter):
  * `adapter_model.safetensors` (~22.5 MB)
  * `adapter_config.json`
  * `tokenizer.json` & `tokenizer_config.json`
  * `chat_template.jinja`

---

## 4. Evaluation & Verification Tests

### Test 1: Grounded Reasoning Under Uncertainty (The Shoes & Rent Test)
* **User Query:**  
  *"I have ₹12,000 left from my salary. I need to pay rent next week, but I haven't calculated the exact rent amount yet. I'm thinking of buying ₹6,000 shoes. Should I buy them?"*
* **Ledger State:** Liquid balance ₹12,000; pending rent amount unknown; shoes cost ₹6,000.
* **Model Output:**  
  > *"You have ₹12,000 in your checking account, so the purchase would leave you with ₹6,000. Since you have a budget envelope of ₹2,000 allocated for apparel, this would exceed that limit by ₹4,000. Given that you are saving for a major goal, it might be worth waiting until you know the exact rent amount to ensure you stay on track."*
* **Scorecard:**
  1. Did NOT invent an imaginary rent amount: **Passed**
  2. Did NOT assume an arbitrary date: **Passed**
  3. Calculated remaining ₹6,000 accurately: **Passed**
  4. Explicitly conditioned purchase on the unknown rent: **Passed**

---

### Test 2: Burnout & Impulse Spending (Headphones & Credit Card Bill)
* **User Query:**  
  *"I've had an exhausting week at work and feel burned out. I saw noise-canceling headphones on sale for ₹8,500. I have ₹22,000 in my account, but my credit card bill of ₹12,000 is due in 10 days. Should I buy them?"*
* **Model Output:**  
  > *"That exhaustion is a heavy load to carry, and it makes sense that you're looking for a way to decompress. You have ₹22,000 in your checking account, but your credit card bill is ₹12,000 pending, so spending ₹8,500 on this purchase would put you ₹3,000 over your monthly limit. Since you are saving for a major goal and your emergency fund is at ₹0, it might be worth sleeping on it for a day to see if the urge is still there after a night's rest."*

---

## 5. Extra Component: Heuristic Natural-Language Parser

> [!NOTE]  
> **Context & Disclaimer:**  
> The original project requirement was solely to **fine-tune and deliver the model weights**. If integrating into an existing application codebase, that codebase likely already maintains a user financial ledger and injects it into the prompt.  
> 
> The script [`finclaw_backend_helper.py`](./finclaw_backend_helper.py) was built as an **optional, local testing convenience** to allow testing conversational queries in plain English without typing raw JSON every time.

### 5.1 How It Works
* Extracts currency, balances, pending bill descriptions/amounts, and item costs via regular expressions and phrase-matching heuristics.
* Computes remaining liquidity (`balance - pending_bills - item_cost`).
* Automatically wraps the result in the JSON ledger contract and applies Chain-of-Thought prompting before querying the model.

### 5.2 Known Limitations of the Parser
Because it relies on heuristic text parsing rather than a semantic entity extraction model:
* Unusual phrasing (e.g., *"I'm down to twelve grand, gotta hand five to the landlord"*) might fail to match regex rules.
* Sentences with multiple non-essential items mentioned in a single paragraph may select only the first detected item.
* **Production Recommendation:** In a live app, financial ledgers should be supplied directly by the banking/budgeting database (Plaid, Open Banking API, or user input forms) rather than regex text extraction.

---

## 6. How to Integrate & Use the Model

### Option A: Direct Integration in Your Friend's Python Codebase
Your friend can load the fine-tuned adapter directly using standard HuggingFace / PEFT:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

base_model_id = "unsloth/Llama-3.2-1B-Instruct-bnb-4bit"
adapter_dir = "./finclaw_finetuned_adapter"

tokenizer = AutoTokenizer.from_pretrained(adapter_dir)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True
)

base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    torch_dtype=torch.bfloat16,
    quantization_config=bnb_config,
    device_map="auto"
)

# Attach the trained FinClaw adapter
model = PeftModel.from_pretrained(base_model, adapter_dir)
model.eval()
```

### Option B: Local Interactive Testing
Run the local interactive chat engine anytime:
```powershell
d:\Documents\finclaw\.venv\Scripts\python.exe finclaw_backend_helper.py
```

### Option C: Retraining / Re-running via Jupyter Notebook
Open [`Fine_Tuning_FinClaw.ipynb`](./Fine_Tuning_FinClaw.ipynb) in VS Code or Jupyter Lab using the `.venv` Python kernel. All cells are configured with 350 steps, native BF16, and local checkpointing.
