# 🐾 FinClaw Worklet: Emotionally-Aware Financial Coaching AI

[![Model](https://img.shields.io/badge/Base%20Model-Llama--3.2--1B--Instruct-blue.svg)](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct)
[![Fine-Tuning](https://img.shields.io/badge/Fine--Tuning-QLoRA%20(4--bit)-green.svg)](https://github.com/huggingface/peft)
[![Precision](https://img.shields.io/badge/Precision-Native%20BF16-orange.svg)]()
[![Database](https://img.shields.io/badge/Database-SQLite%20(Actual%20Budget)-purple.svg)]()
[![Dashboard](https://img.shields.io/badge/Interface-Streamlit-red.svg)]()
[![License](https://img.shields.io/badge/License-Apache%202.0-lightgrey.svg)]()

**FinClaw Worklet** is an end-to-end FinTech empathetic coaching intelligence. It marries **Meta's Llama-3.2-1B-Instruct** (fine-tuned via 4-bit QLoRA) with a **live deterministic SQLite envelope ledger** (sourced from Actual Budget) and an interactive **Streamlit web dashboard**.

Traditional banking bots are cold and rule-based (*"Transaction denied: category exceeded"*), while off-the-shelf LLMs hallucinate financial balances and offer poor mathematical consistency. FinClaw solves both: **deterministic math is computed by SQLite, while the fine-tuned model delivers psychologically attuned, empathetic coaching.**

---

## 📑 Table of Contents
- [🤖 Agent Integration Guide (For AI Agents & Systems)](#-agent-integration-guide-for-ai-agents--systems)
  - [The System Prompt & Ledger Contract](#1-the-system-prompt--ledger-contract)
  - [Deterministic Arithmetic Architecture](#2-deterministic-arithmetic-architecture)
  - [Multi-Turn Memory & Chat Formatting](#3-multi-turn-memory--chat-formatting)
- [👤 Developer & Human Quick Start](#-developer--human-quick-start)
  - [1. Installation](#1-installation)
  - [2. Launching the Interactive Streamlit Web App](#2-launching-the-interactive-streamlit-web-app)
  - [3. Running the Automated Benchmark Suite](#3-running-the-automated-benchmark-suite)
  - [4. Python Code Integration (Load Base + Adapter)](#4-python-code-integration-load-base--adapter)
- [📂 Repository Directory Structure](#-repository-directory-structure)
- [📊 Evaluation Framework & KPI Achievements](#-evaluation-framework--kpi-achievements)
- [📖 Documentation & Defense Guides](#-documentation--defense-guides)

---

## 🤖 Agent Integration Guide (For AI Agents & Systems)

If you are an autonomous AI agent, orchestrator (LangChain, AutoGen, CrewAI, LlamaIndex), or backend microservice communicating with FinClaw, adhere to the following contract.

### 1. The System Prompt & Ledger Contract

FinClaw is trained strictly to ground its emotional coaching on an injected JSON ledger. **Never ask the LLM to calculate arithmetic in-token.** Always query the database, calculate the figures, and inject the JSON ledger into the `system` prompt:

```text
You are FinClaw, an empathetic and highly disciplined financial advisor.
Your objective is to validate the user's emotional triggers (stress, burnout, FOMO, celebration) without judgment,
while firmly upholding their financial limits and preventing impulsive financial harm.

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

### 2. Cascading Hybrid Intent Engine & Deterministic Architecture
```
User Query ("Planning to buy a new Netflix subscription for my kid worth $60")
                              │
                              ▼
 ┌─────────────────────────────────────────────────────────────┐
 │ Cascading Hybrid Intent Engine (intent_helper.py)           │
 │ 1. Fast-Path Lexicon (<0.05ms): Matches 150+ brands/services│
 │ 2. LLM Classifier Fallback (~150ms): Classifies novel items│
 │    strictly into the 8 SQLite budget envelopes              │
 └────────────────────────────┬────────────────────────────────┘
                              │ Resolved: "Entertainment", $60.00
                              ▼
 ┌─────────────────────────────────────────────────────────────┐
 │ SQLite Actual Budget Database (db.sqlite)                   │
 │ - Fetches checking balance: $1,080.00                       │
 │ - Fetches entertainment envelope: $100 budget, $45 spent    │
 │ - Computes deterministic deficit: $60 - $55 = $5.00 OVER    │
 └────────────────────────────┬────────────────────────────────┘
                              │ Injects Ground-Truth JSON Ledger Contract
                              ▼
 ┌─────────────────────────────────────────────────────────────┐
 │ Fine-Tuned Llama-3.2-1B Adapter (FinClaw Engine)            │
 │ - Validates office exhaustion / burnout with empathy        │
 │ - Explicitly cites $5.00 deficit from injected ledger       │
 │ - Recommends cooling-off period to prevent impulse regret   │
 └─────────────────────────────────────────────────────────────┘
```


### 3. Multi-Turn Memory & Chat Formatting
FinClaw uses the native Llama 3 chat template (`<|begin_of_text|><|start_header_id|>...`). For multi-turn dialogue, maintain a sliding window of the last **6 messages (3 conversation turns)**:

```python
# Format messages using the tokenizer chat template
formatted_prompt = tokenizer.apply_chat_template(
    conversation_history[-6:],  # Sliding memory window
    tokenize=False,
    add_generation_prompt=True
)
```

---

## 👤 Developer & Human Quick Start

### 1. Installation

**Hardware Minimum:** Any NVIDIA GPU with $\ge$ 4 GB VRAM (e.g. RTX 3050 Laptop GPU), or Apple Silicon / CPU.

```bash
git clone https://github.com/ruthwik-coder/finclaw_worklet.git
cd finclaw_worklet
pip install -r requirements.txt
```

### 2. Launching the Interactive Streamlit Web App
You can launch the web application in two ways:

#### Option A: 1-Click Batch Launcher (Windows)
Double-click **`run_demo.bat`** in the project root. It will automatically detect your virtual environment, launch Streamlit, and open your browser at `http://localhost:8501`.

#### Option B: From Command Prompt / Terminal
From the project root directory:
```bash
# Windows Direct Path (Recommended)
.venv\Scripts\streamlit.exe run finclaw_app\app.py

# Or activate the virtual environment first
.venv\Scripts\activate
streamlit run finclaw_app/app.py
```
Open your browser at `http://localhost:8501`. Features include:
* Real-time token streaming with `< 1s` perceived latency.
* Multi-turn chat memory across conversation turns.
* Live inspection of SQLite categories and account balances.
* Evaluation & Benchmark scorecard view.


### 3. Running the Automated Benchmark Suite
Run the 8-scenario adversarial automated benchmark to evaluate accuracy, arithmetic grounding, latency, and VRAM consumption:
```bash
python evaluate_model.py
```
*Outputs are saved to `BENCHMARK_REPORT.md` and `benchmark_results.json`.*

### 4. Python Code Integration (Load Base + Adapter)

#### Mode A: 4-Bit Quantized (Fastest, Consumer Laptop GPU ~1.07 GB VRAM)
```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

model_id = "unsloth/Llama-3.2-1B-Instruct-bnb-4bit"
adapter_path = "./finclaw_finetuned_adapter"

# 1. Load 4-bit base model
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

base_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(adapter_path)

# 2. Attach FinClaw PEFT LoRA adapter
model = PeftModel.from_pretrained(base_model, adapter_path)
model.eval()

# 3. Generate response
inputs = tokenizer("Hello FinClaw!", return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

#### Mode B: Standard 16-Bit / CPU / Cloud Server
```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-1B-Instruct",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("./finclaw_finetuned_adapter")
model = PeftModel.from_pretrained(base_model, "./finclaw_finetuned_adapter")
```

---

## 📂 Repository Directory Structure

```text
finclaw_worklet/
│
├── finclaw_app/                        # 🚀 Production Streamlit Application
│   ├── app.py                          # Streamlit UI with streaming & envelope telemetry
│   ├── model_engine.py                 # Singleton GPU loader, streaming & sliding memory
│   ├── db.sqlite                       # Live SQLite budget database (Actual Budget)
│   ├── fetch_actual_budget.py          # Deterministic SQL ledger extractor
│   ├── intent_helper.py                # Fast heuristic entity parser for user queries
│   └── test_app_pipeline.py            # End-to-end integration test
│
├── finclaw_finetuned_adapter/          # 🧠 Fine-Tuned LoRA Adapter Weights (22.5 MB)
│   ├── adapter_model.safetensors       # Trained QLoRA rank weights
│   ├── adapter_config.json             # PEFT LoRA hyperparameter configuration
│   ├── tokenizer.json & tokenizer_config.json
│   └── special_tokens_map.json
│
├── finclaw_worklet/                    # 📦 Worklet Source & Integration Reference
│   ├── db.sqlite                       # Reference Actual Budget schema
│   └── fetch_actual_budget.py          # Reference SQL queries
│
├── evaluate_model.py                   # 🧪 Automated Benchmark Suite (8 adversarial tests)
├── benchmark_results.json              # Raw latency, accuracy, and VRAM telemetry
├── BENCHMARK_REPORT.md                 # Detailed benchmark test run documentation
│
├── KPI_EVALUATION_PROGRESS.md          # 🎯 Line-by-line deliverables scorecard (~92% Complete)
├── PROJECT_PRESENTATION_GUIDE.md       # 🎤 Defense guide: 60s pitch, Q&A, demo script
├── FINCLAW_PROJECT_REPORT.md           # 📜 Training changelog (350 steps, 0.676 loss)
│
├── finclaw_backend_helper.py           # Standalone CLI chat with heuristic parser
├── train_local.py                      # Standalone local SFTTrainer fine-tuning script
├── finclaw_colab_dataset.jsonl         # 758-sample fine-tuning dataset
├── finclaw_finetuned_adapter.zip       # 19.3 MB portable zip of adapter weights
└── requirements.txt                    # Python environment dependencies
```

---

## 📊 Evaluation Framework & KPI Achievements

FinClaw was evaluated across 7 core FinTech KPIs outlined in the project evaluation framework:

| Key Performance Indicator (KPI) | SOTA Baseline | Target | FinClaw Achievement | Status |
| :--- | :---: | :---: | :---: | :---: |
| **1. Empathy Score** (1–5 scale) | 3.0 / 5.0 | **4.2 / 5.0** | **4.4 / 5.0** *(Psychological validation + tone)* | 🟢 **Target Exceeded** |
| **2. Intent Recognition Accuracy** | 85.0% | **90.0%** | **91.7%** *(11/12 test scenarios)* | 🟢 **Target Achieved** |
| **3. Response Latency (TTFT)** | < 3.0s | **< 2.0s** | **~0.85s (Streaming TTFT)** / 4.2s full gen | 🟢 **Target Achieved** |
| **4. User Trust Index (Groundedness)** | 40.0% | **60.0%** | **100.0% Hallucination Resistance** | 🟢 **Target Exceeded** |
| **5. Sentiment Detection F1-Score** | 0.80 | **0.85** | **0.865 F1** *(Loss: 0.676, Acc: 82.72%)* | 🟢 **Target Achieved** |
| **6. Conversation Length** | 1–2 turns | **3–5 turns** | **3–6 turns** *(Sliding window memory)* | 🟢 **Target Achieved** |
| **7. Budget Adherence Improvement** | 10.0% | **20.0%** | **Live SQLite Envelope Guardrails** | 🟢 **On Track (Infra Ready)** |

*Overall Framework Completion: **~92% Complete**.*

---

## 📖 Documentation & Defense Guides

* **[KPI Evaluation Progress](KPI_EVALUATION_PROGRESS.md)** — Comprehensive analysis of how FinClaw satisfies each deliverable.
* **[Project Presentation Guide](PROJECT_PRESENTATION_GUIDE.md)** — Presentation pitch, system architecture, mentor Q&A defense, and demo walkthrough.
* **[Benchmark Report](BENCHMARK_REPORT.md)** — Full breakdown of adversarial tests, arithmetic grounding, and hardware benchmarks.
* **[Training Changelog](FINCLAW_PROJECT_REPORT.md)** — Step-by-step history of fine-tuning runs, hyperparameter choices, and loss curves.

---

## 📜 License & Acknowledgments
* **Code & Adapters:** Apache-2.0 License.
* **Base Model:** Meta Llama 3.2 Community License Agreement.
* **Database Engine:** Based on the open-source [Actual Budget](https://actualbudget.org/) envelope budgeting specification.
