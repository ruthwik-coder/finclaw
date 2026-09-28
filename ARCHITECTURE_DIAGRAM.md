# 🏛️ FinClaw End-to-End System Architecture

This document contains the complete system architecture diagram for **FinClaw** (Emotionally Intelligent LLM Agent for Context-Aware Financial Decision Support). 

You can view the rendered diagram directly on GitHub, copy the Mermaid code into [Mermaid Live Editor](https://mermaid.live) to export a high-resolution PNG/SVG for your PowerPoint slides, or paste it directly into Gamma / Notion.

---

## 📊 End-to-End Workflow Diagram (Mermaid)

```mermaid
flowchart TD
    %% Styling Definitions
    classDef client fill:#1e293b,stroke:#38bdf8,stroke-width:2px,color:#f8fafc;
    classDef intent fill:#312e81,stroke:#818cf8,stroke-width:2px,color:#f8fafc;
    classDef database fill:#064e3b,stroke:#34d399,stroke-width:2px,color:#f8fafc;
    classDef contract fill:#78350f,stroke:#fbbf24,stroke-width:2px,color:#f8fafc;
    classDef ai fill:#581c87,stroke:#c084fc,stroke-width:2px,color:#f8fafc;
    classDef output fill:#0f172a,stroke:#38bdf8,stroke-width:2px,color:#f8fafc;

    %% 1. User Touchpoint
    subgraph S1 ["1. Client Interaction Layer"]
        U["👤 User"] -->|"Types query: 'Stressed from work, want to buy a Netflix subscription for $60'"| UI["📱 Client Interface<br/>(Streamlit Web App / Telegram Bot)"]
    end
    class UI client;

    %% 2. Intent & Entity Extraction
    subgraph S2 ["2. Cascading Hybrid Intent Engine"]
        UI -->|"Raw text"| CE["Cascading Intent Extractor<br/>(intent_helper.py)"]
        CE -->|"Step 1: Fast Lexicon (&lt;0.05ms)<br/>150+ brands & categories"| LK{"Recognized in<br/>Taxonomy?"}
        LK -->|"Yes"| CR["Category: Entertainment<br/>Cost: $60.00"]
        LK -->|"No (Fallback)"| LLMC["Step 2: 1-Step LLM Classifier (~150ms)<br/>Maps strictly to 8 SQLite categories"]
        LLMC --> CR
    end
    class CE,LK,CR,LLMC intent;

    %% 3. Deterministic Database Engine
    subgraph S3 ["3. Deterministic Financial Ledger (Zero Math Hallucination)"]
        CR -->|"SQL Query: Category & Cost"| DB[("🗄️ Actual Budget SQLite<br/>(db.sqlite)")]
        DB -->|"Calculates deterministic math"| CALC["Deterministic Calculations:<br/>• Checking Balance: $1,080.00<br/>• Entertainment Limit: $100.00<br/>• Spent to Date: $45.00<br/>• Remaining Balance: $55.00<br/>• Envelope Deficit: -$5.00 OVER"]
    end
    class DB,CALC database;

    %% 4. Contract & Context Injection
    subgraph S4 ["4. Context Injection & Memory Protocol"]
        CALC -->|"JSON Ledger State"| PROMPT["Structured System Prompt Contract<br/>CURRENT USER FINANCIAL LEDGER:<br/>{ accounts, envelopes, transaction, deficit }"]
        MEM["🧠 Sliding Window Memory<br/>(Preserves last 6 messages / 3 turns)"] -->|"Multi-turn context"| PROMPT
    end
    class PROMPT,MEM contract;

    %% 5. AI Reasoning & Fine-Tuned Model
    subgraph S5 ["5. Empathetic LLM Reasoning Engine"]
        PROMPT -->|"Injected Context"| LLM["🤖 FinClaw Fine-Tuned Model<br/>(Llama-3.2-1B-Instruct + 4-bit QLoRA)"]
        LLM -->|"LoRA Adapter Weights"| ADAPT["🧠 Fine-Tuned Adapter (22.5 MB)<br/>• Trained on 758 emotional dialogues<br/>• Loss: 0.676 | Accuracy: 82.7%<br/>• Native BF16 on 4GB Laptop GPU"]
    end
    class LLM,ADAPT ai;

    %% 6. Real-Time Streaming Output
    subgraph S6 ["6. Sub-Second Real-Time Response"]
        LLM -->|"TextIteratorStreamer (TTFT &lt; 1.0s)"| OUT["💬 Empathetic Financial Response:<br/>1. Validates workplace stress/fatigue without judgment<br/>2. Explicitly cites $55 left & $5.00 over-budget deficit<br/>3. Recommends 24-hr cooling-off behavioral nudge"]
        OUT --> UI
    end
    class OUT output;
```

---

## 🧩 Architectural Component Breakdown (For Speaking Notes)

When presenting this architecture slide to mentors, explain the **6 distinct layers**:

1. **Client Interaction Layer (UI):**
   * Accepts natural, emotional user language across Streamlit or Telegram.
   * Users don't need to enter financial tables manually; they talk like they would to a friend.

2. **Cascading Hybrid Intent Engine:**
   * **Fast Path (< 0.05 ms):** Uses an in-memory dictionary of 150+ brands (Netflix, Starbucks, Nike, Utilities) and robust regex patterns (`worth 60`, `costs 50`) to extract entities instantly with **zero latency**.
   * **LLM Classifier Fallback (~150 ms):** If an unmapped or unusual purchase is detected (e.g. *"ceramic pottery wheel"*), the local model classifies it strictly into one of the 8 SQLite envelope categories (`Entertainment`, `Food`, `Personal Care`, `Bills`, `Bills (Flexible)`, `Rent/housing`, `Savings`, `General`).

3. **Deterministic Financial Ledger (Actual Budget SQLite `db.sqlite`):**
   * **Why it matters:** Neural networks are probabilistic and unreliable at mental arithmetic. 
   * **Separation of Concerns:** 100% of arithmetic calculations (subtractions, envelope ceilings, deficits) are executed deterministically by SQL queries, ensuring **0% math hallucinations**.

4. **Context Injection & Sliding Memory Protocol:**
   * Formats the ground-truth ledger into the strict `CURRENT USER FINANCIAL LEDGER` JSON contract required by the fine-tuned model.
   * Integrates a sliding conversation window (retaining the last 6 messages / 3 turns) so the AI remembers prior turns without context overflow.

5. **Empathetic LLM Reasoning Engine (FinClaw):**
   * Built on Meta's `Llama-3.2-1B-Instruct` fine-tuned with **4-bit QLoRA** and native **BF16**.
   * Consumes only **1.07 GB VRAM**, enabling smooth execution on consumer laptop GPUs (NVIDIA RTX 3050 4GB).
   * Fine-tuned on 758 multi-turn dialogues to combine psychological validation with strict budget discipline.

6. **Sub-Second Streaming Response:**
   * Uses Hugging Face's `TextIteratorStreamer` to deliver **Time-To-First-Token in ~0.85 seconds**.
   * Delivers a three-part response: (1) Emotional validation $\rightarrow$ (2) Arithmetic ground truth $\rightarrow$ (3) Cooling-off / alternative behavioral nudge.
