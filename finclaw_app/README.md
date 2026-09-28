# 🐾 FinClaw Streamlit Dashboard (`finclaw_app`)

Live ChatGPT-style conversational financial coach interface powered by:
1. **Live SQLite Database (`db.sqlite`):** Real-time account balances and category budget envelopes from Actual Budget.
2. **Fine-Tuned FinClaw Model:** Running on your local RTX 3050 GPU with native BF16 & 4-bit QLoRA.
3. **Sub-second Token Streaming:** Instant text generation via `TextIteratorStreamer`.
4. **Multi-turn Conversational Memory:** Remembers previous turns and follow-up adjustments in the conversation.

---

## 🚀 How to Run

From your terminal, run:

```powershell
d:\Documents\finclaw\.venv\Scripts\streamlit.exe run finclaw_app\app.py
```

The web dashboard will automatically open in your default browser at:  
👉 `http://localhost:8501`

---

## 📂 Folder Structure

| File | Purpose |
| :--- | :--- |
| [`app.py`](./app.py) | Main Streamlit web application with chat interface, sidebar budget indicators, and controls. |
| [`db.sqlite`](./db.sqlite) | Live SQLite database from Actual Budget holding accounts and envelopes. |
| [`fetch_actual_budget.py`](./fetch_actual_budget.py) | Database extraction module querying accounts, limits, and transactions. |
| [`intent_helper.py`](./intent_helper.py) | Auto-extracts purchase items, costs, and categories from conversational messages. |
| [`model_engine.py`](./model_engine.py) | Cached GPU model engine with multi-turn memory and real-time token streaming. |
| [`test_app_pipeline.py`](./test_app_pipeline.py) | CLI verification script testing the 2-turn memory and live budget flow. |
