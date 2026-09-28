"""
FinClaw AI Financial Coach — Streamlit Dashboard
================================================
- Fast GPU token streaming (sub-second perceived latency).
- Live SQLite budget integration (Actual Budget db.sqlite).
- Multi-turn conversational memory.
- Built-in Model Evaluation & Benchmark Metrics Tab.
"""

import os
import sys
import json
import streamlit as st

# Ensure current directory is on python path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from fetch_actual_budget import fetch_budget_summary
from intent_helper import get_live_ledger_for_message
from model_engine import FinClawEngine

# -------------------------------------------------------------
# Streamlit Page Config & Styling
# -------------------------------------------------------------
st.set_page_config(
    page_title="FinClaw — AI Financial Coach & Evaluation",
    page_icon="🐾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.1rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
        color: #f0f2f6;
    }
    .sub-header {
        font-size: 0.95rem;
        color: #8b949e;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #161b22;
        padding: 16px;
        border-radius: 8px;
        border: 1px solid #30363d;
        text-align: center;
    }
    .metric-num {
        font-size: 1.8rem;
        font-weight: 700;
        color: #58a6ff;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #8b949e;
        text-transform: uppercase;
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------
# Cached Model Engine Loader
# -------------------------------------------------------------
@st.cache_resource(show_spinner="Initializing FinClaw GPU Engine...")
def load_engine():
    return FinClawEngine.get_instance()

engine = load_engine()
DB_PATH = os.path.join(CURRENT_DIR, "db.sqlite")
BENCHMARK_PATH = os.path.join(CURRENT_DIR, "benchmark_results.json")

# -------------------------------------------------------------
# Session State Initialization (Memory)
# -------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "active_ledger" not in st.session_state:
    st.session_state.active_ledger = None

# -------------------------------------------------------------
# Sidebar: Live SQLite Budget & Account Inspector
# -------------------------------------------------------------
with st.sidebar:
    st.title("🐾 FinClaw Ledger")
    st.caption("Live SQLite Database (`db.sqlite`)")

    # Fetch live summary from SQLite
    try:
        budget_items = fetch_budget_summary(DB_PATH, month_str="202609")
    except Exception as e:
        st.error(f"Error reading db.sqlite: {e}")
        budget_items = []

    # Accounts Overview
    st.subheader("💳 Accounts")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Checking", "$1,080.00")
    with col2:
        st.metric("Emergency", "$5,000.00")

    st.divider()

    # Active Envelopes
    st.subheader("📊 Category Envelopes")
    for item in budget_items[:6]:
        cat = item["category"]
        budgeted = item["budgeted"]
        spent = item["spent"]
        bal = item["balance"]
        
        if budgeted > 0:
            pct = min(1.0, max(0.0, spent / budgeted))
            st.write(f"**{cat}**: ${spent:,.0f} / ${budgeted:,.0f} *(Bal: ${bal:,.0f})*")
            st.progress(pct)

    st.divider()

    # Conversational Memory & Controls
    st.subheader("⚙️ Memory & Settings")
    memory_enabled = st.toggle("Multi-turn Memory", value=True, help="When enabled, FinClaw remembers previous turns in the conversation.")
    
    if st.button("🗑️ Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.active_ledger = None
        st.rerun()

    # Show Active JSON Ledger
    if st.session_state.active_ledger:
        with st.expander("🔍 Active Grounded Ledger (JSON)"):
            st.json(st.session_state.active_ledger)

# -------------------------------------------------------------
# Main Application Tabs (Chat & Evaluation Metrics)
# -------------------------------------------------------------
st.markdown('<div class="main-header">🐾 FinClaw AI Financial Coach</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Emotionally-aware guidance grounded in live SQLite budget data. Fast GPU token streaming enabled.</div>', unsafe_allow_html=True)

tab_chat, tab_metrics = st.tabs(["💬 Chat with FinClaw", "📊 Evaluation & Model Metrics"])

# =============================================================
# TAB 1: Chat Assistant (ChatGPT Interface)
# =============================================================
with tab_chat:
    # Welcome placeholder if conversation is empty
    if len(st.session_state.messages) == 0:
        st.info("""
        👋 **Welcome to FinClaw!**  
        Talk naturally about financial urges, upcoming purchases, or stress. FinClaw automatically checks your live SQLite budget.
        
        **Try asking:**
        * *"I've had a super stressed day at work and I'm really craving a special dinner tonight for $50, but I don't know if I should spend it."*
        * *(Follow-up)* *"What if I find a cheaper place for $20 instead?"*
        """)

    # Display chat history from memory
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # User Chat Input
    if prompt := st.chat_input("Ask FinClaw about a purchase, stress, or your budget..."):
        # 1. Display user prompt
        with st.chat_message("user"):
            st.markdown(prompt)

        # 2. Append to memory
        st.session_state.messages.append({"role": "user", "content": prompt})

        # 3. Resolve live financial ledger from SQLite
        ledger = get_live_ledger_for_message(
            db_path=DB_PATH,
            user_text=prompt,
            active_ledger=st.session_state.active_ledger if memory_enabled else None
        )
        st.session_state.active_ledger = ledger

        # 4. Generate streaming response from FinClaw GPU engine
        with st.chat_message("assistant"):
            history_to_send = st.session_state.messages if memory_enabled else [{"role": "user", "content": prompt}]
            
            response_stream = engine.stream_chat(
                messages_history=history_to_send,
                ledger=ledger,
                max_new_tokens=250,
                temperature=0.3
            )
            full_response = st.write_stream(response_stream)

        # 5. Append assistant reply to memory
        st.session_state.messages.append({"role": "assistant", "content": full_response})

# =============================================================
# TAB 2: Evaluation & Quantitative Statistics
# =============================================================
with tab_metrics:
    st.subheader("📈 Automated Benchmark Performance & Scorecard")
    st.caption("Quantitative evaluation across adversarial tests, arithmetic grounding, and GPU performance.")

    if os.path.exists(BENCHMARK_PATH):
        try:
            with open(BENCHMARK_PATH, "r", encoding="utf-8") as f:
                bdata = json.load(f)
            bm = bdata["metrics"]
            btests = bdata["test_results"]

            # Key KPI Cards
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-num">{bm['overall_accuracy_pct']}%</div>
                    <div class="metric-label">Overall Accuracy</div>
                </div>
                """, unsafe_allow_html=True)
            with c2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-num">{bm['arithmetic_grounding_pct']}%</div>
                    <div class="metric-label">Arithmetic Grounding</div>
                </div>
                """, unsafe_allow_html=True)
            with c3:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-num">{bm['hallucination_resistance_pct']}%</div>
                    <div class="metric-label">Hallucination Resistance</div>
                </div>
                """, unsafe_allow_html=True)
            with c4:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-num">{bm['avg_tokens_per_second']}</div>
                    <div class="metric-label">Tokens / Sec (GPU)</div>
                </div>
                """, unsafe_allow_html=True)

            st.write("")
            st.divider()

            # Hardware & Latency Overview
            col_hw1, col_hw2, col_hw3 = st.columns(3)
            with col_hw1:
                st.info(f"⚡ **Avg Response Latency:** `{bm['avg_response_time_seconds']}s`")
            with col_hw2:
                st.success(f"🎮 **Peak VRAM Utilized:** `{bm['peak_vram_gb']} GB` / 4.00 GB")
            with col_hw3:
                st.info(f"🧪 **Total Scenarios Evaluated:** `{bm['total_tests']}`")

            # Detailed Scenario Test Table
            st.write("#### 🧪 Benchmark Scenario Test Log")
            
            table_rows = []
            for t in btests:
                table_rows.append({
                    "Test ID": t["test_id"],
                    "Category": t["scenario"],
                    "Arithmetic": "✅ Pass" if t["arithmetic_pass"] else "❌ Fail",
                    "Hallucination Test": "✅ Pass" if t["hallucination_pass"] else "❌ Fail",
                    "Status": "🟢 Pass" if t["overall_status"] == "PASS" else "🟡 Notice",
                    "Speed": f"{t['tokens_per_second']} tps",
                    "Tokens": t["tokens"]
                })
            st.dataframe(table_rows, use_container_width=True)

            # Expandable Raw Test Inspect
            with st.expander("🔍 View Detailed Test Prompts & Model Outputs"):
                for t in btests:
                    st.write(f"**[{t['test_id']}] {t['scenario']}**")
                    st.write(f"*Query:* {t['query']}")
                    st.code(t['response'], language="markdown")
                    st.write("---")

        except Exception as e:
            st.error(f"Error loading benchmark statistics: {e}")
    else:
        st.warning("No benchmark results found. Run `python evaluate_model.py` to generate metrics.")
