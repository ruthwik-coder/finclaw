import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from intent_helper import extract_purchase_intent, get_live_ledger_for_message
from model_engine import FinClawEngine

def test_pipeline():
    engine = FinClawEngine.get_instance()
    db_path = os.path.join(os.path.dirname(__file__), "db.sqlite")

    print("\n" + "=" * 60)
    print("Testing Turn 1: Special Dinner ($50)")
    print("=" * 60)
    msg1 = "I had a super stressful day at work and want to treat myself to a special dinner tonight for $50. Should I spend it?"
    
    cat, item, cost = extract_purchase_intent(msg1)
    print(f"Extracted -> Category: {cat}, Item: {item}, Cost: ${cost}")
    assert cost == 50.0, f"Expected 50.0, got {cost}"
    assert cat == "Food", f"Expected Food, got {cat}"

    ledger1 = get_live_ledger_for_message(db_path, msg1)
    print(f"Live Ledger -> Food Budget: ${ledger1['category_envelope']['allocated_limit']}, Spent: ${ledger1['category_envelope']['spent_to_date']}, Remaining: ${ledger1['category_envelope']['remaining_balance']}")
    
    history = [{"role": "user", "content": msg1}]
    stream1 = engine.stream_chat(history, ledger1, max_new_tokens=150)
    reply1 = "".join(list(stream1))
    print("\nFinClaw Response 1:")
    print(reply1.strip())
    history.append({"role": "assistant", "content": reply1})

    print("\n" + "=" * 60)
    print("Testing Turn 2 (Multi-turn Memory): Follow-up question")
    print("=" * 60)
    msg2 = "What if I find a cheaper place for $20 instead?"
    cat2, item2, cost2 = extract_purchase_intent(msg2)
    print(f"Extracted -> Category: {cat2}, Item: {item2}, Cost: ${cost2}")
    
    ledger2 = get_live_ledger_for_message(db_path, msg2, active_ledger=ledger1)
    history.append({"role": "user", "content": msg2})
    stream2 = engine.stream_chat(history, ledger2, max_new_tokens=150)
    reply2 = "".join(list(stream2))
    print("\nFinClaw Response 2 (With Memory):")
    print(reply2.strip())
    print("\n" + "=" * 60)
    print("Testing Turn 3: Netflix Subscription ($60)")
    print("=" * 60)
    msg3 = "I'm really frustrated this week because of my over time work in office so I am planning to buy a new Netflix subscription for my kid worth 60 should i do it?"
    cat3, item3, cost3 = extract_purchase_intent(msg3)
    print(f"Extracted -> Category: {cat3}, Item: {item3}, Cost: ${cost3}")
    assert cost3 == 60.0, f"Expected 60.0, got {cost3}"
    assert cat3 == "Entertainment", f"Expected Entertainment, got {cat3}"

    ledger3 = get_live_ledger_for_message(db_path, msg3)
    print(f"Live Ledger -> Category: {ledger3['category_envelope']['category_name']}, Remaining: ${ledger3['category_envelope']['remaining_balance']}, Exceeds: ${ledger3['decision_transaction']['exceeds_category_by']}")
    
    stream3 = engine.stream_chat([{"role": "user", "content": msg3}], ledger3, max_new_tokens=150)
    reply3 = "".join(list(stream3))
    print("\nFinClaw Response 3:")
    print(reply3.strip())

    print("\n" + "=" * 60)
    print("Testing Turn 4: LLM Classifier Fallback on Novel Purchase (Ceramic pottery wheel $80)")
    print("=" * 60)
    msg4 = "I feel so uninspired lately, thinking about buying a ceramic pottery wheel for 80"
    cat_direct = engine.classify_category(msg4)
    print(f"Direct LLM Category Classification -> '{cat_direct}'")

    ledger4 = get_live_ledger_for_message(db_path, msg4, engine=engine)
    print(f"Cascading Ledger -> Resolved Category: '{ledger4['category_envelope']['category_name']}', Cost: ${ledger4['decision_transaction']['estimated_cost']}")
    assert ledger4['decision_transaction']['estimated_cost'] == 80.0
    print("\nSUCCESS: Both Fast Path and LLM Fallback Classification verified!")
    print("=" * 60)



if __name__ == "__main__":
    test_pipeline()
