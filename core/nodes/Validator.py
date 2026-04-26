from langsmith import traceable
from core.State import AgentState

@traceable(name="Legal Validator")
def validator(state: AgentState):
    print("\n⚖️ VALIDATOR: Inspecting Final Answer...")
    
    final_answer = state["final_answer"]
    retry_count = state["retry_count"]
    
    # 1. Check for Generator Errors
    if "error" in final_answer:
        print("  ❌ Validation Failed: Generator Error.")
        return {
            "validation_status": "retry",
            "feedback": f"Generator failed with error: {final_answer['error']}",
            "retry_count": retry_count + 1
        }
    
    risk_level = final_answer.get("risk_level", "Unknown")
    answer_text = final_answer.get("answer", "").lower()

    # 2. NEW: Accept "Valid Negatives" (The fix for Google)
    # If the answer explicitly says there is NO fee, that is a valid answer!
    valid_negatives = ["no termination fee", "does not charge", "free to stop", "pay only for usage"]
    
    if any(phrase in answer_text for phrase in valid_negatives):
        print("  ✅ Validation Passed (Confirmed Negative Result).")
        return {
            "validation_status": "valid",
            "feedback": None,
            "retry_count": retry_count 
        }

    # 3. Fail if it genuinely gave up
    if risk_level == "Unknown" and "information not found" in answer_text:
        print("  ❌ Validation Failed: Information missing.")
        return {
            "validation_status": "retry",
            "feedback": "The previous search missed the relevant clauses. Try searching for synonyms (e.g., 'Cancellation' instead of 'Termination').",
            "retry_count": retry_count + 1
        }
    
    print("  ✅ Validation Passed.")
    return {
        "validation_status": "valid",
        "feedback": None, 
        "retry_count": retry_count 
    }