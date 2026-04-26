import time
import sys
import os

# Add the project root to the python path so it finds 'core'
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.Agent import app  # Import your compiled graph
from core.State import AgentState

# --- THE GAUNTLET: 10 Queries to Break the System ---
TEST_CASES = [
    # 1. Multi-Entity Split (Tests Planner's ability to separate companies)
    "What is the governing law for Google and AWS?",
    
    # 2. Numeric Extraction (Tests Executor's precision)
    "How many days of notice are required for termination?",
    
    # 3. Abstract Liability (Tests Generator's legal reasoning)
    "If a third party sues me because of content I hosted, will you defend me?",
    
    # 4. "Impossible" Question (Tests Validator's ability to handle missing data)
    "What is the personal home address of the Google CEO?",
    
    # 5. Synonym Trap (Tests Planner's synonym expansion)
    "Can I just stop paying if the service goes down?",
    
    # 6. Specific Clause Hunt (Tests Executor's depth k=50)
    "Does the contract include a waiver of jury trial?",
    
    # 7. Financial Impact (Tests Pydantic Schema parsing)
    "What are the interest charges for late payments?",
    
    # 8. Intellectual Property (Tests Concept Matching)
    "Who owns the rights to the code I deploy on your servers?",
    
    # 9. Comparison/Conflict (Tests Handling conflicting info)
    "Is arbitration mandatory or optional?",
    
    # 10. Broad Policy (Tests Summarization)
    "Summarize the acceptable use policy regarding spam."
]

def run_stress_test():
    print(f"🚀 STARTING STRESS TEST ({len(TEST_CASES)} scenarios)...")
    print("="*60)
    
    results = []

    for i, query in enumerate(TEST_CASES):
        print(f"\n🧪 TEST {i+1}/{len(TEST_CASES)}: '{query}'")
        start_time = time.time()
        
        # Reset State
        initial_state = {
            "query": query,
            "retry_count": 0,
            "feedback": None
        }
        
        try:
            # Run the Agent
            output = app.invoke(initial_state)
            final = output.get("final_answer", {})
            
            elapsed = time.time() - start_time
            
            # Print Result Summary
            print(f"   ⏱️ Time: {elapsed:.2f}s")
            print(f"   🎯 Answer: {final.get('answer', 'Error')[:100]}...") # First 100 chars
            print(f"   ⚖️ Risk Level: {final.get('risk_level', 'Unknown')}")
            
            # Save for report
            results.append({
                "query": query,
                "answer": final.get("answer"),
                "risk": final.get("risk_level"),
                "time": elapsed
            })
            
        except Exception as e:
            print(f"   ❌ CRASHED: {e}")
            results.append({"query": query, "error": str(e)})

    # --- FINAL REPORT ---
    print("\n" + "="*60)
    print("📊 STRESS TEST REPORT")
    print("="*60)
    
    for res in results:
        status = "❌" if "error" in res else "✅"
        print(f"{status} [{res.get('risk', 'ERR')}] {res['query']}")

run_stress_test()