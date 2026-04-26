import sys
import os

# Add the project root to the python path so it finds 'core'
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.nodes.Validator import validator
from core.State import AgentState

# Mock Input 1: A "Good" Negative Answer
mock_good_answer = {
    "risk_level": "Low",
    "answer": "Google does not charge a termination fee (you are free to stop). AWS charges for the full month.",
    "clause_reference": "free to stop using our services",
    "financial_impact": "None for Google",
    "reasoning": "Google has no fee."
}

# Mock Input 2: A "Bad" Unknown Answer
mock_bad_answer = {
    "risk_level": "Unknown",
    "answer": "Information not found in retrieved documents.",
    "clause_reference": "",
    "financial_impact": "N/A",
    "reasoning": "Data missing."
}

state = AgentState(
    query="Test Query",
    final_answer=mock_good_answer,
    retry_count=0,
    plan=[], documents=[], feedback=None, validation_status=""
)

print("🧪 TESTING VALIDATOR NODE...")
result = validator(state)

print("\n--- VALIDATION RESULT ---")
print(f"Status: {result['validation_status']}")
print(f"Feedback: {result['feedback']}")