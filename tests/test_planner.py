import sys
import os

# Add the project root to the python path so it finds 'core'
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.nodes.Planner import planner
from core.State import AgentState

state = AgentState(
    query="Can OpenAI delete my account just because I haven't used it in a while?",
    feedback=None,
    plan=[],
    documents=[],
    final_answer={},
    validation_status="",
    retry_count=0
)

print("🧪 TESTING PLANNER NODE...")
result = planner(state)

print("\n--- PLAN OUTPUT ---")
for i, step in enumerate(result['plan']):
    print(f"Step {i+1}: {step}")
