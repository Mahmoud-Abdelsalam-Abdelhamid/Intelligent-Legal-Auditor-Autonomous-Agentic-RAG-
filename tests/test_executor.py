import sys
import os
import time

# --- 1. SETUP PATHS ---
# Ensures Python can find your 'core' folder
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.nodes.Executer import executer
from core.State import AgentState

# --- 2. YOUR EXACT MOCK PLAN ---
mock_plan = [
    "OpenAI limitation of liability",
    "OpenAI aggregate liability cap",
    "OpenAI damages limit"
]

# --- 3. CONSTRUCT STATE ---
# We manually create the state as if the Planner just finished
state = AgentState(
    query="Test verification of OpenAI terms",
    plan=mock_plan,
    feedback=None,
    documents=[],
    final_answer={},
    validation_status="start",
    retry_count=0
)

# --- 4. RUN EXECUTER ---
print(f"🚀 Running Executer with {len(mock_plan)} search steps...")
start_time = time.time()

try:
    result = executer(state)
    duration = time.time() - start_time
    print(f"✅ Success! Executed in {duration:.2f} seconds.\n")

    # --- 5. INSPECT RESULTS ---
    docs = result.get('documents', [])
    print(f"📂 Total Chunks Retrieved: {len(docs)}")
    print("="*60)

    for i, doc in enumerate(docs):
        # Print Metadata (Source) and a content preview
        source = doc.metadata.get('source', 'Unknown File')
        content = doc.page_content[:200].replace("\n", " ") # Clean up newlines for display
        
        print(f"📄 [Chunk {i+1}] Source: {source}")
        print(f"   Preview: \"{content}...\"")
        print("-" * 60)

except Exception as e:
    print(f"❌ Error: {e}")