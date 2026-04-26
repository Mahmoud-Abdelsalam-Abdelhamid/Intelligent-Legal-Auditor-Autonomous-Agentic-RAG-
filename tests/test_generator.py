import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.nodes.Generator import generator
from core.State import AgentState
from langchain_core.documents import Document

mock_docs = [
    Document(page_content="[[SOURCE: Google]] You are always free to stop using our services at any time.", metadata={"company": "Google"}),
    Document(page_content="[[SOURCE: AWS]] AWS will charge you for the full month during which the cancellation takes place.", metadata={"company": "AWS"})
]

state = AgentState(
    query="What is the termination fee for Google and AWS?",
    documents=mock_docs,
    plan=[],
    feedback=None,
    final_answer={},
    validation_status="",
    retry_count=0
)

print("🧪 TESTING GENERATOR NODE...")
result = generator(state)

print("\n--- LLM OUTPUT ---")
print(result['final_answer'])
