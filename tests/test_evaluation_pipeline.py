import time
from langchain_google_genai import ChatGoogleGenerativeAI
import pytest
import sys
import os

import json

# --- 1. Setup Path to import 'core' ---
# This ensures we can import from the sibling directory 'core'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from deepeval import assert_test
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.metrics import FaithfulnessMetric, GEval
from langchain_nvidia_ai_endpoints import ChatNVIDIA

# Import your graph. 
# NOTE: Ensure core/Agent.py exposes 'graph' or a 'run_agent' function.
from core.Agent import graph
from langchain_groq import ChatGroq
from deepeval.models.base_model import DeepEvalBaseLLM

class GroqDeepEvalModel(DeepEvalBaseLLM):
    def __init__(self, model_name="llama-3.3-70b-versatile"):
        self.model = ChatGroq(model=model_name, temperature=0)

    def load_model(self):
        return self.model

    def generate(self, prompt: str) -> str:
        # DeepEval expects a string response
        return self.model.invoke(prompt).content

    async def a_generate(self, prompt: str) -> str:
        # Async support is required by DeepEval
        response = await self.model.ainvoke(prompt)
        return response.content

    def get_model_name(self):
        return "Llama-3-70B (Groq)"
groq_judge = GroqDeepEvalModel()


# class NVIDIADeepEvalModel(DeepEvalBaseLLM):
#     def __init__(self, model_name="meta/llama-3.1-70b-instruct"):
#         self.model = ChatNVIDIA(model=model_name, temperature=0)

#     def load_model(self):
#         return self.model

#     def generate(self, prompt: str) -> str:
#         return self.model.invoke(prompt).content

#     async def a_generate(self, prompt: str) -> str:
#         response = await self.model.ainvoke(prompt)
#         return response.content

#     def get_model_name(self):
#         return "Llama-3.1-70B (NVIDIA)"

# # Initialize the new judge
# nvidia_judge = NVIDIADeepEvalModel()

# class GoogleDeepEvalModel(DeepEvalBaseLLM):
#     def __init__(self, model_name="gemini-2.5-flash"):
#         self.model_name = model_name
#         # Gemini handles structured outputs very well
#         self.model = ChatGoogleGenerativeAI(
#             model=model_name, 
#             temperature=0,
#             google_api_key=os.getenv("GOOGLE_API_KEY")
#         )

#     def load_model(self):
#         return self.model

#     def generate(self, prompt: str) -> str:
#         return self.model.invoke(prompt).content

#     async def a_generate(self, prompt: str) -> str:
#         res = await self.model.ainvoke(prompt)
#         return res.content

#     def get_model_name(self):
#         return self.model_name
    
# google_judge = GoogleDeepEvalModel()

json_structure_metric = GEval(
    name="JSON Compliance",
    criteria="""
    1. The output must be a valid JSON object.
    2. It must contain the keys: 'risk_level', 'reasoning', and 'clause_reference'.
    3. If risk_level is 'High', the reasoning must be detailed.
    """,
    evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
    model=groq_judge # <--- Pass the custom model here
)

faithfulness_metric = FaithfulnessMetric(
    threshold=0.7,
    model=groq_judge, # <--- Pass the custom model here
)

with open("./data/test_golden_dataset.json", "r") as f:
    generated_scenarios = json.load(f)
    
manual_scenarios = [
    # Scenario 1: Conditional Logic (Termination)
    # The agent must catch BOTH conditions: "inactive > 1 year" AND "no paid account"
    # {
    #     "question": "Can OpenAI delete my account just because I haven't used it in a while?",
    #     "expected_output": "Yes, but only if your account has been inactive for over a year AND you do not have a paid account. If you have a paid subscription, inactivity alone is not grounds for termination.",
    #     "context": ["We also may terminate your account if it has been inactive for over a year and you do not have a paid account."]
    # },

    # # Scenario 2: Numerical Reasoning (Liability Cap)
    # # The agent must understand the 'Greater Of' logic. $120 > $100, so the limit is $120.
    # {
    #     "question": "I paid OpenAI $10 a month for the last year ($120 total). If I sue them for damages, what is the maximum amount I can get?",
    #     "expected_output": "$120",
    #     "expected_reasoning": "The terms limit liability to the greater of (1) the amount paid in the last 12 months ($120) or (2) $100. Since $120 is greater than $100, the limit is $120.",
    #     "context": ["OUR AGGREGATE LIABILITY ... WILL NOT EXCEED ​​THE GREATER OF THE AMOUNT YOU PAID ... DURING THE 12 MONTHS ... OR ONE HUNDRED DOLLARS ($100)."]
    # },

    # # Scenario 3: Synthesis (Ownership vs. Usage Rights)
    # # Common hallucination: Agents often say "You own it, full stop" without mentioning the training rights.
    # {
    #     "question": "Do I own the images I generate with DALL-E, and will OpenAI use them?",
    #     "expected_output": "Yes, you own the Output. However, OpenAI may use your Content to train their models unless you explicitly opt out.",
    #     "context": ["You (a) retain your ownership rights in Input and (b) own the Output.", "We may use Content to provide, maintain, develop, and improve our Services", "If you do not want us to use your Content to train our models, you can opt out"]
    # }
    # Scenario 4: Handling Silence (The "Four Corners" Test)
    # The contract likely does NOT mention a specific "99.9% uptime guarantee" refund.
    # The agent must NOT hallucinate a standard SLA refund if it's not written.
    {
        "question": "If ChatGPT is down for 24 hours, do I get a partial refund for that day?",
        "expected_output": "The provided text does not specify a refund policy for downtime.",
        "expected_reasoning": "The retrieved context discusses general refunds (non-refundable) but does not contain a specific Service Level Agreement (SLA) or clause offering credits for downtime.",
        # Context note: Ensure the retrieved docs actually DON'T have this clause.
        "context": ["Payments are non-refundable, except where required by law."] 
    },
    # Scenario 5: Specific Overrides General
    # General terms might say "Confidentiality is important," but API terms might have a specific exclusion.
    # (Tailor this to your specific documents if you have multiple files like "Privacy Policy" vs "Terms")
    # {
    #     "question": "Does the confidentiality clause apply if I am on the free tier?",
    #     "expected_output": "No, confidentiality obligations typically apply to Enterprise or Paid tiers. The free tier data may be used for training.",
    #     "expected_reasoning": "The specific clause for 'Free Tier' usage rights overrides the general confidentiality section.",
    #     "context": ["Confidentiality obligations apply to Enterprise users.", "Free tier content may be used to improve services."]
    # }
    # Scenario 6: Legal Advice Guardrail
    # The agent should analyze the TEXT, not advise on STRATEGY.
    # {
    #     "question": "My account was banned unfairly. Should I sue them in Small Claims Court or go to Arbitration?",
    #     "expected_output": "The Terms require you to use Arbitration for disputes. I cannot advise you on which legal strategy to pursue.",
    #     "expected_reasoning": "The text contains a 'Mandatory Arbitration' clause. The agent must cite this fact but refuse to give strategic litigation advice.",
    #     "context": ["You and OpenAI agree to resolve any claims... through final and binding arbitration."]
    # }
]

full_test_suite = manual_scenarios

@pytest.fixture(autouse=True)
def slow_down_tests():
    yield # This runs the actual test
    print("\n⏳ Sleeping for 10 seconds to respect Groq Rate Limits...")
    time.sleep(10) # 10s wait = Max 6 tests per minute

@pytest.mark.parametrize("scenario", full_test_suite)
def test_legal_auditor_logic(scenario):
    user_query = scenario.get("question")
    
    if not user_query:
        pytest.fail(f"Scenario missing 'input' or 'question' key: {scenario.keys()}")

    print(f"\nTesting Query: {user_query}")
    
    initial_state = {
        "query": user_query,
        "retry_count": 0,
        "documents": []
    }
    
    final_state = graph.invoke(initial_state)
    
    actual_output_dict = final_state.get("final_answer", {})
    actual_output_str = json.dumps(actual_output_dict, indent = 2)
    
    retrieved_docs = final_state.get("documents", [])
    retrieval_contexts = [doc.page_content for doc in retrieved_docs]
    
    
    # === ADD THIS DEBUG BLOCK ===
    print("\n" + "="*40)
    print(f"🔹 QUERY: {user_query}")
    print("-" * 20)
    print(f"🔸 SYSTEM OUTPUT:\n{actual_output_str}")
    print("-" * 20)
    print(f"🟢 GROUND TRUTH:\n{scenario.get('expected_output')}")
    print("="*40 + "\n")
    # ============================
    
    
    test_case = LLMTestCase(
        input=scenario["question"],
        actual_output=actual_output_str, 
        retrieval_context=retrieval_contexts,
        expected_output=scenario.get("expected_output") or scenario.get("answer", "")
    )
    
    json_structure_metric.measure(test_case)
    print(f"JSON Compliance Reason: {json_structure_metric.reason}")
    
    faithfulness_metric.measure(test_case)
    print(f"Faithfulness Reason: {faithfulness_metric.reason}")
    
    assert_test(test_case, [json_structure_metric, faithfulness_metric])
    
if __name__ == "__main__":
    sys.exit(pytest.main(["-vv", __file__]))