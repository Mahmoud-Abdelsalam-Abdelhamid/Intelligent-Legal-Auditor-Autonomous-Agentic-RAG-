import sys
import os
import json
import time
import statistics
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
# --- 1. SETUP ENVIRONMENT ---
# Fix path to find 'core'
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(project_root)

load_dotenv()

from core.Agent import graph
from langchain_google_genai import ChatGoogleGenerativeAI
from deepeval.test_case import LLMTestCase
from deepeval.metrics import FaithfulnessMetric
from deepeval.models.base_model import DeepEvalBaseLLM

# --- 2. SETUP JUDGE (Gemini 2.5) ---
class LocalDeepEvalModel(DeepEvalBaseLLM):
    def __init__(self, model_name="llama3.1:8b-instruct-q4_K_M"):
        self.model_name = model_name
        self.model = ChatOllama(model=model_name, temperature=0)

    def load_model(self):
        return self.model

    def _clean_json(self, text: str) -> str:
        """
        Force-cleans the output to ensure only valid JSON is returned.
        Removes markdown code blocks and polite conversational filler.
        """
        # 1. Strip Markdown
        clean_text = text.replace("```json", "").replace("```", "").strip()
        
        # 2. Find the actual JSON object (from first '{' to last '}')
        start_idx = clean_text.find("{")
        end_idx = clean_text.rfind("}")
        
        if start_idx != -1 and end_idx != -1:
             # Return only the JSON part
            return clean_text[start_idx : end_idx + 1]
            
        return clean_text

    def generate(self, prompt: str) -> str:
        raw_res = self.model.invoke(prompt).content
        return self._clean_json(raw_res)

    async def a_generate(self, prompt: str) -> str:
        raw_res = await self.model.ainvoke(prompt)
        return self._clean_json(raw_res.content)

    def get_model_name(self):
        return self.model_name

print("⚖️  Initializing Judge (Local Llama-3)...")
local_judge = LocalDeepEvalModel()

# print("⚖️  Initializing Judge (Gemini)...")
# google_judge = GoogleDeepEvalModel()
faithfulness_metric = FaithfulnessMetric(threshold=0.5, model=local_judge)

# --- 3. LOAD DATA ---
DATA_PATH = os.path.join(project_root, "data", "test_golden_dataset.json")
try:
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    print(f"✅ Loaded {len(dataset)} test cases.")
except Exception as e:
    print(f"❌ Could not load dataset: {e}")
    sys.exit(1)

# --- 4. THE ROBUST LOOP ---
results = []
print("\n🚀 STARTING ROBUST EVALUATION")
print("==================================================")

for i, scenario in enumerate(dataset[:7]):
    print(f"\n🔹 Case {i+1}/{len(dataset)}")
    
    # 1. Rate Limit Protection
    print("   ⏳ Cooling down (4s)...")
    time.sleep(4) 
    
    try:
        # 2. Prepare Input
        user_query = scenario.get("input") or scenario.get("question")
        expected_output = scenario.get("expected_output") or scenario.get("answer", "")
        
        print(f"   ❓ Query: {user_query[:50]}...")

        # 3. Run Agent
        initial_state = {"query": user_query, "retry_count": 0, "validation_status": "start", "documents": []}
        final_state = graph.invoke(initial_state)
        
        # 4. Extract Output
        actual_output = final_state.get('final_answer', {})
        if isinstance(actual_output, dict):
            actual_output_str = json.dumps(actual_output)
        else:
            actual_output_str = str(actual_output)
            
        retrieved_docs = [doc.page_content for doc in final_state.get('documents', [])]
        
        # 5. Evaluate (Faithfulness Only for Speed)
        test_case = LLMTestCase(
            input=user_query,
            actual_output=actual_output_str,
            retrieval_context=retrieved_docs,
            expected_output=expected_output
        )
        
        faithfulness_metric.measure(test_case)
        score = faithfulness_metric.score
        is_successful = faithfulness_metric.is_successful()
        
        print(f"   🎯 Score: {score} | {'✅ PASS' if is_successful else '❌ FAIL'}")
        
        results.append({
            "query": user_query,
            "score": score,
            "pass": is_successful,
            "reason": faithfulness_metric.reason
        })

    except Exception as e:
        print(f"   ⚠️ CRASH: {e}")
        results.append({
            "query": user_query,
            "score": 0.0,
            "pass": False,
            "reason": f"System Crash: {str(e)}"
        })

# --- 5. FINAL REPORT ---
print("\n\n==================================================")
print("📊 FINAL VERDICT")
print("==================================================")

total = len(results)
passed = sum(1 for r in results if r['pass'])
avg_score = statistics.mean([r['score'] for r in results]) if results else 0

print(f"Total Cases: {total}")
print(f"Passed:      {passed}")
print(f"Failed:      {total - passed}")
print(f"Accuracy:    { (passed/total)*100 :.1f}%")
print(f"Avg Score:   {avg_score:.2f}")

# Save Report
report_path = os.path.join(project_root, "evaluation_report.json")
with open(report_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\n📝 Detailed report saved to: {report_path}")