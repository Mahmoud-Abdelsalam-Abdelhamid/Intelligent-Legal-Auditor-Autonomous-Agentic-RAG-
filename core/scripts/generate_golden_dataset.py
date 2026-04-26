import json
import os
import random
import sys
import time
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field
from typing import List
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(project_root)


from core.config import llm

# --- CONFIGURATION ---
DATA_DIR = "./data/terms_data"
OUTPUT_FILE = "./data/test_golden_dataset.json"


CHUNK_SIZE = 8000
CHUNK_OVERLAP = 500
QUESTIONS_PER_CHUNK = 3  # Generate 2 hard questions per chunk

# --- PYDANTIC SCHEMA ---
class QAPair(BaseModel):
    question: str = Field(description="A complex question about the legal terms")
    answer: str = Field(description="The correct answer based ONLY on the text")
    context_snippet: str = Field(description="The exact quote from the text used to answer")
    reasoning_type: str = Field(description="Type: 'Numerical', 'Conditional', or 'Fact'")

class SyntheticDataset(BaseModel):
    pairs: List[QAPair]


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n## ", "\n\n", "\n", " ", ""] # Prioritize splitting by Headers
)

generator_prompt = ChatPromptTemplate.from_template("""
You are an expert Legal Tech engineer creating a test dataset.
Generate {num_questions} DIFFICULT question-answer pairs based on this specific text chunk.

CONTEXT CHUNK:
{text_chunk}

RULES:
1. Focus on specific details (numbers, dates, conditions) in THIS chunk.
2. If the chunk is just a table of contents or useless fluff, return an empty list.
3. Questions must be answerable purely from the provided text.

Generate JSON.
""")

def load_and_generate():
    all_test_cases = []
    
    for filename in os.listdir(DATA_DIR):
        if filename.endswith(".json"):
            file_path = os.path.join(DATA_DIR, filename)
            
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                company = data.get("company", "Unknown")
                full_text = data.get("terms", "")
                
                print(f"\n🔹 Processing {company} ({len(full_text)} chars)...")
                
                chunks = text_splitter.split_text(full_text)
                print(f"   ➡️  Split into {len(chunks)} chunks.")
                
                for i, chunk in enumerate(chunks):
                    
                    if i % 5 != 0: 
                        continue

                    # Skip small footer chunks
                    if len(chunk) < 500:
                        continue
                        
                    print(f"   Generating questions for Chunk {i+1}/{len(chunks)}...")
                    
                    chain = generator_prompt | llm.with_structured_output(SyntheticDataset)
                    
                    try:
                        result = chain.invoke({
                            "text_chunk": chunk,
                            "num_questions": QUESTIONS_PER_CHUNK
                        })
                        
                        for pair in result.pairs:
                            all_test_cases.append({
                                "company": company,
                                "question": pair.question,
                                "answer": pair.answer,
                                "context_snippet": pair.context_snippet,
                                "reasoning_type": pair.reasoning_type
                            })
                    except Exception as e:
                        print(f"   ⚠️ Error on chunk {i}: {e}")

                    print("     Thinking (sleeping 10s)...") 
                    time.sleep(10)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as out_f:
        json.dump(all_test_cases, out_f, indent=2)
        print(f"\n✅ SUCCESS! Generated {len(all_test_cases)} test cases across {len(os.listdir(DATA_DIR))} files.")
        print(f"saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    load_and_generate()