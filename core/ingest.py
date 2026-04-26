import os
import json
import glob
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from .DocumentChunks import chunk_legal_text, clean_markdown  


current_dir = os.path.dirname(os.path.abspath(__file__))

# Go up one level to the Project Root (Intelligent_Legal_Auditor)
project_root = os.path.dirname(current_dir)

# Construct the exact path to 'data/terms_data'
data_path = os.path.join(project_root, 'data', 'terms_data')
DB_PATH = os.path.join(project_root, 'data', 'chroma_db_V2')

def ingest_data():
    print("🧠 Initializing Embedding Model (all-MiniLM-L6-v2)...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    json_files = glob.glob(os.path.join(data_path, "*.json"))
    
    if not json_files:
        print(f"❌ No JSON files found in '{data_path}'")
        return

    print(f"📂 Found {len(json_files)} files: {[os.path.basename(f) for f in json_files]}")

    all_chunks = []
    
    for file_path in json_files:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            company = data.get("company", "Unknown")
            url = data.get("url", "Unknown")
            terms = data.get("terms", "")
            
            if not terms:
                print(f"⚠️  Skipping {os.path.basename(file_path)}: No 'terms' key found.")
                continue

            print(f"\n⚙️  Processing {company}...")
            
            cleaned_terms = clean_markdown(terms)
            chunks = chunk_legal_text(cleaned_terms, company, url)
            
            all_chunks.extend(chunks)
            
            print(f"    -> Added {len(chunks)} chunks.")

        except json.JSONDecodeError:
            print(f"❌ Error: Could not parse JSON in {file_path}")
        except Exception as e:
            print(f"❌ Error processing {file_path}: {e}")
    
    if all_chunks:
        print(f"\n💾 Saving {len(all_chunks)} total chunks to ChromaDB at '{DB_PATH}'...")
        vectore_store = Chroma.from_documents(
            documents= all_chunks,
            embedding= embeddings,
            persist_directory= DB_PATH
        )
        print("✅ Ingestion Complete! Database is ready.")
    else:
        print("⚠️ No valid chunks were generated. Database not updated.")

if __name__ == "__main__":
    ingest_data()