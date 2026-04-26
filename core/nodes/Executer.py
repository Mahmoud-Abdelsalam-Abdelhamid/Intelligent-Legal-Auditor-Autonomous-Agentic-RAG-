import os
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langsmith import traceable
from sentence_transformers import CrossEncoder
from langchain_huggingface import HuggingFaceEmbeddings

# import ingest
from core.State import AgentState

# ingest.ingest_data()
current_dir = os.path.dirname(os.path.abspath(__file__))

DB_PATH = os.path.join(current_dir, "../../data/chroma_db_V2")

print("🧠 Loading Reranker Model...")
reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = Chroma(
    persist_directory= DB_PATH,
    embedding_function= embeddings
)

@traceable(name="Legal Executer")
def executer(state: AgentState):
    print("\n🕵️ EXECUTOR: Running Search Steps...")
    plan = state['plan']
    unique_docs = {}
    
    for step_query in plan:
        clean_query = step_query.replace("Search for headers:", "").replace("Search for keywords:", "").replace("Search for:", "").strip()
        print(f"  🔍 Database Query: '{step_query}'")
        
        initial_docs = vector_store.similarity_search(clean_query, k = 50)
        pairs = [[step_query, doc.page_content] for doc in initial_docs]
        
        scores = reranker.predict(pairs)
        scored_docs = sorted(zip(initial_docs, scores), key=lambda x: x[1], reverse=True)
        
        top_docs = [doc for doc, score in scored_docs[:5]]
        print(f"     (Reranked: Kept top {len(top_docs)} out of {len(initial_docs)} matches)")
        
        for doc in top_docs:
            labeled_text = f"[[SOURCE: {doc.metadata['company']}]] {doc.page_content}"
        
            if 'url' in doc.metadata:
                labeled_text += f"\n(Ref: {doc.metadata['url']})"
                
            new_doc = Document(
                page_content=labeled_text,
                metadata = doc.metadata
            )
            
            unique_docs[labeled_text] = new_doc

    final_results = list(unique_docs.values())
    
    print(f"  ✅ Retrieved {len(final_results)} unique, high-quality evidence chunks.")
    return {"documents": final_results}