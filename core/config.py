import os
from dotenv import load_dotenv # New import
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.chat_models import ChatOllama
from langchain_nvidia_ai_endpoints import ChatNVIDIA

# Load variables from .env file
load_dotenv()

if not os.environ.get("GROQ_API_KEY"):
    raise ValueError("GROQ_API_KEY not found! Check your .env file.")

PRIMARY_MODEL = "llama-3.3-70b-versatile"
FALLBACK_MODEL = "llama-3.1-8b-instant" 
GEMINI_MODEL = "gemini-2.5-flash"


def get_llm():
    """
    Returns the primary model with a fallback mechanism.
    If Llama 3.3 is exhausted, you can manually switch the string below 
    or implement try/except logic in the nodes.
    
    For now, since you are rate-limited, we default to the FALLBACK.
    """
    # 🚨 TEMPORARY: Using Fallback because Primary is 429 Rate Limited
    active_model = PRIMARY_MODEL 
    
    print(f"   🔌 LLM Connected: {active_model}")
    
#     return ChatOllama(
#     model="llama3.1:8b-instruct-q4_K_M",
#     temperature=0,
#     base_url="http://localhost:11434" # Default Ollama port
# )
    # return ChatGoogleGenerativeAI(
    #     model=GEMINI_MODEL,
    #     temperature=0,
    #     google_api_key=os.getenv("GOOGLE_API_KEY"),
    #     convert_system_message_to_human=True
    # )
    
    # return ChatNVIDIA(
    #     model="meta/llama-3.1-70b-instruct",  # <--- The best model choice
    #     temperature=0,
    #     max_tokens=1024
    # )
    
    return ChatGroq(
        model=active_model, 
        temperature=0,
        max_retries=2, 
    )

# Initialize the LLM
llm = get_llm()