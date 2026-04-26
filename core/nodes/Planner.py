# from typing import List
# from langsmith import traceable
# from pydantic import BaseModel, Field
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.output_parsers import PydanticOutputParser # <--- NEW IMPORT

# from core.config import llm
# from core.State import AgentState

# # --- 1. DEFINE SCHEMA ---
# class SearchStep(BaseModel):
#     step_number: int
#     description: str = Field(description="A precise keyword search query or section lookup.")
#     reasoning: str = Field(description="Why we need this specific text chunk.")

# class ContractAnalysisPlan(BaseModel):
#     """The execution plan for analyzing the legal document text."""
#     steps: List[SearchStep]

# # --- 2. SETUP PARSER ---
# # This tool teaches Llama 3 how to write the specific JSON we need
# parser = PydanticOutputParser(pydantic_object=ContractAnalysisPlan)

# # --- 3. UPDATED PROMPT ---
# # We added {format_instructions} to the bottom so the model knows the schema.
# prompt_template = ChatPromptTemplate.from_template("""
# ### ROLE
# You are a Search Query Generator for a legal database. Your job is to break a complex question into 3-5 specific keyword search steps.

# ### STRICT RULES
# 1. **OUTPUT ONLY KEYWORDS:** Do NOT use phrases like "Search for".
# 2. **SPLIT ENTITIES:** If the user asks about multiple companies, create SEPARATE search steps for each.
# 3. **USE VARIATIONS:** Generate 1 specific query ("fee") and 1 broad query ("policy").
# 4. **MAX 5 STEPS.**

# ### BAD VS GOOD EXAMPLES
# 🔴 **BAD:** "Search for Google and AWS fees"
# 🟢 **GOOD:** "Google termination fee", "AWS cancellation charges"

# ### FORMATTING INSTRUCTIONS
# {format_instructions}

# ### USER QUERY
# {query}
# """)

# # --- 4. PLANNER NODE LOGIC ---
# @traceable(name="Legal Planner")
# def planner(state: AgentState):
#     print("--- PLANNER: Generating Audit Strategy (Local) ---")
    
#     query = state.get("query")
#     feedback = state.get("feedback", None)
    
#     if feedback:
#         print(f"  🧠 PLANNER: Received negative feedback. Adjusting strategy...")
#         query += f"\n\nIMPORTANT: Previous plan FAILED. Feedback: {feedback}\nGenerate a NEW plan."
    
#     # NEW: Create the chain with the parser
#     # The parser automatically injects the schema instructions into the prompt
#     chain = prompt_template | llm | parser
    
#     try:
#         # We pass the format instructions into the prompt
#         result: ContractAnalysisPlan = chain.invoke({
#             "query": query,
#             "format_instructions": parser.get_format_instructions()
#         })
        
#         # Extract steps
#         plan_steps = [step.description for step in result.steps]
        
#         return {
#             "plan": plan_steps,
#             "feedback": None
#         }
        
#     except Exception as e:
#         # Fallback for Local Models: If it fails to parse, return a basic search
#         print(f"   ⚠️ Planner Parsing Error: {e}")
#         return {
#             "plan": [query, "termination clause", "penalty fees"], # Fallback plan
#             "feedback": None
#         }

# ==========================================================================================================================================
# OLD WORKING CODE
# ===========================================================================================================================================
from typing import List
from langsmith import traceable
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate

from core.config import llm
from core.State import AgentState

# --- 1. GEMINI-SAFE PROMPT (Merged System + User) ---
# We use .from_template() instead of .from_messages() to avoid empty content errors.

# --- IMPROVED PROMPT FOR LLAMA 3 ---
prompt_template = ChatPromptTemplate.from_messages([
    ("system", """
### ROLE
You are a Search Query Generator for a legal database. Your job is to break a complex question into 3-5 specific keyword search steps.

### STRICT RULES
1. **OUTPUT ONLY KEYWORDS:** Do NOT use phrases like "Search for".
2. **SPLIT ENTITIES:** If the user asks about multiple companies (e.g., "Google and AWS"), create SEPARATE search steps for each company.
   - 🔴 Bad: "Google AWS termination fee"
   - 🟢 Good: "Google termination fee", "AWS termination fee"
3. **USE VARIATIONS:** For each entity, generate 1 specific query ("fee") and 1 broad query ("policy" or "terms").
   - 🔴 Bad: "Google termination fee" (Too narrow, misses "free to stop")
   - 🟢 Good: "Google termination fee", "Google cancellation terms"
4. **MAX 5 STEPS.**
5. **FORMATTING:** Do not use single quotes.

### DOMAIN KNOWLEDGE (Legal Translation)
Use these mappings to translate user intent into legal terms:
1. User: "Can I sue them?" -> Keywords: "Limitation of liability", "Dispute resolution", "Arbitration"
2. User: "refund" -> Keywords: "Refund policy", "Cancellation terms"
3. User: "damages" -> Keywords: "Aggregate liability cap", "Indemnification", "Exclusion of damages"

### BAD VS GOOD EXAMPLES

🔴 **BAD:**
1. Search for Google and AWS fees.
2. Google AWS cancellation policy.

🟢 **GOOD (Raw Keywords):**
1. Google termination fee
2. AWS cancellation charges
3. Google early exit penalty
4. AWS termination notice period
"""),
    ("user", "{query}")
])
# prompt_template = ChatPromptTemplate.from_template("""
# ### ROLE
# You are a Search Query Generator for a legal database. Your job is to break a complex question into 3-5 specific keyword search steps.

# ### STRICT RULES
# 1. **OUTPUT ONLY KEYWORDS:** Do NOT use phrases like "Search for".
# 2. **SPLIT ENTITIES:** If the user asks about multiple companies (e.g., "Google and AWS"), create SEPARATE search steps for each company.
#    - 🔴 Bad: "Google AWS termination fee"
#    - 🟢 Good: "Google termination fee", "AWS termination fee"
# 3. **USE VARIATIONS:** For each entity, generate 1 specific query ("fee") and 1 broad query ("policy" or "terms").
#    - 🔴 Bad: "Google termination fee" (Too narrow, misses "free to stop")
#    - 🟢 Good: "Google termination fee", "Google cancellation terms"
# 4. **MAX 5 STEPS.**
# 5. **FORMATTING:** Do not use single quotes.

# ### BAD VS GOOD EXAMPLES

# 🔴 **BAD:**
# 1. Search for Google and AWS fees.
# 2. Google AWS cancellation policy.

# 🟢 **GOOD (Raw Keywords):**
# 1. Google termination fee
# 2. AWS cancellation charges
# 3. Google early exit penalty
# 4. AWS termination notice period

# ### USER QUERY
# {query}
# """)

# --- 2. STRUCTURED OUTPUT SCHEMA ---
class SearchStep(BaseModel):
    step_number: int
    description: str = Field(description="A precise keyword search query or section lookup.")
    reasoning: str = Field(description="Why we need this specific text chunk.")

class ContractAnalysisPlan(BaseModel):
    """The execution plan for analyzing the legal document text."""
    steps: List[SearchStep]

planner_model = llm.with_structured_output(ContractAnalysisPlan)

# --- 3. PLANNER NODE LOGIC ---
@traceable(name="Legal Planner")
def planner(state: AgentState):
    print("--- PLANNER: Generating Audit Strategy ---")
    
    query = state.get("query")
    feedback = state.get("feedback", None)
    
    # Handle Feedback Loop
    if feedback:
        print(f"  🧠 PLANNER: Received negative feedback: '{feedback}'. Adjusting strategy...")
        query += f"\n\nIMPORTANT: Your previous plan FAILED. Feedback: {feedback}\nGenerate a NEW, DIFFERENT plan."
    
    # Run Chain
    chain = prompt_template | planner_model
    result: ContractAnalysisPlan = chain.invoke({"query": query})
    
    # Extract just the search strings
    plan_steps = [step.description for step in result.steps]
    
    return {
        "plan": plan_steps,
        "feedback": None
    }