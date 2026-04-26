# from typing import Optional
# from langsmith import traceable
# from pydantic import BaseModel, Field
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.output_parsers import PydanticOutputParser # <--- NEW IMPORT

# from core.config import llm
# from core.State import AgentState

# # --- 1. DEFINE SCHEMA ---
# class LegalRisk(BaseModel):
#     """The final structured audit report for a specific legal query."""
#     answer: str = Field(description="A direct, plain-language answer to the user's question.")
#     clause_reference: str = Field(description="The exact quote from the document that supports the answer.")
#     risk_level: str = Field(description="The risk level: 'Low', 'Medium', or 'High' based on standard legal norms.")
#     financial_impact: Optional[str] = Field(description="Extract any specific costs (e.g., '$500'). If none, use 'N/A'.")
#     reasoning: str = Field(description="A brief explanation of why this risk level was assigned.")

# # --- 2. SETUP PARSER ---
# # This teaches the local model how to format the output
# parser = PydanticOutputParser(pydantic_object=LegalRisk)

# # --- 3. UPDATED PROMPT ---
# # We merge instructions + context + format_instructions into one block
# generator_prompt_template = ChatPromptTemplate.from_template("""
# You are a Senior Legal Auditor. 
# Analyze the provided Contract Context to answer the User Query.

# ### STRICT INSTRUCTIONS
# 1. **Answer strictly** based on the provided context.
# 2. **Handle "No Fee" Scenarios:** If the text says "cancel anytime" or "free to stop", the answer is: "There is no termination fee." 
# 3. **Risk Level:**
#    - Specific fee exists -> **High** or **Medium**.
#    - No fee / Cancel anytime -> **Low**.
#    - Text silent -> **Unknown**.
# 4. **Extract exact quotes** for 'clause_reference'.

# ### FORMATTING INSTRUCTIONS
# {format_instructions}

# ### CONTRACT CONTEXT
# {context_block}

# ### USER QUERY
# {query}
# """)

# @traceable(name="Legal Generator")
# def generator(state: AgentState):
#     print("\n📝 GENERATOR: Synthesizing Final Answer (Local)...")
    
#     query = state['query']
#     docs = state['documents']
    
#     context = "\n\n".join([doc.page_content for doc in docs])
    
#     # --- 4. NEW CHAIN ---
#     # Prompt -> LLM -> Parser
#     chain = generator_prompt_template | llm | parser
    
#     try:
#         # We inject the parser instructions here
#         result: LegalRisk = chain.invoke({
#             "query": query, 
#             "context_block": context,
#             "format_instructions": parser.get_format_instructions()
#         })
        
#         print(f"   ✅ Generated Structured Answer: Risk Level = {result.risk_level}")
#         return {"final_answer": result.model_dump()}
        
#     except Exception as e:
#         print(f"   ❌ Generation Error: {e}")
#         # Robust Fallback for Resume: Return a safe error dict
#         return {
#             "final_answer": {
#                 "answer": "I found relevant documents but could not format the final answer strictly.",
#                 "risk_level": "Unknown",
#                 "reasoning": f"Local model parsing error: {str(e)[:100]}...",
#                 "clause_reference": "See retrieved documents."
#             }
#         }



# ==========================================================================================================================================
# OLD WORKING CODE
# ===========================================================================================================================================

from typing import Optional
from langsmith import traceable
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate

from core.config import llm
from core.State import AgentState


class LegalRisk(BaseModel):
    """The final structured audit report for a specific legal query."""
    
    answer: str = Field(
        description="A direct, plain-language answer to the user's question."
    )
    
    clause_reference: str = Field(
        description="The exact quote from the document that supports the answer."
    )
    
    risk_level: str = Field(
        description="The risk level: 'Low', 'Medium', or 'High' based on standard legal norms."
    )
    
    financial_impact: Optional[str] = Field(
        description="Extract any specific costs, fees, or penalties (e.g., '$500', '2 months revenue'). If none, use 'N/A'."
    )
    
    reasoning: str = Field(
        description="A brief explanation of why this risk level was assigned."
    )


generator_model = llm.with_structured_output(LegalRisk)

generator_system_prompt = """
You are a Literal Forensic Contract Analyzer. 
Your goal is to assess risk based ONLY on the explicit text provided. 
You must treat the provided context as the entire universe of truth.

USER QUERY: {query}

CONTRACT CONTEXT:
{context_block}

---
CRITICAL RULES (READ CAREFULLY):

1. **The "Four Corners" Rule:** - Do not use outside knowledge of legal standards, common practices, or "fairness."
   - If the text does not explicitly say "advance notice will be provided," you MUST assume NO notice is guaranteed.
   - If the text does not explicitly mention a fee, you MUST NOT invent one.

2. **Handling Silence:** - If the context mentions a right to terminate but is silent on notice periods, your answer must state: "The text does not specify a notice period."
   - Do not infer that "standard procedure" applies.

3. **Risk & Fees:**
   - **High Risk:** Explicit mention of liquidated damages, "remaining contract value," or percentage-based penalties.
   - **Low Risk:** "Cancel anytime," "no penalty," or pay only for services used.
   - **Unknown Risk:** If the text describes termination but does not mention fees at all.

4. **Citation Requirement:** - Every factual claim in your answer must be backed by a verbatim quote in 'clause_reference'.

5. **Damages vs. Refunds:**
   - If the user asks about "suing", "damages", or "lawsuits", look specifically for the **"Limitation of Liability"** clause.
   - Do NOT use the "Refund Policy" to answer questions about lawsuits.

OUTPUT INSTRUCTIONS:
- Answer the user's query directly.
- If the answer depends on a condition (e.g., "inactive for 1 year"), state the condition clearly.
"""
    
generator_prompt_template = ChatPromptTemplate.from_messages([
    ("system", generator_system_prompt)
])

@traceable(name="Legal Generator")
def generator(state: AgentState):
    print("\n📝 GENERATOR: Synthesizing Final Answer...")
    
    query = state['query']
    docs = state['documents']
    
    context = "\n\n".join([doc.page_content for doc in docs])
    
    chain = generator_prompt_template | generator_model
    
    try:
        result: LegalRisk = chain.invoke({"query": query, "context_block": context})
        print(f"  ✅ Generated Structured Answer: Risk Level = {result.risk_level}")
        return {"final_answer": result.model_dump()}
        
    except Exception as e:
        print(f"  ❌ Generation Error: {e}")
        # Fallback if JSON fails
        return {"final_answer": {"answer": "Error generating structured response.", "error": str(e)}}