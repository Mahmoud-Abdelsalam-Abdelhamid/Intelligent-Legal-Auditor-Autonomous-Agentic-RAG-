import operator
from typing import List, TypedDict, Annotated, Optional
from langchain_core.documents import Document

class AgentState(TypedDict):
    query: str
    plan: List[str]
    documents: Annotated[List[Document], operator.add] 
    final_answer: dict
    feedback: Optional[str]
    validation_status: str       # "valid" or "retry"
    retry_count: int             # Default: 0