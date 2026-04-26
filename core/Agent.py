from typing import Literal
from langgraph.graph import StateGraph, END
from .State import AgentState

from .nodes.Planner import planner
from .nodes.Executer import executer
from .nodes.Generator import generator
from .nodes.Validator import validator


def router(state: AgentState):
    status = state['validation_status']
    retries = state['retry_count']
    
    if status == "retry":
        if retries > 3:
            print("  ⚠️ Max retries reached. Stopping loop.")
            return "END"
        
        print(f"  🔄 Looping back to PLANNER (Attempt {retries+1})...")
        return "planner"
    return "END"


agent = StateGraph(AgentState)

agent.add_node("planner", planner)
agent.add_node("executer", executer)
agent.add_node("generator", generator)
agent.add_node("validator", validator)

agent.set_entry_point("planner")
agent.add_edge("planner", "executer")
agent.add_edge("executer", "generator")
agent.add_edge("generator", "validator")

agent.add_conditional_edges(
    source= "validator",
    path= router,
    path_map= {
        "planner": "planner",
        "END": END
    }
)

app = agent.compile()
graph = app

if __name__ == "__main__":
    query = "What is the phone number for the CEO?"
    
    print(f"🚀 STARTING AGENT for: '{query}'")
    
    initial_state = {
        "query": query,
        "retry_count": 0,
        "feedback": None
    }
    
    result = app.invoke(initial_state)
    
    print("\n🏁 FINAL OUTPUT 🏁")
    print(result.get("final_answer"))

