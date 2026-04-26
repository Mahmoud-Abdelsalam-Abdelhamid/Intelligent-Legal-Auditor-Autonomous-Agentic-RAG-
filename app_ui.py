import streamlit as st
import time


from core.Agent import app  # Import your compiled graph
from core.State import AgentState

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Intelligent Legal Auditor",
    page_icon="⚖️",
    layout="wide"
)

# --- CUSTOM CSS (For that "Enterprise" look) ---
# --- CUSTOM CSS (Refined for Professional Contrast) ---
# --- CUSTOM CSS (Fixed Text Contrast) ---
st.markdown("""
    <style>
    /* Main Chat Message Container */
    .stChatMessage {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        border-radius: 12px;
        padding: 15px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    /* FORCE Dark Text for all content inside chat messages */
    .stChatMessage p, .stChatMessage li, .stChatMessage span, .stChatMessage div {
        color: #2d3748 !important; /* Dark Slate Blue */
    }

    /* Headers (e.g. "Findings") */
    .stMarkdown h3 {
        color: #1a202c !important; /* Almost Black */
        font-family: 'Helvetica Neue', sans-serif;
    }
    
    /* Solid Color Risk Badges (White text on color background) */
    .risk-high {
        background-color: #e53e3e;
        color: white !important; /* Force white text ONLY for badge */
        font-weight: bold;
        padding: 4px 12px;
        border-radius: 16px;
        font-size: 0.9em;
        display: inline-block;
    }
    
    .risk-medium {
        background-color: #dd6b20;
        color: white !important;
        font-weight: bold;
        padding: 4px 12px;
        border-radius: 16px;
        font-size: 0.9em;
        display: inline-block;
    }
    
    .risk-low {
        background-color: #38a169;
        color: white !important;
        font-weight: bold;
        padding: 4px 12px;
        border-radius: 16px;
        font-size: 0.9em;
        display: inline-block;
    }
    
    .risk-unknown {
        background-color: #718096;
        color: white !important;
        font-weight: bold;
        padding: 4px 12px;
        border-radius: 16px;
        font-size: 0.9em;
        display: inline-block;
    }
    
    /* Evidence Box Styling */
    .evidence-box {
        background-color: #f7fafc;
        border-left: 4px solid #4299e1;
        padding: 15px;
        margin-bottom: 10px;
        border-radius: 4px;
        font-size: 0.9em;
        color: #2d3748 !important; /* Force text dark */
    }
    
    /* Fix for Streamlit Expander Text */
    .streamlit-expanderContent p {
        color: #2d3748 !important;
    }
    </style>
""", unsafe_allow_html=True)

# --- HEADER ---
col1, col2 = st.columns([1, 5])
with col1:
    st.image("https://cdn-icons-png.flaticon.com/512/2666/2666505.png", width=80)
with col2:
    st.title("Intelligent Legal Auditor")
    st.markdown("*Autonomous Agentic RAG for Contract Analysis*")

# --- SIDEBAR (History & Settings) ---
with st.sidebar:
    st.header("⚙️ Control Panel")
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()
    
    st.markdown("---")
    st.markdown("### 🧠 How it works")
    st.markdown("1. **Planner:** Breaks down queries.")
    st.markdown("2. **Executor:** Searches Vector DB (k=50).")
    st.markdown("3. **Generator:** Analyzes risk.")
    st.markdown("4. **Validator:** Double-checks answers.")

# --- SESSION STATE ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- DISPLAY CHAT HISTORY ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        # If there's extra data (like risk level), display it
        if "data" in message:
            data = message["data"]
            with st.expander("📊 Audit Details & Evidence", expanded=False):
                st.markdown(f"**Risk Level:** {data.get('risk_badge', 'N/A')}", unsafe_allow_html=True)
                st.markdown(f"**Reasoning:** {data.get('reasoning')}")
                st.markdown("---")
                st.markdown("**🔍 Evidence Used:**")
                for doc in data.get('docs', []):
                    st.caption(f"📄 {doc[:150]}...")

# --- USER INPUT ---
if prompt := st.chat_input("Ask a legal question about the contracts..."):
    # 1. Display User Message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Display Assistant "Thinking"
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        status_placeholder = st.status("🕵️ Auditor is working...", expanded=True)
        
        try:
            # --- RUN THE AGENT ---
            initial_state = {
                "query": prompt,
                "retry_count": 0,
                "feedback": None
            }
            
            # Streaming updates simulation (Optional visual flair)
            status_placeholder.write("🧠 Planner: Analyzing query structure...")
            time.sleep(1) # Fake delay for UI feel
            
            status_placeholder.write("🔍 Executor: Retrieving & Reranking 50 chunks...")
            
            # CALL THE ACTUAL BACKEND
            result = app.invoke(initial_state)
            
            status_placeholder.write("📝 Generator: Synthesizing legal opinion...")
            status_placeholder.write("⚖️ Validator: Quality checking answer...")
            
            status_placeholder.update(label="✅ Audit Complete", state="complete", expanded=False)
            
            # --- PARSE RESULTS ---
            final = result.get("final_answer", {})
            answer_text = final.get("answer", "No answer generated.")
            risk_level = final.get("risk_level", "Unknown")
            reasoning = final.get("reasoning", "N/A")
            docs = [doc.page_content for doc in result.get("documents", [])[:3]] # Top 3 docs
            
            # Create Risk Badge HTML
            if "High" in risk_level: badge_class = "risk-high"
            elif "Medium" in risk_level: badge_class = "risk-medium"
            else: badge_class = "risk-low"
            risk_badge = f'<span class="{badge_class}">{risk_level}</span>'

            # Display Answer
            full_response = f"### 💡 Findings\n{answer_text}"
            message_placeholder.markdown(full_response)
            

            with st.expander("📊 Audit Details & Evidence", expanded=True):
                st.markdown(f"**Risk Level:** {risk_badge}", unsafe_allow_html=True)
                st.markdown(f"**Reasoning:** {reasoning}")
                st.markdown("---")
                st.markdown("**🔍 Top Evidence:**")
                
                for i, doc in enumerate(docs):
                    st.markdown(f"""
                    <div class="evidence-box">
                        <b>Chunk {i+1}:</b><br>
                        {doc[:300]}...
                    </div>
                    """, unsafe_allow_html=True)

            # Save to History
            st.session_state.messages.append({
                "role": "assistant", 
                "content": full_response,
                "data": {
                    "risk_badge": risk_badge,
                    "reasoning": reasoning,
                    "docs": docs
                }
            })
            
        except Exception as e:
            status_placeholder.update(label="❌ Error", state="error")
            st.error(f"System Error: {str(e)}")