import streamlit as st
import os
import shutil
from utils import logger, compute_file_hash, clean_filename
from chat_logic import RAGChatbot

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

st.set_page_config(
    page_title="NetConfig GenAI",
    page_icon="📡",
    layout="wide"
)

# Custom CSS for Apple-like Aesthetic
CUSTOM_CSS = """
<style>
    /* Main Background & Fonts */
    .stApp {
        background-color: #f5f5f7;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e5e5e5;
    }

    /* Chat Messages */
    .stChatMessage {
        border-radius: 12px;
        padding: 1rem;
        box-shadow: 0 1px 2px rgba(0,0,0,0.05);
    }
    .stChatMessage[data-testid="stChatMessageUser"] {
        background-color: #0071e3;
        color: white;
    }
    .stChatMessage[data-testid="stChatMessageSystem"] {
        background-color: #ffffff;
        color: #1d1d1f;
    }

    /* Input Box */
    .stTextInput > div > div > input {
        border-radius: 12px;
        border: 1px solid #d2d2d7;
        padding: 0.75rem;
    }
    .stTextInput > div > div > input:focus {
        border-color: #0071e3;
        box-shadow: 0 0 0 4px rgba(0,113,227,0.1);
    }
    
    /* Buttons */
    .stButton > button {
        border-radius: 8px;
        font-weight: 500;
        border: none;
        transition: all 0.2s ease;
    }
    .stButton > button:hover {
        transform: scale(1.02);
    }

    /* Header */
    h1, h2, h3 {
        color: #1d1d1f;
        font-weight: 600;
    }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Session State Initialization
# -----------------------------------------------------------------------------

if "chatbot" not in st.session_state:
    st.session_state.chatbot = RAGChatbot()

if "messages" not in st.session_state:
    st.session_state.messages = []

# -----------------------------------------------------------------------------
# Sidebar
# -----------------------------------------------------------------------------

with st.sidebar:
    st.title("🎛️ Network RAG")
    
    st.markdown("### Model Selection")
    selected_model = st.selectbox(
        "Choose LLM Backend",
        ["llama3.2:3b", "phi3.5", "mistral", "custom"],
        index=0
    )
    
    if selected_model != st.session_state.chatbot.model_name:
        st.session_state.chatbot.update_model(selected_model)
        st.toast(f"Switched to {selected_model}")

    st.markdown("---")
    st.markdown("### 📂 Ingestion")
    uploaded_file = st.file_uploader("Upload Config (TXT/PDF)", type=['txt', 'pdf', 'cfg', 'log'])
    
    if uploaded_file:
        file_bytes = uploaded_file.getvalue()
        file_hash = compute_file_hash(file_bytes)
        
        # Simple local save mechanics
        upload_dir = "./uploads"
        os.makedirs(upload_dir, exist_ok=True)
        safe_name = clean_filename(uploaded_file.name)
        save_path = os.path.join(upload_dir, safe_name)
        
        # Check if already processed (simple session check)
        if f"processed_{file_hash}" not in st.session_state:
            with open(save_path, "wb") as f:
                f.write(file_bytes)
            
            with st.spinner("Parsing and Indexing..."):
                success = st.session_state.chatbot.process_file(save_path)
                if success:
                    st.session_state[f"processed_{file_hash}"] = True
                    st.success("File indexed successfully!")
                else:
                    st.error("Failed to process file.")
        else:
            st.info("File already indexed.")

    st.markdown("---")
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# -----------------------------------------------------------------------------
# Main Chat Interface
# -----------------------------------------------------------------------------

st.title("Network Assistant")
st.caption("Ask questions about your uploaded Cisco/Aruba configurations.")

# Display Chat History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "citations" in msg and msg["citations"]:
            with st.expander("📚 Sources"):
                for doc in msg["citations"]:
                    st.markdown(f"**{doc.metadata.get('source', 'Unknown')}** (Line {doc.metadata.get('line_start')}-{doc.metadata.get('line_end')})")
                    st.code(doc.page_content, language="text")

# Chat Input
if prompt := st.chat_input("Ex: What VLANs are configured on the core switch?"):
    # Add User Message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate Response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response_data = st.session_state.chatbot.ask(prompt)
            
            answer = response_data["result"]
            sources = response_data["source_documents"]
            latency = response_data["latency"]
            
            st.markdown(answer)
            st.caption(f"⏱️ {latency}s | Model: {response_data['model']}")
            
            if sources:
                with st.expander("📚 Sources"):
                    for doc in sources:
                        st.markdown(f"**{doc.metadata.get('source', 'Unknown')}** (Line {doc.metadata.get('line_start')}-{doc.metadata.get('line_end')})")
                        st.code(doc.page_content, language="text")
            
            # Save history
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "citations": sources
            })
