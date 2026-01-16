import streamlit as st
import os
import shutil
from utils import logger, compute_file_hash, clean_filename
from chat_logic import RAGChatbot
from ingestion import NetworkConfigParser

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

st.set_page_config(
    page_title="NOC Copilot Config Compare",
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
        ["llama3.2:3b", "phi3.5", "dr-ry/Foundation-Sec-8B-Instruct-Chat-Q8_0.gguf:latest", "llama3.1:8b","qwen2.5-coder:7b-instruct"],
        index=0
    )
    
    if selected_model != st.session_state.chatbot.model_name:
        st.session_state.chatbot.update_model(selected_model)
        st.toast(f"Switched to {selected_model}")

    st.markdown("---")
    st.markdown("---")
    st.markdown("### 📂 Ingestion")
    
    # Golden Config Uploader
    st.markdown("#### 1. Golden Config (Reference)")
    golden_file = st.file_uploader("Upload Golden Config", type=['txt', 'pdf', 'cfg', 'log'], key="golden_uploader")
    
    if golden_file:
        file_bytes = golden_file.getvalue()
        file_hash = compute_file_hash(file_bytes)
        
        # Simple local save mechanics
        upload_dir = "./uploads"
        os.makedirs(upload_dir, exist_ok=True)
        safe_name = clean_filename(golden_file.name)
        save_path = os.path.join(upload_dir, safe_name)
        
        # Check if already processed
        if f"processed_golden_{file_hash}" not in st.session_state:
            with open(save_path, "wb") as f:
                f.write(file_bytes)
            
            with st.spinner("Indexing Golden Config..."):
                success = st.session_state.chatbot.process_file(save_path, extra_metadata={"config_role": "golden"})
                if success:
                    st.session_state[f"processed_golden_{file_hash}"] = True
                    st.session_state["golden_name"] = golden_file.name
                    st.session_state["golden_filename_clean"] = safe_name # Store sanitized name
                    st.session_state["golden_hash"] = file_hash # Store Hash
                    st.success("✅ Golden Config Indexed!")
                else:
                    st.error("Failed to process file.")
        else:
            st.info("✅ Golden Config Ready")
            st.session_state["golden_name"] = golden_file.name
            st.session_state["golden_filename_clean"] = safe_name
            st.session_state["golden_hash"] = file_hash

    # Candidate Config Uploader
    st.markdown("#### 2. Candidate Config (Target)")
    candidate_file = st.file_uploader("Upload Candidate Config", type=['txt', 'pdf', 'cfg', 'log'], key="candidate_uploader")

    if candidate_file:
        file_bytes = candidate_file.getvalue()
        file_hash = compute_file_hash(file_bytes)
        
        # Simple local save mechanics
        upload_dir = "./uploads"
        os.makedirs(upload_dir, exist_ok=True)
        safe_name = clean_filename(candidate_file.name)
        save_path = os.path.join(upload_dir, safe_name)
        
        # Check if already processed
        if f"processed_candidate_{file_hash}" not in st.session_state:
            with open(save_path, "wb") as f:
                f.write(file_bytes)
            
            with st.spinner("Indexing Candidate Config..."):
                success = st.session_state.chatbot.process_file(save_path, extra_metadata={"config_role": "candidate"})
                if success:
                    st.session_state[f"processed_candidate_{file_hash}"] = True
                    st.session_state["candidate_name"] = candidate_file.name
                    st.session_state["candidate_filename_clean"] = safe_name # Store sanitized name
                    st.session_state["candidate_hash"] = file_hash # Store Hash
                    st.success("✅ Candidate Config Indexed!")
                else:
                    st.error("Failed to process file.")
        else:
            st.info("✅ Candidate Config Ready")
            st.session_state["candidate_name"] = candidate_file.name
            st.session_state["candidate_filename_clean"] = safe_name
            st.session_state["candidate_hash"] = file_hash

    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("⚖️ Deep Compare", disabled=not (golden_file and candidate_file), use_container_width=True):
            if "golden_name" in st.session_state and "candidate_name" in st.session_state:
                prompt = (
                    f"Compare the candidate configuration '{st.session_state['candidate_name']}' "
                    f"against the golden configuration '{st.session_state['golden_name']}'. "
                    "Provide a detailed analysis of differences in: "
                    "2. Interfaces "
                    "3. Routing Protocols "
                    "4. Security ACLs "
                    "5. QoS & Management. "
                    "Highlight missing or extra configurations in the candidate file."
                    "Highlight missing or extra configurations in the candidate file."
                )
                st.session_state.messages.append({"role": "user", "content": prompt})
                st.rerun()

    with col2:
        if st.button("⚡ Quick Diff", disabled=not (golden_file and candidate_file), use_container_width=True):
             if "golden_name" in st.session_state and "candidate_name" in st.session_state:
                prompt = (
                    f"Compare '{st.session_state['candidate_name']}' against '{st.session_state['golden_name']}'. "
                    "Produce a **Markdown Table** only. No summary text.\n"
                    "Columns: | Feature/Line | Golden Config | Candidate Status |\n"
                    "Rules for 'Candidate Status':\n"
                    "- ✅ MATCH\n"
                    "- ❌ MISSING\n"
                    "- ➕ EXTRA\n"
                    "- ⚠️ DIFF: <Show value>\n"
                    "Focus on VLANs, Interfaces, Routes, QoS, ACLs, and Management."
                )
                st.session_state.messages.append({"role": "user", "content": prompt})
                st.rerun()

    st.markdown("---")
    
    # -------------------------------------------------------------------------
    # RAG Inspector (Debug Tool)
    # -------------------------------------------------------------------------
    with st.expander("🔍 RAG Inspector"):
        st.caption("Inspect how files are chunked for RAG.")
        
        inspect_target = st.radio(
            "Select File to Inspect",
            ["Golden Config", "Candidate Config"],
            horizontal=True
        )
        
        if st.button("View Chunks"):
            target_file_name = None
            if inspect_target == "Golden Config":
                target_file_name = st.session_state.get("golden_filename_clean")
            else:
                target_file_name = st.session_state.get("candidate_filename_clean")
            
            if target_file_name:
                file_path = os.path.join("./uploads", target_file_name)
                if os.path.exists(file_path):
                    with st.spinner(f"Parsing {target_file_name}..."):
                        try:
                            # Re-read and re-parse on demand for inspection
                            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                                content = f.read()
                            
                            parser = NetworkConfigParser(content, target_file_name)
                            blocks = parser.parse()
                            
                            st.markdown(f"**Found {len(blocks)} chunks in `{target_file_name}`**")
                            
                            # Display chunks
                            for i, block in enumerate(blocks):
                                with st.expander(f"Chunk {i+1}: {block.header_type} (Lines {block.line_start}-{block.line_end})"):
                                    st.code(block.full_text, language="text")
                                    st.markdown(f"**Metadata:**")
                                    st.json({
                                        "parent_line": block.parent_line,
                                        "has_secret": block.has_secret,
                                        "size_chars": len(block.full_text)
                                    })
                        except Exception as e:
                            st.error(f"Error parsing file: {e}")
                else:
                    st.error(f"File not found: {file_path}")
            else:
                st.warning("No file uploaded for this role yet.")

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
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.rerun()

# Logic to Handle Response Generation
# Checks if the last message is from the user, implying we need to reply.
if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # Get the last user message text
            user_input = st.session_state.messages[-1]["content"]
            
            # Detect if it's a comparison query (simple heuristic based on our button prompts)
            is_comparison = "Compare" in user_input and "golden" in user_input.lower()
            
            if is_comparison:
                # Deterministic Check: Are the files identical?
                g_hash = st.session_state.get("golden_hash")
                c_hash = st.session_state.get("candidate_hash")
                
                if g_hash and c_hash and g_hash == c_hash:
                    # Short-circuit
                    response_data = {
                        "result": "### ✅ Identical Configurations\n\nThe Golden and Candidate configuration files are **digitally identical** (SHA256 Match). No differences found.",
                        "source_documents": [],
                        "model": "Deterministic Check",
                        "latency": 0.00
                    }
                else:
                    # Determine comparison mode based on prompt keywords
                    # Deep Compare asks for "detailed analysis"
                    # Quick Diff asks for "Markdown Table"
                    if "detailed analysis" in user_input.lower() or "provide a" in user_input.lower():
                        comparison_mode = "deep"
                    else:
                        comparison_mode = "quick"
                    
                    # Retrieve the clean filenames from session state (if available) to ensure we compare the RIGHT files
                    g_meta = st.session_state.get("golden_filename_clean")
                    c_meta = st.session_state.get("candidate_filename_clean")
                    
                    response_data = st.session_state.chatbot.compare_configs(
                        user_input, 
                        golden_filename=g_meta, 
                        candidate_filename=c_meta,
                        mode=comparison_mode
                    )
            else:
                response_data = st.session_state.chatbot.ask(user_input)
            
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
            
            # Rerun to update state properly
            st.rerun()
