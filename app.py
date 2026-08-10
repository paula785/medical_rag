import os
import time
import requests
import streamlit as st
from PIL import Image
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Configuration
API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")

# Page Config
st.set_page_config(
    page_title="MedRAG - Cloud Multimodal Medical RAG",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Modern Premium UI
st.markdown("""
<style>
    /* Theme Colors & Typography */
    :root {
        --primary-color: #0284c7;
        --primary-hover: #0369a1;
        --bg-glass: rgba(15, 23, 42, 0.65);
        --border-color: rgba(56, 189, 248, 0.2);
    }
    
    .main-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0369a1 100%);
        padding: 24px;
        border-radius: 16px;
        color: white;
        margin-bottom: 24px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3);
    }
    
    .main-header h1 {
        color: #f8fafc !important;
        margin: 0;
        font-weight: 800;
        font-size: 2.2rem;
        display: flex;
        align-items: center;
        gap: 12px;
    }

    .main-header p {
        color: #94a3b8;
        margin-top: 6px;
        margin-bottom: 0;
        font-size: 1.05rem;
    }

    .source-box {
        background-color: rgba(15, 23, 42, 0.6);
        border-left: 4px solid #0284c7;
        padding: 12px 16px;
        border-radius: 6px;
        margin-bottom: 10px;
        font-size: 0.92rem;
    }
    
    .badge-tag {
        background: #0284c7;
        color: white;
        padding: 3px 8px;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
    }

    .stTabs [data-baseweb="tab"] {
        height: 48px;
        white-space: pre;
        background-color: rgba(30, 41, 59, 0.5);
        border-radius: 10px 10px 0px 0px;
        border: 1px solid rgba(255, 255, 255, 0.05);
        padding: 10px 20px;
    }

    .stTabs [aria-selected="true"] {
        background-color: #0284c7 !important;
        color: white !important;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# Helper Functions
@st.cache_data(ttl=3)
def fetch_backend_health(gemini_key: str = ""):
    try:
        r = requests.get(f"{API_BASE_URL}/health", params={"gemini_key": gemini_key}, timeout=2.5)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return {"status": "offline", "gemini_active": False, "user_documents_count": 0}

# Sidebar Setup
with st.sidebar:
    st.image("https://img.icons8.com/isometric/100/hospital-folder.png", width=64)
    st.title("MedRAG Settings")
    
    st.subheader("🔑 Gemini API Configuration")
    gemini_api_key = os.getenv("GEMINI_API_KEY", "")
    input_gemini = st.text_input("Gemini API Key", value=gemini_api_key, type="password", help="Used for Cloud OCR (Gemini 1.5 Flash) & Embeddings (text-embedding-004)")
    if input_gemini:
        os.environ["GEMINI_API_KEY"] = input_gemini
        gemini_api_key = input_gemini

    st.subheader("🔑 Groq API Configuration")
    groq_api_key = os.getenv("GROQ_API_KEY", "")
    input_groq = st.text_input("Groq API Key", value=groq_api_key, type="password", help="Used for Cloud LLM Reasoning (Llama-3.3-70B)")
    if input_groq:
        os.environ["GROQ_API_KEY"] = input_groq
        groq_api_key = input_groq

    st.divider()
    
    st.subheader("⚡ Cloud API Status")
    health = fetch_backend_health(gemini_api_key)
    
    if health.get("status") == "online":
        st.caption("Backend Server: 🟢 **Connected (FastAPI)**")
        if health.get("gemini_active"):
            st.success("Gemini Cloud API: 🟢 **Active** (OCR & Embeddings)")
        else:
            st.warning("Gemini Cloud API: 🔴 Key Missing")
            
        if health.get("groq_active"):
            st.success("Groq Cloud API: 🟢 **Active** (Llama-3.3-70B)")
        else:
            st.warning("Groq Cloud API: 🔴 Key Missing")
    else:
        st.error("⚠️ FastAPI Backend Service Offline at http://127.0.0.1:8000. Start `python run.py`.")
        if st.button("🔄 Refresh Status"):
            st.cache_data.clear()
            st.rerun()

    st.text_input("Vision OCR Engine", value="gemini-2.5-flash (Cloud)", disabled=True)
    st.text_input("Embedding Model", value="gemini-embedding-001 (Cloud)", disabled=True)
    st.caption("💡 *Cloud API processing ensures ultra-fast OCR transcription and embedding generation in seconds.*")

    st.divider()
    
    st.subheader("📦 FAISS Vector Database Status")
    if st.button("Refresh Vector Store Stats"):
        st.cache_data.clear()
        st.rerun()
    
    st.metric("User Uploaded Documents", health.get("user_documents_count", 0))
        
    if st.button("🗑️ Clear User Uploads"):
        try:
            r = requests.delete(f"{API_BASE_URL}/user-documents", params={"gemini_key": gemini_api_key}, timeout=5.0)
            if r.status_code == 200:
                st.success("Cleared user uploads from FAISS database!")
                st.cache_data.clear()
                time.sleep(1)
                st.rerun()
        except Exception as err:
            st.error(f"Reset failed: {err}")

# Main Layout Header
st.markdown("""
<div class="main-header">
    <h1>🩺 MedRAG: Cloud-Powered Multimodal Assistant</h1>
    <p>High-Speed Cloud Processing: Gemini 1.5 Flash (OCR) + Gemini Embeddings + Groq Llama-3.3-70B Clinical Reasoning</p>
</div>
""", unsafe_allow_html=True)

# Main Application Tabs
tab1, tab2 = st.tabs(["📄 Document Ingestion (Cloud OCR & PDF)", "💬 Medical RAG Chatbot"])

# ---------------------------------------------------------
# TAB 1: DOCUMENT INGESTION (OCR & PDF)
# ---------------------------------------------------------
with tab1:
    st.subheader("Upload Clinical Document (Image or PDF)")
    st.caption("Extract clinical details using Gemini 1.5 Flash (for images) or PyPDF (for PDFs) and index into FAISS.")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Choose a medical file (Image or PDF)",
            type=["jpg", "jpeg", "png", "pdf"],
            help="Upload a clear scanned prescription, blood test lab report, clinical summary image, or PDF document."
        )
        
        doc_category = st.selectbox(
            "Document Category Tag",
            ["Prescription", "Lab Report", "Diagnostic Imaging Report", "Doctor Consultation Note", "General Record"]
        )
        
        if uploaded_file is not None:
            file_type = uploaded_file.name.split(".")[-1].lower()
            file_bytes = uploaded_file.getvalue()
            
            if file_type in ["jpg", "jpeg", "png"]:
                image = Image.open(uploaded_file)
                st.image(image, caption=f"Uploaded Image: {uploaded_file.name}", use_container_width=True)
            else:
                st.info(f"📄 Uploaded PDF: **{uploaded_file.name}** ({round(len(file_bytes)/1024, 2)} KB)")
            
            process_btn = st.button("🚀 Process & Index Document", type="primary", use_container_width=True)

    with col2:
        if uploaded_file is not None and 'process_btn' in locals() and process_btn:
            if not gemini_api_key:
                st.error("⚠️ Missing Gemini API Key! Please enter your key in the sidebar.")
            else:
                with st.spinner("Processing document via Gemini 1.5 Flash & Google Embeddings..."):
                    try:
                        files = {"file": (uploaded_file.name, file_bytes, uploaded_file.type)}
                        data = {"doc_type": doc_category, "gemini_api_key": gemini_api_key}
                        res = requests.post(f"{API_BASE_URL}/ingest", files=files, data=data, timeout=60.0)
                        
                        if res.status_code == 200:
                            result = res.json()
                            st.success("✅ Document successfully indexed into FAISS vector store!")
                            st.markdown("### 📋 Extracted Clinical Details")
                            st.text_area("Document Content Result", result.get("extracted_text", ""), height=320)
                            st.cache_data.clear()
                        else:
                            st.error(f"Ingestion Error ({res.status_code}): {res.text}")
                    except Exception as err:
                        st.error(f"Backend Request Failed: {err}")

# ---------------------------------------------------------
# TAB 2: MEDICAL RAG CHATBOT (GROQ LLAMA-3.3-70B)
# ---------------------------------------------------------
with tab2:
    st.subheader("Ask Patient Clinical Assistant")
    st.caption("Queries FAISS vector database and synthesizes answers using Groq Llama-3.3-70B with expert medical persona.")
    
    # Initialize Chat History
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "Hello! I am your MedRAG Clinical Assistant. You can ask about patient history, baseline lab trends, active prescriptions, or uploaded medical records. How can I assist you today?"
            }
        ]
        
    # Render Chat History
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "sources" in message and message["sources"]:
                with st.expander("📚 View Retrieved User Source Citations"):
                    for idx, src in enumerate(message["sources"], 1):
                        st.markdown(f"""
                        <div class="source-box">
                            <strong>Source #{idx}: {src.get('source', 'Unknown')}</strong> 
                            <span class="badge-tag">{src.get('doc_type', 'Record')}</span><br/>
                            <small><em>Date: {src.get('date', 'N/A')}</em></small><br/>
                            <p style="margin-top:6px; font-size:0.9rem;">{src.get('snippet', '')}</p>
                        </div>
                        """, unsafe_allow_html=True)

    # Chat Input Box
    user_query = st.chat_input("Ask a clinical question (e.g., 'What are the patient's baseline lab values and drug allergies?')")
    
    if user_query:
        # Display user message
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        # Query FastAPI RAG Chat Endpoint
        with st.chat_message("assistant"):
            if not groq_api_key:
                st.error("⚠️ Missing Groq API Key! Please enter your key in the sidebar.")
            elif not gemini_api_key:
                st.error("⚠️ Missing Gemini API Key! Please enter your key in the sidebar.")
            else:
                with st.spinner("Analyzing patient records with Groq Llama-3.3-70B..."):
                    try:
                        payload = {
                            "question": user_query,
                            "groq_api_key": groq_api_key,
                            "gemini_api_key": gemini_api_key
                        }
                        r = requests.post(f"{API_BASE_URL}/chat", json=payload, timeout=60.0)
                        
                        if r.status_code == 200:
                            data = r.json()
                            answer_text = data.get("answer", "")
                            user_sources = data.get("user_sources", [])
                            
                            st.markdown(answer_text)
                            
                            if user_sources:
                                with st.expander("📚 View Retrieved User Source Citations"):
                                    for idx, src in enumerate(user_sources, 1):
                                        st.markdown(f"""
                                        <div class="source-box">
                                            <strong>Source #{idx}: {src.get('source', 'Unknown')}</strong> 
                                            <span class="badge-tag">{src.get('doc_type', 'Record')}</span><br/>
                                            <small><em>Date: {src.get('date', 'N/A')}</em></small><br/>
                                            <p style="margin-top:6px; font-size:0.9rem;">{src.get('snippet', '')}</p>
                                        </div>
                                        """, unsafe_allow_html=True)
                                        
                            st.session_state.messages.append({
                                "role": "assistant",
                                "content": answer_text,
                                "sources": user_sources
                            })
                            st.rerun()
                        else:
                            st.error(f"Chat API Error ({r.status_code}): {r.text}")
                    except Exception as err:
                        st.error(f"Backend Request Failed: {err}")
