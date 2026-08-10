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

# ─── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    :root {
        --primary: #0284c7;
        --primary-hover: #0369a1;
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
    .doc-card {
        background: rgba(15, 23, 42, 0.5);
        border: 1px solid rgba(56, 189, 248, 0.2);
        border-radius: 10px;
        padding: 14px 18px;
        margin-bottom: 12px;
    }

    .stTabs [data-baseweb="tab-list"] { gap: 12px; }
    .stTabs [data-baseweb="tab"] {
        height: 48px;
        background-color: rgba(30, 41, 59, 0.5);
        border-radius: 10px 10px 0 0;
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

# ─── Helper Functions ──────────────────────────────────────────────────────────
@st.cache_data(ttl=3)
def fetch_backend_health():
    try:
        r = requests.get(f"{API_BASE_URL}/health", timeout=2.5)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return {"status": "offline", "gemini_active": False, "groq_active": False, "user_documents_count": 0}

@st.cache_data(ttl=10)
def fetch_documents():
    try:
        r = requests.get(f"{API_BASE_URL}/documents", timeout=5.0)
        if r.status_code == 200:
            return r.json().get("documents", [])
    except Exception:
        pass
    return []

# ─── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/isometric/100/hospital-folder.png", width=64)
    st.title("MedRAG Settings")

    st.subheader("⚡ Cloud API Status")
    health = fetch_backend_health()

    if health.get("status") == "online":
        st.caption("Backend Server: 🟢 **Connected (FastAPI)**")
        if health.get("gemini_active"):
            st.success("Gemini Cloud API: 🟢 **Active** (OCR & Embeddings)")
        else:
            st.warning("Gemini Cloud API: 🔴 Key Missing — add to `.env`")

        if health.get("groq_active"):
            st.success("Groq Cloud API: 🟢 **Active** (Llama-3.3-70B)")
        else:
            st.warning("Groq Cloud API: 🔴 Key Missing — add to `.env`")
    else:
        st.error("⚠️ FastAPI Backend Offline. Run `python run.py`.")
        if st.button("🔄 Retry Connection"):
            st.cache_data.clear()
            st.rerun()

    st.text_input("Vision OCR Engine", value="gemini-2.5-flash (Cloud)", disabled=True)
    st.text_input("Embedding Model", value="gemini-embedding-001 (Cloud)", disabled=True)
    st.caption("💡 *Keys are loaded from your `.env` file automatically.*")

    st.divider()
    st.subheader("📦 FAISS Vector Database")
    if st.button("🔄 Refresh Stats"):
        st.cache_data.clear()
        st.rerun()

    st.metric("User Uploaded Documents", health.get("user_documents_count", 0))

    if st.button("🗑️ Clear ALL User Uploads"):
        try:
            r = requests.delete(f"{API_BASE_URL}/user-documents", timeout=10.0)
            if r.status_code == 200:
                st.success("All user uploads cleared. Baseline records kept.")
                st.cache_data.clear()
                time.sleep(1)
                st.rerun()
        except Exception as err:
            st.error(f"Reset failed: {err}")

# ─── Main Header ───────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🩺 MedRAG: Cloud-Powered Multimodal Assistant</h1>
    <p>Gemini 2.5 Flash (OCR) · Gemini Embeddings · Groq Llama-3.3-70B Clinical Reasoning</p>
</div>
""", unsafe_allow_html=True)

# ─── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "📄 Document Ingestion",
    "💬 Medical RAG Chatbot",
    "📋 Document Manager"
])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1: DOCUMENT INGESTION
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Upload Clinical Document (Image or PDF)")
    st.caption("Images → Gemini 2.5 Flash Cloud OCR · PDFs → PyPDF text extraction · Both indexed into FAISS.")

    col1, col2 = st.columns([1, 1])

    with col1:
        uploaded_file = st.file_uploader(
            "Choose a medical file (Image or PDF)",
            type=["jpg", "jpeg", "png", "pdf"],
            help="Upload a scanned prescription, lab report, clinical summary, or PDF document (max 10 MB)."
        )
        doc_category = st.selectbox(
            "Document Category Tag",
            ["Prescription", "Lab Report", "Diagnostic Imaging Report", "Doctor Consultation Note", "General Record"]
        )

        if uploaded_file is not None:
            file_type = uploaded_file.name.rsplit(".", 1)[-1].lower()
            file_bytes = uploaded_file.getvalue()

            if file_type in ["jpg", "jpeg", "png"]:
                image = Image.open(uploaded_file)
                st.image(image, caption=f"📷 {uploaded_file.name}", use_container_width=True)
            else:
                st.info(f"📄 PDF: **{uploaded_file.name}** ({round(len(file_bytes)/1024, 2)} KB)")

            # File size warning
            if len(file_bytes) > 10 * 1024 * 1024:
                st.error("⚠️ File exceeds 10 MB limit. Please upload a smaller file.")
            else:
                process_btn = st.button("🚀 Process & Index Document", type="primary", use_container_width=True)

    with col2:
        if uploaded_file is not None and 'process_btn' in locals() and process_btn:
            if not health.get("gemini_active"):
                st.error("⚠️ Gemini API Key not configured. Add GEMINI_API_KEY to your `.env` file.")
            else:
                with st.spinner("Processing with Gemini 2.5 Flash & Google Embeddings..."):
                    try:
                        files = {"file": (uploaded_file.name, file_bytes, uploaded_file.type)}
                        data = {"doc_type": doc_category}
                        res = requests.post(f"{API_BASE_URL}/ingest", files=files, data=data, timeout=90.0)

                        if res.status_code == 200:
                            result = res.json()
                            if result.get("status") == "duplicate":
                                st.warning(f"⚠️ **Duplicate detected**: {result.get('message', '')}")
                            else:
                                st.success("✅ Document successfully indexed into FAISS vector store!")
                            st.markdown("### 📋 Extracted Clinical Details")
                            st.text_area("Document Content", result.get("extracted_text", ""), height=360)
                            st.cache_data.clear()
                        else:
                            st.error(f"Ingestion Error ({res.status_code}): {res.text}")
                    except Exception as err:
                        st.error(f"Backend Request Failed: {err}")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2: MEDICAL RAG CHATBOT
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Ask Patient Clinical Assistant")
    st.caption("Queries FAISS with semantic search · Answers via Groq Llama-3.3-70B with full conversation memory.")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "Hello! I am your MedRAG Clinical Assistant. You can ask about patient history, baseline lab trends, active prescriptions, or uploaded medical records. How can I assist you today?"
            }
        ]

    # Render existing chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("sources"):
                with st.expander("📚 View Retrieved Source Citations"):
                    for idx, src in enumerate(message["sources"], 1):
                        st.markdown(f"""
                        <div class="source-box">
                            <strong>Source #{idx}: {src.get('source', 'Unknown')}</strong>
                            <span class="badge-tag">{src.get('doc_type', 'Record')}</span><br/>
                            <small><em>Date: {src.get('date', 'N/A')}</em></small><br/>
                            <p style="margin-top:6px; font-size:0.9rem;">{src.get('snippet', '')}</p>
                        </div>
                        """, unsafe_allow_html=True)

    # Chat input
    user_query = st.chat_input("Ask a clinical question (e.g., 'What are the patient's drug allergies?')")

    if user_query:
        if not health.get("groq_active"):
            st.error("⚠️ Groq API Key not configured. Add GROQ_API_KEY to your `.env` file.")
        elif not health.get("gemini_active"):
            st.error("⚠️ Gemini API Key not configured. Add GEMINI_API_KEY to your `.env` file.")
        else:
            st.session_state.messages.append({"role": "user", "content": user_query})
            with st.chat_message("user"):
                st.markdown(user_query)

            with st.chat_message("assistant"):
                # Build chat history payload (last 6 messages excluding the one just added)
                history_payload = [
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.messages[:-1][-6:]
                ]

                # Streaming via SSE
                full_response = ""
                placeholder = st.empty()
                try:
                    payload = {
                        "question": user_query,
                        "chat_history": history_payload
                    }
                    with requests.post(
                        f"{API_BASE_URL}/chat/stream",
                        json=payload,
                        stream=True,
                        timeout=90.0
                    ) as r:
                        if r.status_code == 200:
                            for line in r.iter_lines():
                                if line:
                                    decoded = line.decode("utf-8")
                                    if decoded.startswith("data: "):
                                        token = decoded[6:]
                                        if token == "[DONE]":
                                            break
                                        if token.startswith("[ERROR]"):
                                            st.error(token)
                                            break
                                        full_response += token
                                        placeholder.markdown(full_response + "▌")
                            placeholder.markdown(full_response)
                        else:
                            st.error(f"Chat API Error ({r.status_code}): {r.text}")

                    # Fetch sources from non-streaming endpoint (light metadata call)
                    user_sources = []
                    try:
                        meta_payload = {"question": user_query, "chat_history": history_payload}
                        meta_r = requests.post(f"{API_BASE_URL}/chat", json=meta_payload, timeout=60.0)
                        if meta_r.status_code == 200:
                            user_sources = meta_r.json().get("user_sources", [])
                    except Exception:
                        pass

                    if user_sources:
                        with st.expander("📚 View Retrieved Source Citations"):
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
                        "content": full_response,
                        "sources": user_sources
                    })
                    st.rerun()

                except Exception as err:
                    st.error(f"Backend Request Failed: {err}")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3: DOCUMENT MANAGER
# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("📋 User Uploaded Documents")
    st.caption("View and manage all documents you have indexed into the FAISS vector store.")

    col_refresh, col_clear = st.columns([1, 1])
    with col_refresh:
        if st.button("🔄 Refresh List", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
    with col_clear:
        if st.button("🗑️ Clear All User Documents", type="secondary", use_container_width=True):
            try:
                r = requests.delete(f"{API_BASE_URL}/user-documents", timeout=10.0)
                if r.status_code == 200:
                    st.success("All user documents cleared.")
                    st.cache_data.clear()
                    time.sleep(0.5)
                    st.rerun()
            except Exception as e:
                st.error(f"Failed to clear: {e}")

    st.divider()
    documents = fetch_documents()

    if not documents:
        st.info("📭 No user documents indexed yet. Upload files in the **Document Ingestion** tab.")
    else:
        st.markdown(f"**{len(documents)} document(s) in your knowledge base:**")
        for doc in documents:
            with st.container():
                col_info, col_del = st.columns([5, 1])
                with col_info:
                    st.markdown(f"""
                    <div class="doc-card">
                        <strong>📄 {doc.get('source', 'Unknown')}</strong>
                        &nbsp;<span class="badge-tag">{doc.get('doc_type', 'Record')}</span>
                        &nbsp;<span class="badge-tag" style="background:#475569;">{doc.get('file_type','?').upper()}</span><br/>
                        <small style="color:#94a3b8;">🗓️ {doc.get('date','N/A')}</small><br/>
                        <p style="margin-top:8px; font-size:0.88rem; color:#cbd5e1;">{doc.get('snippet','')[:180]}…</p>
                    </div>
                    """, unsafe_allow_html=True)
                with col_del:
                    st.write("")
                    st.write("")
                    if st.button("🗑️", key=f"del_{doc['id']}", help=f"Delete {doc.get('source','this document')}"):
                        try:
                            r = requests.delete(f"{API_BASE_URL}/documents/{doc['id']}", timeout=10.0)
                            if r.status_code == 200:
                                st.success(f"Deleted **{doc.get('source', doc['id'])}**")
                                st.cache_data.clear()
                                time.sleep(0.5)
                                st.rerun()
                            else:
                                st.error(f"Delete failed: {r.text}")
                        except Exception as e:
                            st.error(f"Error: {e}")
