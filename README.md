# 🩺 MedRAG: Cloud-Powered Multimodal Medical RAG System

A cloud-powered **Multimodal Medical Retrieval-Augmented Generation (RAG)** web application for clinical document analysis and intelligent patient Q&A.

Built with **FastAPI** (backend), **Streamlit** (frontend), **FAISS** (vector search), **Google Gemini** (OCR & embeddings), and **Groq Llama-3.3-70B** (clinical reasoning).

---

## 🌟 Key Features

### 📄 Cloud Vision OCR & PDF Ingestion
- Upload scanned prescriptions, lab reports, or clinical documents (`.png`, `.jpg`, `.jpeg`, `.pdf`).
- **Images** are processed with **Gemini 2.5 Flash** Vision OCR — extracts text, medications, dosages, lab values, and clinical notes in seconds.
- **PDFs** are parsed with **PyPDF** for fast text extraction.
- All extracted content is automatically embedded and indexed into the FAISS vector store.

### 🧠 Intelligent Medical RAG Chatbot
- Ask natural-language clinical questions about patient records.
- Retrieves relevant context from the FAISS vector database using **Gemini Embedding** semantic search.
- Synthesizes expert-level answers using **Groq Llama-3.3-70B** with a board-certified clinical AI persona.
- Includes allergy contraindication checks, lab correlation analysis, and structured medical guidance.
- Interactive source citation drawer showing which documents informed each answer.

### 🗄️ Baseline Patient Knowledge
- Pre-seeded with synthetic patient history (demographics, allergies, lab results, active medications).
- Baseline records are **unchangeable system defaults** — they persist even when user uploads are cleared.
- Only user-uploaded documents appear in the UI counters and can be deleted.

### ⚡ Decoupled Architecture
- **FastAPI** backend microservice handles all AI processing (OCR, embeddings, RAG chat).
- **Streamlit** frontend communicates via clean REST API calls.
- Unified launcher (`run.py`) starts both services with a single command.

---

## 🛠️ Tech Stack

| Component           | Technology                                          |
|---------------------|-----------------------------------------------------|
| **Frontend**        | Streamlit                                           |
| **Backend API**     | FastAPI + Uvicorn                                   |
| **Vision OCR**      | Google Gemini 2.5 Flash (Cloud)                     |
| **Embeddings**      | Google Gemini Embedding (`gemini-embedding-001`, 3072-dim) |
| **Vector Database** | FAISS (local, CPU-based)                            |
| **Chat / Reasoning**| Groq API (`llama-3.3-70b-versatile`)                |
| **PDF Parsing**     | PyPDF                                               |
| **Framework**       | LangChain (`langchain-groq`, `langchain-google-genai`) |

---

## 📁 Project Structure

```
medical_rag/
├── server.py          # FastAPI backend — OCR, embeddings, FAISS, RAG chat endpoints
├── app.py             # Streamlit frontend — UI, file upload, chat interface
├── run.py             # Unified launcher — starts FastAPI + Streamlit together
├── start_app.py       # Legacy launcher (Ollama-based, kept for reference)
├── requirements.txt   # Python dependencies
├── .env.example       # Template for API keys
├── .gitignore         # Git exclusions (secrets, venv, caches)
└── README.md          # This file
```

---

## 🚀 Quick Start

### 1. Prerequisites
- **Python 3.10+**
- A [Google AI Studio](https://aistudio.google.com/apikey) API key (for Gemini OCR & Embeddings)
- A [Groq Console](https://console.groq.com/keys) API key (for Llama-3.3-70B reasoning)

### 2. Clone & Setup
```bash
git clone https://github.com/paula785/medical_rag.git
cd medical_rag

# Create and activate virtual environment
python -m venv venv

# Windows
.\venv\Scripts\Activate.ps1

# macOS / Linux
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Configure API Keys
Copy the example and fill in your keys:
```bash
cp .env.example .env
```

Edit `.env`:
```env
GEMINI_API_KEY=your_gemini_api_key_here
GROQ_API_KEY=your_groq_api_key_here
```

> **Note**: Keys can also be entered directly in the Streamlit sidebar at runtime.

### 4. Run the Application
```bash
python run.py
```

This starts:
- **FastAPI backend** on `http://127.0.0.1:8000`
- **Streamlit frontend** on `http://localhost:8501` (opens automatically)

---

## 🔌 API Endpoints

| Method   | Endpoint            | Description                                    |
|----------|---------------------|------------------------------------------------|
| `GET`    | `/health`           | Backend status, API key validation, doc count  |
| `POST`   | `/ingest`           | Upload & process a document (image or PDF)     |
| `POST`   | `/chat`             | Ask a clinical question against patient records|
| `DELETE` | `/user-documents`   | Clear user-uploaded documents (keeps baseline) |

---

## 📝 Usage

1. **Upload Documents** — Go to the "Document Ingestion" tab, upload a medical image or PDF, select a category tag, and click "Process & Index Document".
2. **Ask Questions** — Switch to the "Medical RAG Chatbot" tab and type clinical questions like:
   - *"What are the patient's known drug allergies?"*
   - *"Summarize the latest lab results and flag any abnormal values."*
   - *"Is the current medication regimen safe given the patient's renal function?"*
3. **View Sources** — Expand the citation drawer under each answer to see which documents were retrieved.

---

## ⚠️ Disclaimer

This application is a **demonstration and educational tool only**. It is **not** a certified medical device and must **not** be used for real clinical decision-making. Always consult qualified healthcare professionals for medical advice.

---

## 📄 License

This project is for educational and demonstration purposes.
