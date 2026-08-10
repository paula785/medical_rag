# MedRAG System: Local-First Multimodal Medical RAG

A local-first Multimodal Medical Retrieval-Augmented Generation (RAG) web application powered by **Streamlit**, **LangChain**, **Ollama**, **ChromaDB**, and **Groq API**.

---

## 🌟 Key Features

1. **Local Vision OCR Ingestion (Ollama VLM)**
   - Upload scanned prescriptions, blood test lab reports, or diagnostic printouts (`.png`, `.jpg`).
   - Uses local Ollama Vision VLM (`gemma4:e2b`) to transcribe and summarize clinical details offline.

2. **Persistent Vector Database (ChromaDB & LangChain)**
   - Automatically embeds clinical records using `nomic-embed-text` embeddings via `OllamaEmbeddings`.
   - Stores documents in `./med_rag_db` with rich metadata (document type, source, timestamp).
   - Pre-seeds baseline synthetic patient history (allergies, diabetes history, lab trends, active prescriptions).

3. **Fast Expert Medical RAG Chatbot (Groq Llama-3.3-70B)**
   - High-speed inference using Groq's `llama-3.3-70b-versatile` model.
   - Expert Board-Certified Medical Assistant persona providing deep clinical insights, allergy contraindication checks, and structured medical advice.
   - Interactive source citation drawer displaying retrieved context chunks.

---

## 🛠️ System Architecture & Tech Stack

- **Frontend**: Streamlit
- **Framework**: LangChain (`langchain`, `langchain-groq`, `langchain-chroma`, `langchain-ollama`)
- **Local Vision VLM**: Ollama (`gemma4:e2b`)
- **Embedding Model**: Ollama (`nomic-embed-text`)
- **Vector Database**: ChromaDB (Persistent local storage at `./med_rag_db`)
- **Chat Engine**: Groq API (`llama-3.3-70b-versatile`)
- **Image Processing**: Pillow (PIL)

---

## 🚀 Quick Start Guide

### 1. Prerequisites
- Python 3.10+
- [Ollama Desktop / CLI](https://ollama.com/) installed and running.

### 2. Environment Setup
```powershell
# Create Python Virtual Environment
python -m venv venv

# Activate Virtual Environment
.\venv\Scripts\Activate.ps1

# Install Dependencies
pip install -r requirements.txt
```

### 3. Ollama Model Setup
Ensure the local Ollama service is running, then pull the required models:
```powershell
ollama pull gemma4:e2b
ollama pull nomic-embed-text
```

### 4. Groq API Key Configuration
Create a `.env` file in the project root:
```env
GROQ_API_KEY=your_groq_api_key_here
GEMINI_API_KEY=your_gemini_api_key_here
```

### 5. Run Application
Launch the MedRAG application orchestrator (auto-checks and starts Ollama daemon background service, then launches Streamlit):
```powershell
python start_app.py
```
Or launch Streamlit directly:
```powershell
streamlit run app.py
```
Open your browser at `http://localhost:8501`.
