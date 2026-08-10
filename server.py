"""
MedRAG FastAPI Backend Microservice (Cloud-Powered)
---------------------------------------------------
Decoupled REST API backend using Cloud APIs:
- Google Gemini 2.5 Flash for ultra-fast Vision OCR
- Google Gemini Embedding (gemini-embedding-001, 3072-dim) for embeddings
- Groq Llama-3.3-70B for expert clinical RAG reasoning
- FAISS vector store with in-memory singleton cache
"""

import os
import base64
import hashlib
import logging
from io import BytesIO
from datetime import datetime
from typing import AsyncGenerator

from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from pypdf import PdfReader

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

load_dotenv()

# ─── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("medrag")

# ─── App & Middleware ──────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Modern lifespan handler: runs startup logic, then yields for shutdown."""
    logger.info("MedRAG v4.0 starting — Gemini OCR: %s | Embed: %s | Groq: %s",
                GEMINI_OCR_MODEL, GEMINI_EMBED_MODEL, GROQ_MODEL_DEFAULT)
    if GEMINI_API_KEY:
        try:
            get_faiss_store()
        except Exception as e:
            logger.warning("FAISS warm-up skipped: %s", e)
    else:
        logger.warning("GEMINI_API_KEY not set — FAISS will initialize on first request.")
    yield  # app runs here
    logger.info("MedRAG shutting down.")

app = FastAPI(title="MedRAG Cloud Backend Microservice", version="4.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8501",
        "http://127.0.0.1:8501",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Constants ─────────────────────────────────────────────────────────────────
FAISS_DB_DIR = "./med_rag_faiss"
GROQ_MODEL_DEFAULT = "llama-3.3-70b-versatile"
GEMINI_EMBED_MODEL = "models/gemini-embedding-001"
GEMINI_OCR_MODEL = "gemini-2.5-flash"
MAX_FILE_SIZE = 10 * 1024 * 1024   # 10 MB
MAX_QUESTION_LEN = 2000             # characters

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

# ─── Baseline Patient Records (immutable system defaults) ─────────────────────
BASELINE_PATIENT_RECORDS = [
    Document(
        page_content="""PATIENT DEMOGRAPHICS & CLINICAL HISTORY:
Patient Name: John Doe
DOB: 1978-05-14 (Age: 48)
Primary Conditions:
1. Type 2 Diabetes Mellitus (Diagnosed 2021) - Managed with oral hypoglycemics.
2. Essential Hypertension (Diagnosed 2019) - Moderate control.
3. Mild Chronic Kidney Disease Stage 2 (Baseline eGFR 85 mL/min/1.73m²).
Family History: Coronary Artery Disease (Father), Type 2 Diabetes (Mother).""",
        metadata={"source": "baseline_medical_history.pdf", "doc_type": "Patient History", "date": "2026-01-10", "is_baseline": True}
    ),
    Document(
        page_content="""ALLERGY & ADVERSE REACTION RECORD:
Severe Allergy: Penicillin / Amoxicillin - Causes Anaphylaxis & Severe Bronchospasm.
Moderate Allergy: Sulfonamide Antibiotics (Bactrim/Septra) - Causes Generalized Maculopapular Rash and Urticaria.
No Known Food or Latex Allergies.""",
        metadata={"source": "allergies_record.pdf", "doc_type": "Allergy Alert", "date": "2026-01-10", "is_baseline": True}
    ),
    Document(
        page_content="""BASELINE LAB RESULTS & RECENT VITAL SIGNS (Date: 2026-02-01):
Blood Pressure: 138/86 mmHg, Pulse: 72 bpm, BMI: 28.4 kg/m²
HbA1c: 7.4% (Target < 7.0%) - Suboptimal glycemic control.
Fasting Blood Glucose: 142 mg/dL.
Renal Function: Serum Creatinine 1.0 mg/dL, BUN 18 mg/dL, eGFR 85 mL/min/1.73m².
Lipid Panel: Total Cholesterol 210 mg/dL, Triglycerides 180 mg/dL, HDL 42 mg/dL, LDL 132 mg/dL (Elevated).""",
        metadata={"source": "lab_results_feb2026.pdf", "doc_type": "Lab Report", "date": "2026-02-01", "is_baseline": True}
    ),
    Document(
        page_content="""CURRENT ACTIVE MEDICATION REGIMEN:
1. Metformin Hydrochloride 1000 mg Oral Tablet - Take 1 tablet twice daily with meals.
2. Lisinopril 10 mg Oral Tablet - Take 1 tablet once daily in the morning for hypertension & renal protection.
3. Atorvastatin 20 mg Oral Tablet - Take 1 tablet daily at bedtime for hyperlipidemia.""",
        metadata={"source": "active_medications.pdf", "doc_type": "Prescription List", "date": "2026-02-01", "is_baseline": True}
    )
]

# ─── FAISS In-Memory Singleton Cache ──────────────────────────────────────────
_faiss_store = None

def get_embeddings(key: str = None) -> GoogleGenerativeAIEmbeddings:
    api_key = key or GEMINI_API_KEY
    if not api_key:
        raise HTTPException(status_code=400, detail="Missing Gemini API Key. Set GEMINI_API_KEY in your .env file.")
    return GoogleGenerativeAIEmbeddings(model=GEMINI_EMBED_MODEL, google_api_key=api_key)

def get_faiss_store(force_reload: bool = False) -> FAISS:
    """Return the in-memory FAISS store, loading or initializing it only once."""
    global _faiss_store
    if _faiss_store is not None and not force_reload:
        return _faiss_store

    embeddings = get_embeddings()
    index_path = os.path.join(FAISS_DB_DIR, "index.faiss")

    if os.path.exists(FAISS_DB_DIR) and os.path.exists(index_path):
        try:
            logger.info("Loading FAISS index from disk: %s", FAISS_DB_DIR)
            _faiss_store = FAISS.load_local(FAISS_DB_DIR, embeddings, allow_dangerous_deserialization=True)
            logger.info("FAISS loaded — %d vectors in store.", _faiss_store.index.ntotal)
            return _faiss_store
        except Exception as e:
            logger.warning("FAISS load failed (%s). Reinitializing from baseline records.", e)

    logger.info("Initializing fresh FAISS index from %d baseline records.", len(BASELINE_PATIENT_RECORDS))
    _faiss_store = FAISS.from_documents(BASELINE_PATIENT_RECORDS, embeddings)
    _faiss_store.save_local(FAISS_DB_DIR)
    logger.info("FAISS initialized and saved to disk.")
    return _faiss_store



# ─── Request / Response Models ─────────────────────────────────────────────────
class ChatMessage(BaseModel):
    role: str   # "user" or "assistant"
    content: str

class ChatRequest(BaseModel):
    question: str = Field(..., max_length=MAX_QUESTION_LEN)
    chat_history: list[ChatMessage] = Field(default=[], max_length=10)

# ─── Helper: Build Clinical Prompt ────────────────────────────────────────────
SYSTEM_PROMPT = """You are an elite, board-certified Clinical AI Specialist and Pharmacotherapy Consultant.
Your objective is to provide high-precision, clinically accurate, and safe answers to medical inquiries using the patient's retrieved medical records.

RETRIEVED PATIENT CONTEXT:
{context}

CLINICAL GUIDELINES FOR YOUR RESPONSE:
1. Ground your analysis strictly in the provided patient records and established medical knowledge.
2. If discussing medications, ALWAYS check against patient allergies (e.g. Penicillin anaphylaxis, Sulfa allergies) and point out any potential contraindications.
3. Structure your response clearly using markdown sections:
   - **Executive Summary & Observations**
   - **Detailed Medical Analysis / Lab & Drug Correlation**
   - **Actionable Patient Guidance**
   - **Important Medical Disclaimer**
4. Maintain an empathetic, professional, and rigorous clinical tone."""

def build_llm_chain(context: str):
    groq_key = GROQ_API_KEY
    if not groq_key:
        raise HTTPException(status_code=400, detail="Missing Groq API Key. Set GROQ_API_KEY in your .env file.")

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
    ])
    llm = ChatGroq(model_name=GROQ_MODEL_DEFAULT, groq_api_key=groq_key, temperature=0.2)
    return prompt | llm, context

def convert_history(chat_history: list[ChatMessage]) -> list:
    """Convert pydantic ChatMessage list to LangChain message objects."""
    messages = []
    for msg in chat_history:
        if msg.role == "user":
            messages.append(HumanMessage(content=msg.content))
        elif msg.role == "assistant":
            messages.append(AIMessage(content=msg.content))
    return messages

# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health")
async def health_check():
    gemini_configured = bool(GEMINI_API_KEY)
    groq_configured = bool(GROQ_API_KEY)

    faiss_count = 0
    if gemini_configured:
        try:
            vs = get_faiss_store()
            for _, doc in vs.docstore._dict.items():
                if not doc.metadata.get("is_baseline", False):
                    faiss_count += 1
        except Exception as e:
            logger.error("Health check FAISS error: %s", e)

    return {
        "status": "online",
        "gemini_active": gemini_configured,
        "groq_active": groq_configured,
        "user_documents_count": faiss_count,
        "models": {
            "ocr": GEMINI_OCR_MODEL,
            "embed": GEMINI_EMBED_MODEL,
            "chat": GROQ_MODEL_DEFAULT,
        }
    }


@app.post("/ingest")
async def ingest_document(
    file: UploadFile = File(...),
    doc_type: str = Form(...),
):
    if not GEMINI_API_KEY:
        raise HTTPException(status_code=400, detail="Gemini API Key not configured. Set GEMINI_API_KEY in .env.")

    contents = await file.read()
    filename = file.filename
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else "bin"

    # ── Validation: file size ──
    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail=f"File exceeds 10 MB limit ({len(contents) // 1024 // 1024} MB uploaded).")

    logger.info("Ingesting '%s' (%d KB, type=%s)", filename, len(contents) // 1024, doc_type)

    # ── Deduplication via SHA-256 ──
    content_hash = hashlib.sha256(contents).hexdigest()
    vs = get_faiss_store()
    for _, doc in vs.docstore._dict.items():
        if doc.metadata.get("content_hash") == content_hash:
            logger.info("Duplicate detected for '%s' — skipping ingestion.", filename)
            return {"status": "duplicate", "filename": filename, "message": "This document has already been indexed.", "extracted_text": doc.page_content}

    # ── Extraction ──
    extracted_text = ""
    if ext == "pdf":
        try:
            reader = PdfReader(BytesIO(contents))
            pages = [f"--- PAGE {i+1} ---\n{p.extract_text().strip()}" for i, p in enumerate(reader.pages) if p.extract_text()]
            extracted_text = "\n\n".join(pages) if pages else "No text extracted from PDF."
            logger.info("PDF extracted: %d pages, %d chars.", len(reader.pages), len(extracted_text))
        except Exception as e:
            logger.error("PDF extraction failed for '%s': %s", filename, e)
            raise HTTPException(status_code=400, detail=f"PDF extraction error: {e}")
    else:
        try:
            img_b64 = base64.b64encode(contents).decode("utf-8")
            mime_type = file.content_type or f"image/{ext}"
            ocr_prompt = """You are an expert clinical document OCR specialist and medical transcription assistant.
Examine this medical image carefully. Extract all visible text, doctor details, dates, medications, dosages, lab parameters, and clinical values accurately.
Summarize key findings in structured headings: DOCUMENT TYPE & DATE, PATIENT & PROVIDER INFO, LAB PARAMETERS & TEST RESULTS, MEDICATION PRESCRIPTIONS, CLINICAL NOTES."""

            llm = ChatGoogleGenerativeAI(model=GEMINI_OCR_MODEL, google_api_key=GEMINI_API_KEY, temperature=0.1)
            message = HumanMessage(content=[
                {"type": "text", "text": ocr_prompt},
                {"type": "image_url", "image_url": f"data:{mime_type};base64,{img_b64}"}
            ])
            res = llm.invoke([message])
            extracted_text = res.content
            logger.info("Gemini OCR complete for '%s': %d chars extracted.", filename, len(extracted_text))
        except Exception as e:
            logger.error("Gemini OCR failed for '%s': %s", filename, e)
            raise HTTPException(status_code=500, detail=f"Gemini Cloud OCR error: {e}")

    # ── Embed & Index into FAISS ──
    try:
        doc = Document(
            page_content=extracted_text,
            metadata={
                "source": filename,
                "doc_type": doc_type,
                "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "is_baseline": False,
                "file_type": ext,
                "content_hash": content_hash,
            }
        )
        vs.add_documents([doc])
        vs.save_local(FAISS_DB_DIR)
        logger.info("Document '%s' indexed into FAISS. Total vectors: %d", filename, vs.index.ntotal)
    except Exception as err:
        logger.error("FAISS embedding error for '%s': %s", filename, err)
        raise HTTPException(status_code=500, detail=f"FAISS embedding error: {err}")

    return {"status": "success", "filename": filename, "extracted_text": extracted_text}


@app.post("/chat")
async def chat_medical_rag(req: ChatRequest):
    if len(req.question) > MAX_QUESTION_LEN:
        raise HTTPException(status_code=400, detail=f"Question exceeds {MAX_QUESTION_LEN} character limit.")

    logger.info("Chat query: '%s...' | history_len=%d", req.question[:60], len(req.chat_history))

    vs = get_faiss_store()
    retriever = vs.as_retriever(search_kwargs={"k": 4})
    retrieved_docs = retriever.invoke(req.question)
    context_str = "\n\n---\n\n".join([doc.page_content for doc in retrieved_docs])

    history = convert_history(req.chat_history[-6:])  # last 6 messages max

    # Guard: validate Groq key before entering try block
    if not GROQ_API_KEY:
        raise HTTPException(status_code=400, detail="Missing Groq API Key. Set GROQ_API_KEY in .env.")

    try:
        chain, _ = build_llm_chain(context_str)
        response = chain.invoke({"context": context_str, "question": req.question, "history": history})

        user_sources = [
            {
                "source": d.metadata.get("source", "Unknown"),
                "doc_type": d.metadata.get("doc_type", "Record"),
                "date": d.metadata.get("date", "N/A"),
                "snippet": d.page_content[:300] + "..."
            }
            for d in retrieved_docs if not d.metadata.get("is_baseline", False)
        ]

        logger.info("Chat response generated: %d chars | %d user sources.", len(response.content), len(user_sources))
        return {"answer": response.content, "user_sources": user_sources}
    except Exception as e:
        logger.error("Groq chat error: %s", e)
        raise HTTPException(status_code=500, detail=f"Groq API Error: {str(e)}")


@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    """Streaming chat using Server-Sent Events (SSE). Yields tokens as they arrive."""
    if len(req.question) > MAX_QUESTION_LEN:
        raise HTTPException(status_code=400, detail=f"Question exceeds {MAX_QUESTION_LEN} character limit.")

    vs = get_faiss_store()
    retrieved_docs = vs.as_retriever(search_kwargs={"k": 4}).invoke(req.question)
    context_str = "\n\n---\n\n".join([doc.page_content for doc in retrieved_docs])
    history = convert_history(req.chat_history[-6:])

    async def token_generator() -> AsyncGenerator[str, None]:
        try:
            chain, _ = build_llm_chain(context_str)
            async for chunk in chain.astream({"context": context_str, "question": req.question, "history": history}):
                if chunk.content:
                    yield f"data: {chunk.content}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            logger.error("Stream error: %s", e)
            yield f"data: [ERROR] {str(e)}\n\n"

    return StreamingResponse(token_generator(), media_type="text/event-stream")


@app.get("/documents")
async def list_documents():
    """List all user-uploaded documents (excludes baseline records)."""
    vs = get_faiss_store()
    docs = []
    for doc_id, doc in vs.docstore._dict.items():
        if not doc.metadata.get("is_baseline", False):
            docs.append({
                "id": doc_id,
                "source": doc.metadata.get("source", "Unknown"),
                "doc_type": doc.metadata.get("doc_type", "Record"),
                "date": doc.metadata.get("date", "N/A"),
                "file_type": doc.metadata.get("file_type", "unknown"),
                "snippet": doc.page_content[:200],
            })
    logger.info("Document list requested — %d user docs returned.", len(docs))
    return {"documents": docs, "count": len(docs)}


@app.delete("/documents/{doc_id}")
async def delete_document(doc_id: str):
    """Delete a single user-uploaded document by its FAISS store ID."""
    vs = get_faiss_store()

    if doc_id not in vs.docstore._dict:
        raise HTTPException(status_code=404, detail=f"Document ID '{doc_id}' not found.")

    doc = vs.docstore._dict[doc_id]
    if doc.metadata.get("is_baseline", False):
        raise HTTPException(status_code=403, detail="Baseline records cannot be deleted.")

    source_name = doc.metadata.get("source", doc_id)
    vs.delete([doc_id])
    vs.save_local(FAISS_DB_DIR)
    logger.info("Document '%s' (id=%s) deleted from FAISS.", source_name, doc_id)
    return {"status": "deleted", "id": doc_id, "source": source_name}


@app.delete("/user-documents")
async def clear_all_user_documents():
    """Clear ALL user-uploaded documents. Baseline records are preserved."""
    global _faiss_store
    embeddings = get_embeddings()
    _faiss_store = FAISS.from_documents(BASELINE_PATIENT_RECORDS, embeddings)
    _faiss_store.save_local(FAISS_DB_DIR)
    logger.info("All user documents cleared. Baseline records retained.")
    return {"status": "success", "message": "User documents cleared. Baseline records retained."}
