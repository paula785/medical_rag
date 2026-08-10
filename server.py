"""
MedRAG FastAPI Backend Microservice (Cloud-Powered)
---------------------------------------------------
Decoupled REST API backend using Cloud APIs:
- Google Gemini 1.5 Flash for ultra-fast Vision OCR
- Google Gemini text-embedding-004 for high-precision embeddings
- Groq Llama-3.3-70B for expert clinical RAG reasoning
- FAISS vector store for fast local indexing
"""

import os
import base64
from io import BytesIO
from datetime import datetime

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pypdf import PdfReader

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

load_dotenv()

# App & Middleware
app = FastAPI(title="MedRAG Cloud Backend Microservice", version="3.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

FAISS_DB_DIR = "./med_rag_faiss"
GROQ_MODEL_DEFAULT = "llama-3.3-70b-versatile"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

# Baseline Patient Data (Unchangeable system background records)
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

def get_embeddings(gemini_key: str = None):
    key = gemini_key or os.getenv("GEMINI_API_KEY", "")
    if not key:
        raise HTTPException(status_code=400, detail="Missing Gemini API Key for Embeddings.")
    return GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", google_api_key=key)

def load_or_init_faiss(gemini_key: str = None):
    embeddings = get_embeddings(gemini_key)
    index_path = os.path.join(FAISS_DB_DIR, "index.faiss")
    if os.path.exists(FAISS_DB_DIR) and os.path.exists(index_path):
        try:
            return FAISS.load_local(FAISS_DB_DIR, embeddings, allow_dangerous_deserialization=True)
        except Exception:
            pass
    store = FAISS.from_documents(BASELINE_PATIENT_RECORDS, embeddings)
    store.save_local(FAISS_DB_DIR)
    return store

# Request Models
class ChatRequest(BaseModel):
    question: str
    groq_api_key: str
    gemini_api_key: str = ""

# Endpoints
@app.get("/health")
async def health_check(gemini_key: str = ""):
    key = gemini_key or os.getenv("GEMINI_API_KEY", "")
    gemini_configured = bool(key)
    groq_configured = bool(os.getenv("GROQ_API_KEY", ""))
    
    faiss_count = 0
    if gemini_configured:
        try:
            vs = load_or_init_faiss(key)
            for _, doc in vs.docstore._dict.items():
                if not doc.metadata.get("is_baseline", False):
                    faiss_count += 1
        except Exception:
            pass

    return {
        "status": "online",
        "gemini_active": gemini_configured,
        "groq_active": groq_configured,
        "user_documents_count": faiss_count
    }

@app.post("/ingest")
async def ingest_document(
    file: UploadFile = File(...),
    doc_type: str = Form(...),
    gemini_api_key: str = Form("")
):
    key = gemini_api_key or os.getenv("GEMINI_API_KEY", "")
    if not key:
        raise HTTPException(status_code=400, detail="Gemini API Key is required for Cloud OCR & Embeddings.")

    contents = await file.read()
    filename = file.filename
    ext = filename.split(".")[-1].lower() if "." in filename else "bin"
    
    extracted_text = ""
    if ext == "pdf":
        try:
            reader = PdfReader(BytesIO(contents))
            pages = [f"--- PAGE {i+1} ---\n{p.extract_text().strip()}" for i, p in enumerate(reader.pages) if p.extract_text()]
            extracted_text = "\n\n".join(pages) if pages else "No text extracted from PDF."
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"PDF extraction error: {e}")
    else:
        # Perform ultra-fast Cloud Vision OCR using Gemini 2.5 Flash
        try:
            img_b64 = base64.b64encode(contents).decode("utf-8")
            mime_type = file.content_type or f"image/{ext}"
            
            prompt = """You are an expert clinical document OCR specialist and medical transcription assistant.
Examine this medical image carefully. Extract all visible text, doctor details, dates, medications, dosages, lab parameters, and clinical values accurately.
Summarize key findings in structured headings: DOCUMENT TYPE & DATE, PATIENT & PROVIDER INFO, LAB PARAMETERS & TEST RESULTS, MEDICATION PRESCRIPTIONS, CLINICAL NOTES."""

            llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash",
                google_api_key=key,
                temperature=0.1
            )
            message = HumanMessage(
                content=[
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": f"data:{mime_type};base64,{img_b64}"
                    }
                ]
            )
            res = llm.invoke([message])
            extracted_text = res.content
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Gemini Cloud OCR error: {e}")

    # Embed & index into FAISS using Gemini Embeddings
    try:
        doc = Document(
            page_content=extracted_text,
            metadata={
                "source": filename,
                "doc_type": doc_type,
                "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "is_baseline": False,
                "file_type": ext
            }
        )
        vs = load_or_init_faiss(key)
        vs.add_documents([doc])
        vs.save_local(FAISS_DB_DIR)
    except Exception as err:
        raise HTTPException(status_code=500, detail=f"FAISS embedding error: {err}")

    return {"status": "success", "filename": filename, "extracted_text": extracted_text}

@app.post("/chat")
async def chat_medical_rag(req: ChatRequest):
    groq_key = req.groq_api_key or os.getenv("GROQ_API_KEY", "")
    gemini_key = req.gemini_api_key or os.getenv("GEMINI_API_KEY", "")
    
    if not groq_key:
        raise HTTPException(status_code=400, detail="Missing Groq API Key.")
    if not gemini_key:
        raise HTTPException(status_code=400, detail="Missing Gemini API Key.")

    vs = load_or_init_faiss(gemini_key)
    retriever = vs.as_retriever(search_kwargs={"k": 4})
    retrieved_docs = retriever.invoke(req.question)
    
    context_str = "\n\n---\n\n".join([doc.page_content for doc in retrieved_docs])
    
    system_prompt = """You are an elite, board-certified Clinical AI Specialist and Pharmacotherapy Consultant.
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

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{question}")
    ])
    
    try:
        llm = ChatGroq(model_name=GROQ_MODEL_DEFAULT, groq_api_key=groq_key, temperature=0.2)
        chain = prompt | llm
        response = chain.invoke({"context": context_str, "question": req.question})
        
        user_sources = [
            {
                "source": d.metadata.get("source", "Unknown"),
                "doc_type": d.metadata.get("doc_type", "Record"),
                "date": d.metadata.get("date", "N/A"),
                "snippet": d.page_content[:300] + "..."
            }
            for d in retrieved_docs if not d.metadata.get("is_baseline", False)
        ]
        
        return {"answer": response.content, "user_sources": user_sources}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Groq API Error: {str(e)}")

@app.delete("/user-documents")
async def clear_user_documents(gemini_key: str = ""):
    key = gemini_key or os.getenv("GEMINI_API_KEY", "")
    embeddings = get_embeddings(key)
    vs = FAISS.from_documents(BASELINE_PATIENT_RECORDS, embeddings)
    vs.save_local(FAISS_DB_DIR)
    return {"status": "success", "message": "User documents cleared. Baseline records retained."}
