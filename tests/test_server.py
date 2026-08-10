"""
MedRAG Backend — Test Suite
Tests all FastAPI endpoints using httpx.AsyncClient (in-process, no Uvicorn needed).

Run:
    pytest tests/ -v
"""

import io
import os
import hashlib
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

# Set dummy env keys before importing server so startup doesn't fail
os.environ.setdefault("GEMINI_API_KEY", "test-gemini-key")
os.environ.setdefault("GROQ_API_KEY", "test-groq-key")

from server import app, BASELINE_PATIENT_RECORDS, MAX_FILE_SIZE, MAX_QUESTION_LEN

client = TestClient(app, raise_server_exceptions=False)


# ─── Fixtures ─────────────────────────────────────────────────────────────────

def make_pdf_bytes() -> bytes:
    """Minimal valid 1-page PDF with text."""
    return b"""%PDF-1.4
1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj
2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj
3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792]
  /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >> endobj
4 0 obj << /Length 44 >>
stream
BT /F1 12 Tf 100 700 Td (Lab Report: HbA1c 7.2%) Tj ET
endstream
endobj
5 0 obj << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> endobj
xref
0 6
0000000000 65535 f
0000000009 00000 n
0000000058 00000 n
0000000115 00000 n
0000000266 00000 n
0000000360 00000 n
trailer << /Size 6 /Root 1 0 R >>
startxref
441
%%EOF"""

def make_png_bytes() -> bytes:
    """Minimal 1x1 white PNG."""
    return (
        b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01'
        b'\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00'
        b'\x00\x0cIDATx\x9cc\xf8\xff\xff?\x00\x05\xfe\x02\xfe'
        b'\xdc\xccY\xe7\x00\x00\x00\x00IEND\xaeB`\x82'
    )


# ─── Mock helpers ──────────────────────────────────────────────────────────────

def mock_faiss_store():
    """Returns a mock FAISS store backed by real baseline docs for testing."""
    store = MagicMock()
    # Simulate docstore with only baseline records
    store.docstore._dict = {
        f"baseline-{i}": doc for i, doc in enumerate(BASELINE_PATIENT_RECORDS)
    }
    store.index.ntotal = len(BASELINE_PATIENT_RECORDS)
    store.as_retriever.return_value.invoke.return_value = BASELINE_PATIENT_RECORDS[:2]
    return store


# ─── Phase 1 Tests: Health Endpoint ───────────────────────────────────────────

class TestHealth:
    def test_health_returns_online(self):
        """GET /health should return status=online when server is running."""
        with patch("server.get_faiss_store", return_value=mock_faiss_store()):
            r = client.get("/health")
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "online"

    def test_health_gemini_active_when_key_set(self):
        """gemini_active should be True when GEMINI_API_KEY env var is set."""
        with patch("server.get_faiss_store", return_value=mock_faiss_store()):
            r = client.get("/health")
        assert r.json()["gemini_active"] is True

    def test_health_groq_active_when_key_set(self):
        """groq_active should be True when GROQ_API_KEY env var is set."""
        with patch("server.get_faiss_store", return_value=mock_faiss_store()):
            r = client.get("/health")
        assert r.json()["groq_active"] is True

    def test_health_includes_model_names(self):
        """health response should include model names."""
        with patch("server.get_faiss_store", return_value=mock_faiss_store()):
            r = client.get("/health")
        models = r.json().get("models", {})
        assert "ocr" in models
        assert "embed" in models
        assert "chat" in models

    def test_health_missing_gemini_key(self):
        """gemini_active=False when GEMINI_API_KEY is empty."""
        with patch("server.GEMINI_API_KEY", ""), \
             patch("server.get_faiss_store", return_value=mock_faiss_store()):
            r = client.get("/health")
        assert r.json()["gemini_active"] is False


# ─── Phase 2 Tests: Document Ingestion ────────────────────────────────────────

class TestIngest:
    def test_ingest_pdf_success(self):
        """POST /ingest with a valid PDF should return status=success."""
        pdf_bytes = make_pdf_bytes()
        store = mock_faiss_store()
        store.add_documents = MagicMock()
        store.save_local = MagicMock()

        with patch("server.get_faiss_store", return_value=store):
            r = client.post(
                "/ingest",
                files={"file": ("report.pdf", pdf_bytes, "application/pdf")},
                data={"doc_type": "Lab Report"},
            )
        assert r.status_code == 200
        assert r.json()["status"] == "success"
        assert r.json()["filename"] == "report.pdf"

    def test_ingest_oversized_file_rejected(self):
        """POST /ingest with file >10MB should return 413."""
        big_file = b"x" * (MAX_FILE_SIZE + 1)
        store = mock_faiss_store()
        with patch("server.get_faiss_store", return_value=store):
            r = client.post(
                "/ingest",
                files={"file": ("big.pdf", big_file, "application/pdf")},
                data={"doc_type": "General Record"},
            )
        assert r.status_code == 413

    def test_ingest_duplicate_rejected(self):
        """Uploading the same file twice should return status=duplicate on second upload."""
        pdf_bytes = make_pdf_bytes()
        content_hash = hashlib.sha256(pdf_bytes).hexdigest()

        # Simulate a docstore that already has this hash
        from langchain_core.documents import Document
        existing_doc = Document(
            page_content="Lab Report: HbA1c 7.2%",
            metadata={"source": "report.pdf", "doc_type": "Lab Report", "is_baseline": False, "content_hash": content_hash}
        )
        store = mock_faiss_store()
        store.docstore._dict["existing-doc-id"] = existing_doc

        with patch("server.get_faiss_store", return_value=store):
            r = client.post(
                "/ingest",
                files={"file": ("report.pdf", pdf_bytes, "application/pdf")},
                data={"doc_type": "Lab Report"},
            )
        assert r.status_code == 200
        assert r.json()["status"] == "duplicate"

    def test_ingest_missing_gemini_key(self):
        """POST /ingest without GEMINI_API_KEY should return 400."""
        with patch("server.GEMINI_API_KEY", ""):
            r = client.post(
                "/ingest",
                files={"file": ("report.pdf", b"%PDF-1.4", "application/pdf")},
                data={"doc_type": "Lab Report"},
            )
        assert r.status_code == 400


# ─── Phase 3 Tests: Chat ──────────────────────────────────────────────────────

class TestChat:
    def _mock_groq_response(self, content: str = "Test clinical answer."):
        mock_resp = MagicMock()
        mock_resp.content = content
        chain = MagicMock()
        chain.invoke.return_value = mock_resp
        return chain

    def test_chat_question_too_long(self):
        """POST /chat with question >2000 chars should return 422 (pydantic validation)."""
        r = client.post("/chat", json={
            "question": "x" * (MAX_QUESTION_LEN + 1),
            "chat_history": []
        })
        assert r.status_code == 422

    def test_chat_basic_response(self):
        """POST /chat with valid question should return answer."""
        store = mock_faiss_store()
        mock_chain = self._mock_groq_response("Patient has Penicillin allergy.")

        with patch("server.get_faiss_store", return_value=store), \
             patch("server.build_llm_chain", return_value=(mock_chain, "")):
            r = client.post("/chat", json={
                "question": "What are the patient's drug allergies?",
                "chat_history": []
            })

        assert r.status_code == 200
        assert "answer" in r.json()
        assert "user_sources" in r.json()

    def test_chat_missing_groq_key(self):
        """POST /chat without GROQ_API_KEY should return 400."""
        store = mock_faiss_store()
        with patch("server.GROQ_API_KEY", ""), \
             patch("server.get_faiss_store", return_value=store):
            r = client.post("/chat", json={
                "question": "Any allergies?",
                "chat_history": []
            })
        assert r.status_code == 400

    def test_chat_with_history(self):
        """POST /chat should accept chat_history without error."""
        store = mock_faiss_store()
        mock_chain = self._mock_groq_response("Follow-up answer.")

        history = [
            {"role": "user", "content": "What is the HbA1c?"},
            {"role": "assistant", "content": "HbA1c is 7.4%."},
        ]

        with patch("server.get_faiss_store", return_value=store), \
             patch("server.build_llm_chain", return_value=(mock_chain, "")):
            r = client.post("/chat", json={
                "question": "Is that value normal?",
                "chat_history": history
            })

        assert r.status_code == 200


# ─── Phase 4 Tests: Document Management ───────────────────────────────────────

class TestDocumentManagement:
    def test_list_documents_returns_user_docs_only(self):
        """GET /documents should return only non-baseline documents."""
        from langchain_core.documents import Document
        store = mock_faiss_store()
        # Add a user doc to the mock store
        user_doc = Document(
            page_content="Uploaded lab report.",
            metadata={"source": "labs.pdf", "doc_type": "Lab Report", "date": "2026-08-10", "is_baseline": False, "file_type": "pdf"}
        )
        store.docstore._dict["user-doc-1"] = user_doc

        with patch("server.get_faiss_store", return_value=store):
            r = client.get("/documents")

        assert r.status_code == 200
        data = r.json()
        assert "documents" in data
        # All returned docs must be user docs (not baseline)
        for doc in data["documents"]:
            assert doc.get("source") != "baseline_medical_history.pdf"

    def test_delete_document_success(self):
        """DELETE /documents/{doc_id} should remove a user document."""
        from langchain_core.documents import Document
        store = mock_faiss_store()
        user_doc = Document(
            page_content="Lab result text.",
            metadata={"source": "labs.pdf", "doc_type": "Lab Report", "date": "2026-08-10", "is_baseline": False}
        )
        store.docstore._dict["user-doc-abc"] = user_doc
        store.delete = MagicMock()
        store.save_local = MagicMock()

        with patch("server.get_faiss_store", return_value=store):
            r = client.delete("/documents/user-doc-abc")

        assert r.status_code == 200
        assert r.json()["status"] == "deleted"

    def test_delete_baseline_document_forbidden(self):
        """DELETE /documents/{id} on a baseline doc should return 403."""
        store = mock_faiss_store()
        # baseline-0 is a baseline doc in our mock

        with patch("server.get_faiss_store", return_value=store):
            r = client.delete("/documents/baseline-0")

        assert r.status_code == 403

    def test_clear_user_documents(self):
        """DELETE /user-documents should succeed and rebuild baseline-only store."""
        mock_embeddings = MagicMock()
        new_store = mock_faiss_store()
        new_store.save_local = MagicMock()

        with patch("server.get_embeddings", return_value=mock_embeddings), \
             patch("server.FAISS.from_documents", return_value=new_store):
            r = client.delete("/user-documents")

        assert r.status_code == 200
        assert r.json()["status"] == "success"
