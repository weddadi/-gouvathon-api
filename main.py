"""
Gouvathon 2026 — RAG Chatbot API
Ministère du Commerce et du Tourisme de Mauritanie

Converted from Jupyter Notebook to FastAPI
"""

import os
import pickle
import numpy as np
import faiss
import google.generativeai as genai

from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from contextlib import asynccontextmanager

# ─── Config ───────────────────────────────────────────────────────────

DATA_DIR = Path("data")
INDEX_PATH = DATA_DIR / "faiss_index.bin"
DOCS_PATH = DATA_DIR / "documents.pkl"
UPLOAD_DIR = Path("uploads")

# ─── Global State ─────────────────────────────────────────────────────

documents: list = []
index: faiss.IndexFlatL2 | None = None
chat_sessions: dict[str, list] = {}

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)


# ─── Helpers ──────────────────────────────────────────────────────────

def configure_gemini():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY environment variable is not set. "
            "Get one at https://aistudio.google.com/app/apikey"
        )
    genai.configure(api_key=api_key)


def read_pdf(path: str) -> str:
    reader = PdfReader(path)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text


def get_embedding(text: str) -> list[float]:
    response = genai.embed_content(
        model="models/gemini-embedding-2",
        content=text,
    )
    return response["embedding"]


def load_index():
    """Load saved FAISS index and documents if they exist."""
    global index, documents
    if INDEX_PATH.exists() and DOCS_PATH.exists():
        index = faiss.read_index(str(INDEX_PATH))
        with open(DOCS_PATH, "rb") as f:
            documents = pickle.load(f)
        print(f"✅ Loaded index with {index.ntotal} vectors and {len(documents)} chunks")
        return True
    return False


def save_index():
    """Persist FAISS index and documents to disk."""
    DATA_DIR.mkdir(exist_ok=True)
    faiss.write_index(index, str(INDEX_PATH))
    with open(DOCS_PATH, "wb") as f:
        pickle.dump(documents, f)
    print("💾 Index saved to disk")


def search(query: str, k: int = 5) -> list[dict]:
    if index is None or index.ntotal == 0:
        return []
    query_embedding = np.array([get_embedding(query)]).astype("float32")
    distances, indices = index.search(query_embedding, k)
    results = []
    for i, idx in enumerate(indices[0]):
        if idx < len(documents):
            result = documents[idx].copy()
            result["score"] = float(distances[0][i])
            results.append(result)
    return results


# ─── Lifespan ─────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_gemini()
    UPLOAD_DIR.mkdir(exist_ok=True)
    DATA_DIR.mkdir(exist_ok=True)
    load_index()
    yield


# ─── App ──────────────────────────────────────────────────────────────

app = FastAPI(
    title="Gouvathon 2026 — المستشار API",
    description=(
        "API RAG Chatbot pour les procédures du "
        "Ministère du Commerce et du Tourisme de Mauritanie"
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Models ───────────────────────────────────────────────────────────

class AskRequest(BaseModel):
    question: str
    session_id: str = "default"
    top_k: int = 5

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "question": "ما هي وثائق اعتماد مؤسسة إيواء؟",
                    "session_id": "user-123",
                    "top_k": 5,
                }
            ]
        }
    }


class AskResponse(BaseModel):
    answer: str
    sources: list[dict]
    session_id: str


class SearchRequest(BaseModel):
    query: str
    top_k: int = 5


class StatusResponse(BaseModel):
    status: str
    total_chunks: int
    total_vectors: int
    files_indexed: list[str]


# ─── Endpoints ────────────────────────────────────────────────────────

@app.get("/", tags=["Health"])
def root():
    return {
        "service": "Gouvathon 2026 — المستشار API",
        "status": "running",
        "docs": "/docs",
    }


@app.get("/status", response_model=StatusResponse, tags=["Health"])
def get_status():
    files = list({doc["source"] for doc in documents}) if documents else []
    return StatusResponse(
        status="ready" if index and index.ntotal > 0 else "empty",
        total_chunks=len(documents),
        total_vectors=index.ntotal if index else 0,
        files_indexed=files,
    )


@app.post("/upload", tags=["Documents"])
async def upload_pdf(file: UploadFile = File(...)):
    """
    Upload a PDF file, extract text, generate embeddings,
    and add to the FAISS index.
    """
    global index, documents

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted")

    # Save uploaded file
    file_path = UPLOAD_DIR / file.filename
    content = await file.read()
    with open(file_path, "wb") as f:
        f.write(content)

    # Extract text
    text = read_pdf(str(file_path))
    if not text.strip():
        raise HTTPException(status_code=400, detail="Could not extract text from PDF")

    # Split into chunks
    chunks = splitter.split_text(text)

    # Generate embeddings
    new_embeddings = []
    new_docs = []
    for i, chunk in enumerate(chunks):
        emb = get_embedding(chunk)
        new_embeddings.append(emb)
        new_docs.append({
            "text": chunk,
            "source": file.filename,
            "chunk": i,
        })

    # Add to FAISS index
    embedding_matrix = np.array(new_embeddings).astype("float32")

    if index is None:
        dimension = embedding_matrix.shape[1]
        index = faiss.IndexFlatL2(dimension)

    index.add(embedding_matrix)
    documents.extend(new_docs)

    # Save to disk
    save_index()

    return {
        "message": f"✅ {file.filename} indexed successfully",
        "chunks_added": len(new_docs),
        "total_chunks": len(documents),
        "total_vectors": index.ntotal,
    }


@app.post("/ask", response_model=AskResponse, tags=["Chat"])
def ask_question(req: AskRequest):
    """
    Ask a question — the system retrieves relevant chunks
    from indexed PDFs and generates an answer using Gemini.
    """
    if index is None or index.ntotal == 0:
        raise HTTPException(
            status_code=400,
            detail="No documents indexed yet. Upload a PDF first via /upload",
        )

    # Retrieve relevant chunks
    contexts = search(req.question, k=req.top_k)

    context_text = ""
    for c in contexts:
        context_text += (
            f"\nSource: {c['source']}\nChunk: {c['chunk']}\n\n"
            f"{c['text']}\n\n--------------------\n"
        )

    # Build conversation history
    history = chat_sessions.get(req.session_id, [])
    history_text = ""
    for h in history[-3:]:
        history_text += f"User: {h['user']}\nAssistant: {h['assistant']}\n"

    # Build prompt
    prompt = f"""أنت "المستشار"، مساعد ذكي متخصص في إجراءات وزارة التجارة والسياحة الموريتانية.

القواعد:
- أجب فقط باستخدام السياق المقدم أدناه.
- إذا لم تجد الجواب في السياق، قل: "لم أجد هذه المعلومة في الوثائق المتاحة."
- كن دقيقًا ومختصرًا.
- أجب بنفس لغة السؤال (عربي أو فرنسي).
- اذكر الرسوم والآجال والوثائق المطلوبة عند الإمكان.

سجل المحادثة:
{history_text}

السياق:
{context_text}

السؤال:
{req.question}
"""

    # Generate answer
    model = genai.GenerativeModel("gemini-2.5-flash")
    response = model.generate_content(prompt)
    answer = response.text

    # Update session history
    if req.session_id not in chat_sessions:
        chat_sessions[req.session_id] = []
    chat_sessions[req.session_id].append({
        "user": req.question,
        "assistant": answer,
    })

    # Deduplicate sources
    seen = set()
    sources = []
    for c in contexts:
        key = (c["source"], c["chunk"])
        if key not in seen:
            seen.add(key)
            sources.append({
                "file": c["source"],
                "chunk": c["chunk"],
                "score": c.get("score", 0),
                "preview": c["text"][:150] + "...",
            })

    return AskResponse(
        answer=answer,
        sources=sources,
        session_id=req.session_id,
    )


@app.post("/search", tags=["Search"])
def search_documents(req: SearchRequest):
    """
    Search the indexed documents without generating an answer.
    Useful for debugging or browsing chunks.
    """
    if index is None or index.ntotal == 0:
        raise HTTPException(status_code=400, detail="No documents indexed yet")

    results = search(req.query, k=req.top_k)
    return {
        "query": req.query,
        "results": [
            {
                "file": r["source"],
                "chunk": r["chunk"],
                "score": r.get("score", 0),
                "text": r["text"],
            }
            for r in results
        ],
    }


@app.delete("/sessions/{session_id}", tags=["Chat"])
def clear_session(session_id: str):
    """Clear conversation history for a session."""
    if session_id in chat_sessions:
        del chat_sessions[session_id]
        return {"message": f"Session '{session_id}' cleared"}
    raise HTTPException(status_code=404, detail="Session not found")


@app.delete("/index", tags=["Documents"])
def reset_index():
    """Delete all indexed documents and reset the FAISS index."""
    global index, documents
    index = None
    documents = []
    if INDEX_PATH.exists():
        INDEX_PATH.unlink()
    if DOCS_PATH.exists():
        DOCS_PATH.unlink()
    return {"message": "🗑️ Index reset successfully"}
