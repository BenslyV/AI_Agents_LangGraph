# ============================================================
# app.py
# FastAPI application for Conversational PDF RAG
# ============================================================
#
# API Flow:
#
# 1. Upload PDF
#       ↓
# 2. Process PDF
#       ↓
# 3. Ask Question
#       ↓
# 4. Get Answer using RAG + Chat History
#
# Swagger UI:
# http://127.0.0.1:8000/docs
#
# ============================================================

# ============================================================
# STEP 1: IMPORT REQUIRED LIBRARIES
# ============================================================

from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
from typing import List

from rag_service import (
    process_pdfs,
    ask_question,
    get_chat_history
)

# ============================================================
# STEP 2: CREATE FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="Conversational PDF RAG API",
    description="RAG application using PDF, Chroma, Hugging Face and Groq",
    version="1.0.0"
)

# ============================================================
# STEP 3: CREATE REQUEST MODEL FOR /ask
# ============================================================

class QuestionRequest(BaseModel):
    question: str
    session_id: str = "default_session"

# ============================================================
# STEP 4: HOME ENDPOINT
# ============================================================

@app.get("/")
def home():
    return {"message": "Conversational RAG API is running"}

# ============================================================
# STEP 5: PDF UPLOAD ENDPOINT (FIXED)
# ============================================================
#
# UploadFile + File(...) ensures Swagger shows file pickers.
# ============================================================

@app.post("/upload-pdf")
async def upload_pdf(
    files: List[UploadFile] = File(..., description="Upload one or more PDF files")
):
    try:
        # Connect to your RAG service for processing
        processed = process_pdfs(files)

        return {
            "message": "Files uploaded and processed successfully",
            "filenames": [file.filename for file in files],
            "result": processed
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================
# STEP 6: ASK QUESTION ENDPOINT
# ============================================================

@app.post("/ask")
def ask(request: QuestionRequest):
    try:
        answer = ask_question(
            question=request.question,
            session_id=request.session_id
        )
        return {
            "session_id": request.session_id,
            "question": request.question,
            "answer": answer
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

# ============================================================
# STEP 7: GET CHAT HISTORY
# ============================================================

@app.get("/history/{session_id}")
def history(session_id: str):
    messages = get_chat_history(session_id)
    return {
        "session_id": session_id,
        "history": [
            {"type": message.type, "content": message.content}
            for message in messages
        ]
    }