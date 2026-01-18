from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
from pydantic import BaseModel
from typing import Optional
from datetime import datetime
import os

from pymongo import MongoClient
from bson import ObjectId

from rag_qa import answer_question  

app = FastAPI()

templates = Jinja2Templates(directory="templates")

# --------------------------------------------------
# MongoDB Configuration (Chat History Only)
# --------------------------------------------------
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
mongo_client = MongoClient(MONGO_URI)
db = mongo_client["examaid"]
messages_collection = db["chat_messages"]

# Create index for efficient querying
messages_collection.create_index([("session_id", 1), ("timestamp", 1)])


# --------------------------------------------------
# Pydantic Models
# --------------------------------------------------
class QuestionRequest(BaseModel):
    question: str
    session_id: str


class MessageResponse(BaseModel):
    id: str
    session_id: str
    role: str  # "user" or "assistant"
    content: str
    timestamp: str


# --------------------------------------------------
# Helper Functions
# --------------------------------------------------
def save_message(session_id: str, role: str, content: str) -> dict:
    """Save a message to MongoDB and return the saved document."""
    message = {
        "session_id": session_id,
        "role": role,
        "content": content,
        "timestamp": datetime.utcnow()
    }
    result = messages_collection.insert_one(message)
    message["_id"] = result.inserted_id
    return message


def get_chat_history(session_id: str) -> list:
    """Retrieve all messages for a session, sorted by timestamp ascending."""
    messages = messages_collection.find(
        {"session_id": session_id}
    ).sort("timestamp", 1)
    
    return [
        {
            "id": str(msg["_id"]),
            "session_id": msg["session_id"],
            "role": msg["role"],
            "content": msg["content"],
            "timestamp": msg["timestamp"].isoformat()
        }
        for msg in messages
    ]


def clear_session_history(session_id: str) -> int:
    """Delete all messages for a session. Returns count of deleted messages."""
    result = messages_collection.delete_many({"session_id": session_id})
    return result.deleted_count


def get_all_sessions() -> list:
    """Get all unique sessions with their first message and timestamp."""
    pipeline = [
        {"$sort": {"timestamp": 1}},
        {"$group": {
            "_id": "$session_id",
            "first_message": {"$first": "$content"},
            "first_role": {"$first": "$role"},
            "created_at": {"$first": "$timestamp"},
            "last_message_at": {"$last": "$timestamp"},
            "message_count": {"$sum": 1}
        }},
        {"$sort": {"last_message_at": -1}}
    ]
    
    sessions = list(messages_collection.aggregate(pipeline))
    
    return [
        {
            "session_id": s["_id"],
            "preview": s["first_message"][:100] + "..." if len(s["first_message"]) > 100 else s["first_message"],
            "created_at": s["created_at"].isoformat(),
            "last_message_at": s["last_message_at"].isoformat(),
            "message_count": s["message_count"]
        }
        for s in sessions
    ]


# --------------------------------------------------
# Routes
# --------------------------------------------------
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "answer": None}
    )


@app.post("/ask", response_class=HTMLResponse)
def ask_question_form(request: Request, question: str = Form(...)):
    answer = answer_question(question)
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "answer": answer}
    )


@app.post("/api/ask")
async def ask_question_api(req: QuestionRequest):
    """
    JSON API endpoint for AJAX requests.
    Saves user question and assistant response to MongoDB.
    RAG/answer generation still uses Qdrant.
    """
    try:
        # Step 1: Save user message immediately
        user_msg = save_message(req.session_id, "user", req.question)
        
        # Step 2: Generate answer using RAG (Qdrant-based)
        answer = answer_question(req.question)
        
        # Step 3: Save assistant response immediately
        assistant_msg = save_message(req.session_id, "assistant", answer)
        
        return JSONResponse(content={
            "success": True,
            "answer": answer,
            "user_message": {
                "id": str(user_msg["_id"]),
                "role": "user",
                "content": req.question,
                "timestamp": user_msg["timestamp"].isoformat()
            },
            "assistant_message": {
                "id": str(assistant_msg["_id"]),
                "role": "assistant",
                "content": answer,
                "timestamp": assistant_msg["timestamp"].isoformat()
            }
        })
    except Exception as e:
        return JSONResponse(
            content={
                "success": False,
                "answer": "An error occurred while processing your question.",
                "error": str(e)
            },
            status_code=500
        )


@app.get("/api/history/{session_id}")
async def get_history(session_id: str):
    """Retrieve chat history for a session."""
    try:
        messages = get_chat_history(session_id)
        return JSONResponse(content={
            "success": True,
            "messages": messages,
            "count": len(messages)
        })
    except Exception as e:
        return JSONResponse(
            content={"success": False, "error": str(e), "messages": []},
            status_code=500
        )


@app.delete("/api/history/{session_id}")
async def delete_history(session_id: str):
    """Clear chat history for a session."""
    try:
        deleted_count = clear_session_history(session_id)
        return JSONResponse(content={
            "success": True,
            "deleted_count": deleted_count
        })
    except Exception as e:
        return JSONResponse(
            content={"success": False, "error": str(e)},
            status_code=500
        )


@app.get("/api/sessions")
async def list_sessions():
    """Get all chat sessions with previews."""
    try:
        sessions = get_all_sessions()
        return JSONResponse(content={
            "success": True,
            "sessions": sessions,
            "count": len(sessions)
        })
    except Exception as e:
        return JSONResponse(
            content={"success": False, "error": str(e), "sessions": []},
            status_code=500
        )
