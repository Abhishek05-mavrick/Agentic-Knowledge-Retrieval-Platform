from __future__ import annotations

import os
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from flask import Flask, jsonify, render_template, request
from werkzeug.utils import secure_filename

sys.path.insert(0, str(Path(__file__).parent.parent))

from services.chunking import re_te_sp
from services.faiss_db import FAISSDB
from services.ingestion import audio_extractor, pdf_extractor, web_extractor, yt_extractor_robust
from services.langgraph_agent import SimpleRAGLangGraphAgent
from services.retriever import vector_store
from utility.logger import logger

app = Flask(__name__, template_folder="../templates", static_folder="../static")
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024
app.config["UPLOAD_FOLDER"] = "../uploads_pdf"

agent = SimpleRAGLangGraphAgent()
threads_db_path = str(Path("thread_metadata.sqlite").resolve())


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _init_thread_store() -> None:
    with sqlite3.connect(threads_db_path) as connection:
        cursor = connection.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS threads (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                last_message TEXT
            )
            """
        )
        connection.commit()


def _upsert_thread(thread_id: str, title: str, last_message: str) -> None:
    now = _utc_now_iso()
    with sqlite3.connect(threads_db_path) as connection:
        cursor = connection.cursor()
        cursor.execute("SELECT id FROM threads WHERE id = ?", (thread_id,))
        exists = cursor.fetchone() is not None

        if exists:
            cursor.execute(
                """
                UPDATE threads
                SET title = ?, updated_at = ?, last_message = ?
                WHERE id = ?
                """,
                (title, now, last_message, thread_id),
            )
        else:
            cursor.execute(
                """
                INSERT INTO threads (id, title, created_at, updated_at, last_message)
                VALUES (?, ?, ?, ?, ?)
                """,
                (thread_id, title, now, now, last_message),
            )
        connection.commit()


def _get_thread(thread_id: str) -> dict | None:
    with sqlite3.connect(threads_db_path) as connection:
        cursor = connection.cursor()
        cursor.execute(
            "SELECT id, title, created_at, updated_at, last_message FROM threads WHERE id = ?",
            (thread_id,),
        )
        row = cursor.fetchone()

    if not row:
        return None

    return {
        "thread_id": row[0],
        "title": row[1],
        "created_at": row[2],
        "updated_at": row[3],
        "last_message": row[4] or "",
    }


def _list_threads() -> list[dict]:
    with sqlite3.connect(threads_db_path) as connection:
        cursor = connection.cursor()
        cursor.execute(
            """
            SELECT id, title, created_at, updated_at, last_message
            FROM threads
            ORDER BY updated_at DESC
            """
        )
        rows = cursor.fetchall()

    return [
        {
            "thread_id": row[0],
            "title": row[1],
            "created_at": row[2],
            "updated_at": row[3],
            "last_message": row[4] or "",
        }
        for row in rows
    ]


def _ensure_upload_folder() -> None:
    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)


@app.get("/")
def home():
    return render_template("chat.html")


@app.get("/health")
def health():
    return jsonify({"status": "ok", "service": "langgraph-rag"}), 200


@app.post("/api/chat")
def chat_api():
    payload = request.get_json(silent=True) or {}
    message = (payload.get("message") or "").strip()
    thread_id = (payload.get("thread_id") or "").strip() or str(uuid4())

    if not message:
        return jsonify({"error": "message is required"}), 400

    try:
        thread = _get_thread(thread_id)
        if not thread:
            title = agent.generate_thread_title(message)
        else:
            title = thread["title"]

        answer = agent.chat(message=message, thread_id=thread_id)
        _upsert_thread(thread_id=thread_id, title=title, last_message=message)

        return (
            jsonify(
                {
                    "thread_id": thread_id,
                    "title": title,
                    "message": message,
                    "answer": answer,
                }
            ),
            200,
        )
    except Exception as error:
        logger.error(f"/api/chat failed: {error}")
        return jsonify({"error": str(error)}), 500


@app.post("/api/threads")
def create_thread():
    payload = request.get_json(silent=True) or {}
    thread_id = (payload.get("thread_id") or "").strip() or str(uuid4())
    title = (payload.get("title") or "New Chat").strip() or "New Chat"
    _upsert_thread(thread_id=thread_id, title=title, last_message="")
    return jsonify({"thread_id": thread_id, "title": title}), 201


@app.get("/api/threads")
def list_threads_api():
    return jsonify({"threads": _list_threads()}), 200


@app.get("/api/threads/<thread_id>/messages")
def thread_messages(thread_id: str):
    thread = _get_thread(thread_id)
    if not thread:
        return jsonify({"error": "thread not found"}), 404

    try:
        history = agent.get_thread_messages(thread_id=thread_id)
        return jsonify({"thread": thread, "messages": history}), 200
    except Exception as error:
        logger.error(f"Failed to fetch thread messages for {thread_id}: {error}")
        return jsonify({"error": str(error)}), 500


@app.post("/api/upload")
def upload_api():
    try:
        documents = []
        _ensure_upload_folder()

        if "files" in request.files:
            files = request.files.getlist("files")
            for file in files:
                if not file or not file.filename:
                    continue
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
                file.save(filepath)

                lower_name = filename.lower()
                if lower_name.endswith(".pdf"):
                    documents.extend(pdf_extractor(filepath))
                elif lower_name.endswith((".mp3", ".wav", ".m4a", ".flac", ".aac", ".ogg")):
                    documents.extend(audio_extractor(filepath))

        url = (request.form.get("url") or "").strip()
        url_type = (request.form.get("url_type") or "").strip().lower()
        if url:
            if url_type == "youtube":
                documents.extend(yt_extractor_robust(url))
            else:
                documents.extend(web_extractor(url))

        if not documents:
            return jsonify({"status": "error", "message": "No valid file/url/audio/youtube input provided"}), 400

        chunks = re_te_sp(documents)

        store = vector_store
        if isinstance(vector_store, FAISSDB):
            store = vector_store
        store.load_into_db(chunks)
        store.save_db("vector_db")

        return (
            jsonify(
                {
                    "status": "success",
                    "message": f"Indexed {len(documents)} documents into {len(chunks)} chunks",
                    "chunks": len(chunks),
                }
            ),
            200,
        )
    except Exception as error:
        logger.error(f"/api/upload failed: {error}")
        return jsonify({"status": "error", "message": str(error)}), 500


if __name__ == "__main__":
    _init_thread_store()
    logger.info("Starting simple LangGraph RAG server on port 5001")
    app.run(debug=True, host="0.0.0.0", port=5001)