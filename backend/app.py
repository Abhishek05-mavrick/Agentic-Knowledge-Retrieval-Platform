from flask import Flask, request, render_template, jsonify, redirect
import sys
from pathlib import Path
import os
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))
load_dotenv()
from services.ingestion import pdf_extractor, web_extractor, yt_extractor_robust
from services.chunking import re_te_sp
from services.embedding import Embedder
from services.faiss_db import FAISSDB
from services.gen import get_answer
from utility.logger import logger

app = Flask(__name__, template_folder="../templates")
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max
app.config['UPLOAD_FOLDER'] = '../uploads_pdf'

# Initialize services
embedder = Embedder()
vector_db = FAISSDB(embedder)
vector_db.load_db("vector_db")  # Load existing DB if available


# Routes
@app.route("/", methods=["GET"])
def home():
    """Serve the upload page"""
    return render_template("uploader.html")


@app.route("/query", methods=["GET"])
def query():
    """Serve the query page"""
    return render_template("query.html")


@app.route("/upload", methods=["POST"])
def upload():
    """Handle file/URL uploads and process documents"""
    try:
        documents = []
        
        # Handle file uploads
        if 'files' in request.files:
            files = request.files.getlist('files')
            for file in files:
                if file and file.filename.endswith('.pdf'):
                    # Save file temporarily
                    filename = secure_filename(file.filename)
                    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
                    file.save(filepath)
                    
                    # Extract text from PDF
                    logger.info(f"Processing PDF: {filename}")
                    docs = pdf_extractor(filepath)
                    documents.extend(docs)
        
        # Handle URL uploads
        url = request.form.get('url', '').strip()
        url_type = request.form.get('url_type', '').strip()
        
        if url and url_type:
            logger.info(f"Processing URL ({url_type}): {url}")
            if url_type == 'youtube':
                docs = yt_extractor_robust(url)
            else:  # webpage or wikipedia
                docs = web_extractor(url)
            documents.extend(docs)
        
        # If we got documents, process them
        if documents:
            logger.info(f"Processing {len(documents)} documents")
            
            # Chunk documents
            chunks = re_te_sp(documents)
            logger.info(f"Created {len(chunks)} chunks")
            
            # Add to vector DB
            vector_db.load_into_db(chunks)
            
            # Save the updated DB
            vector_db.save_db("vector_db")
            logger.info("Documents indexed and saved to FAISS")
            
            return jsonify({
                "status": "success",
                "message": f"Successfully processed {len(documents)} documents into {len(chunks)} chunks",
                "chunks": len(chunks)
            }), 200
        else:
            return jsonify({
                "status": "error",
                "message": "No valid files or URLs provided"
            }), 400
            
    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        return jsonify({
            "status": "error",
            "message": f"Upload failed: {str(e)}"
        }), 500


@app.route("/results", methods=["GET"])
def results():
    """Handle query and return RAG results as HTML page"""
    try:
        query_text = request.args.get('q', '').strip()
        query_type = request.args.get('type', 'search').strip()
        
        if not query_text:
            return render_template("results.html", query=query_text, query_type=query_type, answer="No query provided", error=True)
        
        logger.info(f"Processing query: {query_text}")
        
        # Get answer from RAG pipeline
        answer = get_answer(query_text)
        
        # Render HTML page with results
        return render_template("results.html", query=query_text, query_type=query_type, answer=answer, error=False)
        
    except Exception as e:
        logger.error(f"Query error: {str(e)}")
        return render_template("results.html", query=query_text, query_type=query_type, answer=f"Error: {str(e)}", error=True)


if __name__ == "__main__":
    logger.info("Starting RAG application")
    app.run(debug=True, host="0.0.0.0", port=5000)
