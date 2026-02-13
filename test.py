"""
Test script for RAG implementation with Case Study.pdf
Tests the complete pipeline: Ingestion -> Embedding -> Retrieval -> Generation
"""

import sys
from pathlib import Path
import os

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from services.ingestion import pdf_extractor
from services.embedding import Embedder
from services.faiss_db import FAISSDB
from services.gen import get_answer
from utility.logger import logger
from colorama import Fore, Style, init

# Initialize colorama for colored output
init(autoreset=True)

# ============================================================================
# TEST CONFIGURATION
# ============================================================================

PDF_PATH = Path(__file__).parent / "uploads_pdf" / "Case Study.pdf"
VECTOR_DB_PATH = Path(__file__).parent / "vector_db"

# Test queries to verify RAG functionality
TEST_QUERIES = [
    "What is the main topic of this case study?",
    "What are the key findings or conclusions?",
    "Who are the main stakeholders or companies involved?",
    "What challenges or problems are addressed?",
    "What solutions or recommendations are provided?",
]

# ============================================================================
# TEST FUNCTIONS
# ============================================================================

def test_pdf_loading():
    """Test 1: PDF Loading and Extraction"""
    print(f"\n{Fore.CYAN}{'='*70}")
    print(f"TEST 1: PDF LOADING AND EXTRACTION")
    print(f"{'='*70}{Style.RESET_ALL}")
    
    try:
        if not PDF_PATH.exists():
            print(f"{Fore.RED}✗ PDF not found at {PDF_PATH}{Style.RESET_ALL}")
            return None
        
        print(f"{Fore.GREEN}✓ PDF found at {PDF_PATH}{Style.RESET_ALL}")
        documents = pdf_extractor(str(PDF_PATH))
        
        print(f"{Fore.GREEN}✓ Successfully extracted {len(documents)} pages{Style.RESET_ALL}")
        
        for i, doc in enumerate(documents[:3], 1):
            print(f"\n{Fore.YELLOW}Page {i} Preview:{Style.RESET_ALL}")
            content_preview = doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
            print(f"  Content: {content_preview}")
            print(f"  Metadata: {doc.metadata}")
        
        if len(documents) > 3:
            print(f"\n{Fore.YELLOW}... and {len(documents) - 3} more pages{Style.RESET_ALL}")
        
        return documents
    
    except Exception as e:
        print(f"{Fore.RED}✗ PDF extraction failed: {str(e)}{Style.RESET_ALL}")
        logger.error(f"PDF extraction failed: {str(e)}")
        return None


def test_embedding_initialization():
    """Test 2: Embedding Model Initialization"""
    print(f"\n{Fore.CYAN}{'='*70}")
    print(f"TEST 2: EMBEDDING MODEL INITIALIZATION")
    print(f"{'='*70}{Style.RESET_ALL}")
    
    try:
        embedder = Embedder()
        print(f"{Fore.GREEN}✓ Embedding model initialized successfully{Style.RESET_ALL}")
        
        # Test embedding a sample text
        test_text = "This is a test document"
        embedding = embedder.embed_query(test_text)
        print(f"{Fore.GREEN}✓ Sample text embedded successfully{Style.RESET_ALL}")
        print(f"  Embedding dimension: {len(embedding)}")
        
        return embedder
    
    except Exception as e:
        print(f"{Fore.RED}✗ Embedding initialization failed: {str(e)}{Style.RESET_ALL}")
        logger.error(f"Embedding initialization failed: {str(e)}")
        return None


def test_vector_db_creation(documents, embedder):
    """Test 3: Vector Database Creation and Document Insertion"""
    print(f"\n{Fore.CYAN}{'='*70}")
    print(f"TEST 3: VECTOR DATABASE CREATION AND DOCUMENT INSERTION")
    print(f"{'='*70}{Style.RESET_ALL}")
    
    try:
        if not documents:
            print(f"{Fore.RED}✗ No documents to insert{Style.RESET_ALL}")
            return None
        
        vector_db = FAISSDB(embedder)
        print(f"{Fore.GREEN}✓ FAISS Vector DB initialized{Style.RESET_ALL}")
        
        vector_db.load_into_db(documents)
        print(f"{Fore.GREEN}✓ Successfully inserted {len(documents)} documents into vector DB{Style.RESET_ALL}")
        
        # Save vector DB
        vector_db.save_db(str(VECTOR_DB_PATH))
        print(f"{Fore.GREEN}✓ Vector DB saved to {VECTOR_DB_PATH}{Style.RESET_ALL}")
        
        return vector_db
    
    except Exception as e:
        print(f"{Fore.RED}✗ Vector DB creation/insertion failed: {str(e)}{Style.RESET_ALL}")
        logger.error(f"Vector DB operation failed: {str(e)}")
        return None


def test_retrieval(vector_db):
    """Test 4: Document Retrieval"""
    print(f"\n{Fore.CYAN}{'='*70}")
    print(f"TEST 4: DOCUMENT RETRIEVAL")
    print(f"{'='*70}{Style.RESET_ALL}")
    
    try:
        if not vector_db:
            print(f"{Fore.RED}✗ Vector DB not available{Style.RESET_ALL}")
            return False
        
        test_query = "What is the main topic?"
        results = vector_db.vector_store.similarity_search(query=test_query, k=3)
        
        print(f"{Fore.GREEN}✓ Successfully retrieved {len(results)} documents for query: '{test_query}'{Style.RESET_ALL}")
        
        for i, doc in enumerate(results, 1):
            print(f"\n{Fore.YELLOW}Retrieved Document {i}:{Style.RESET_ALL}")
            print(f"  Source: {doc.metadata.get('source', 'Unknown')}")
            print(f"  Page: {doc.metadata.get('page', 'N/A')}")
            content_preview = doc.page_content[:150] + "..." if len(doc.page_content) > 150 else doc.page_content
            print(f"  Content: {content_preview}")
        
        return True
    
    except Exception as e:
        print(f"{Fore.RED}✗ Retrieval test failed: {str(e)}{Style.RESET_ALL}")
        logger.error(f"Retrieval test failed: {str(e)}")
        return False


def test_rag_pipeline():
    """Test 5: Full RAG Pipeline (Retrieval + Generation)"""
    print(f"\n{Fore.CYAN}{'='*70}")
    print(f"TEST 5: FULL RAG PIPELINE (RETRIEVAL + GENERATION)")
    print(f"{'='*70}{Style.RESET_ALL}")
    
    try:
        results = []
        for i, query in enumerate(TEST_QUERIES, 1):
            print(f"\n{Fore.YELLOW}Query {i}: {query}{Style.RESET_ALL}")
            
            try:
                answer = get_answer(query)
                print(f"{Fore.GREEN}✓ Answer Generated:{Style.RESET_ALL}")
                
                # Print answer with word wrapping for readability
                answer_preview = answer[:300] + "..." if len(answer) > 300 else answer
                print(f"  {answer_preview}")
                
                results.append({
                    "query": query,
                    "answer": answer,
                    "status": "success"
                })
                
            except Exception as e:
                print(f"{Fore.RED}✗ Failed to generate answer: {str(e)}{Style.RESET_ALL}")
                results.append({
                    "query": query,
                    "answer": None,
                    "status": "failed",
                    "error": str(e)
                })
        
        print(f"\n{Fore.GREEN}✓ RAG Pipeline test completed{Style.RESET_ALL}")
        print(f"  Successful queries: {sum(1 for r in results if r['status'] == 'success')}/{len(results)}")
        
        return results
    
    except Exception as e:
        print(f"{Fore.RED}✗ RAG pipeline test failed: {str(e)}{Style.RESET_ALL}")
        logger.error(f"RAG pipeline test failed: {str(e)}")
        return None


def print_test_summary(test_results):
    """Print summary of all tests"""
    print(f"\n{Fore.CYAN}{'='*70}")
    print(f"TEST SUMMARY")
    print(f"{'='*70}{Style.RESET_ALL}")
    
    if test_results:
        successful = sum(1 for r in test_results if r['status'] == 'success')
        total = len(test_results)
        success_rate = (successful / total) * 100
        
        print(f"\n{Fore.GREEN}RAG Pipeline Results:{Style.RESET_ALL}")
        print(f"  Total Queries: {total}")
        print(f"  Successful: {successful}")
        print(f"  Failed: {total - successful}")
        print(f"  Success Rate: {success_rate:.1f}%")
        
        if successful == total:
            print(f"\n{Fore.GREEN}✓ All tests passed! RAG implementation is working correctly.{Style.RESET_ALL}")
        else:
            print(f"\n{Fore.YELLOW}⚠ Some tests failed. Review the errors above.{Style.RESET_ALL}")
    else:
        print(f"{Fore.RED}✗ No results to summarize{Style.RESET_ALL}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print(f"\n{Fore.MAGENTA}")
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║         RAG TEST SUITE - Case Study PDF                          ║")
    print("║  Testing: Ingestion -> Embedding -> Retrieval -> Generation      ║")
    print("╚══════════════════════════════════════════════════════════════════╝")
    print(f"{Style.RESET_ALL}")
    
    # Run tests
    print(f"{Fore.CYAN}Starting test execution...{Style.RESET_ALL}")
    
    # Test 1: PDF Loading
    documents = test_pdf_loading()
    if not documents:
        print(f"\n{Fore.RED}Tests stopped: PDF loading failed{Style.RESET_ALL}")
        sys.exit(1)
    
    # Test 2: Embedding Initialization
    embedder = test_embedding_initialization()
    if not embedder:
        print(f"\n{Fore.RED}Tests stopped: Embedding initialization failed{Style.RESET_ALL}")
        sys.exit(1)
    
    # Test 3: Vector DB Creation
    vector_db = test_vector_db_creation(documents, embedder)
    if not vector_db:
        print(f"\n{Fore.RED}Tests stopped: Vector DB creation failed{Style.RESET_ALL}")
        sys.exit(1)
    
    # UPDATE GLOBAL VECTOR STORE IN RETRIEVER WITH OUR TEST DATA
    print(f"\n{Fore.CYAN}Syncing vector store with retriever module...{Style.RESET_ALL}")
    from services import retriever
    retriever.vector_store = vector_db
    print(f"{Fore.GREEN}✓ Vector store synced successfully{Style.RESET_ALL}")
    
    # Test 4: Retrieval
    retrieval_success = test_retrieval(vector_db)
    if not retrieval_success:
        print(f"\n{Fore.RED}Tests stopped: Retrieval test failed{Style.RESET_ALL}")
        sys.exit(1)
    
    # Test 5: Full RAG Pipeline
    rag_results = test_rag_pipeline()
    
    # Print Summary
    print_test_summary(rag_results)
    
    print(f"\n{Fore.CYAN}Test execution completed!{Style.RESET_ALL}\n")
