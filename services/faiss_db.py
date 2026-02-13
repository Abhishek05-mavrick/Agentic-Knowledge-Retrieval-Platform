from typing import List
import torch
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from langchain_core.documents import Document
from utility.logger import logger
from utility.error_handling import customException
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from services.embedding import Embedder
embedder=Embedder()
##note to myslef iam writing this to let you know i changed th architecture a bit now instead of documents getting embedded in embedding.py we will intilise it there and use the method here in faiss_db.py to decrase overhead 
##but rember if i want change the embedding model you must tweak the faiss as well :)

class FAISSDB():
    def __init__(self,embedder):
# IMPORTANT:
# - Embeddings: all-MiniLM-L6-v2
# - Normalized: True
# - Index: L2 (acts as cosine on normalized vectors)
        try:
            logger.info("the FAISS DB started")
            dummy="intitation"
            length=len(embedder.embed_query(dummy))
            indexing=faiss.IndexFlatL2(length)
            self.vector_store = FAISS(
            index=indexing,
            embedding_function=embedder.model,
            docstore=InMemoryDocstore(),
            index_to_docstore_id={},)
        except Exception as e:
            logger.info("error while intitialising to vector db")
            raise customException(e,sys)

    def load_into_db(self,documents:list[Document]):
        try:
            logger.info("inserting into db")
            self.vector_store.add_documents(documents)
            logger.info("data added to db sucessfully")

        except Exception as e:
            logger.info("failed to insert into db")
            raise customException(e,sys)
        
    def save_db(self, folder_path="vector_db"):
        try:
            self.vector_store.save_local(folder_path)
            logger.info(f"DB saved locally to {folder_path}")
        except Exception as e:
            logger.info("failed to  save DB locally")
            raise customException(e,sys)

    def load_db(self, folder_path="vector_db"):
        try:
            logger.info(f"Loading DB from {folder_path}")
            if Path(folder_path).exists():
                self.vector_store = FAISS.load_local(
                    folder_path,
                    embedder.model,
                    allow_dangerous_deserialization=True
                )
                logger.info(f"DB loaded successfully from {folder_path}")
                return True
            else:
                logger.info(f"DB folder not found at {folder_path}")
                return False
        except Exception as e:
            logger.info(f"failed to load DB from {folder_path}: {str(e)}")
            return False
