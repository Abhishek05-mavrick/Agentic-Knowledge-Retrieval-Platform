from typing import List
import torch
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from utility.logger import logger
from utility.error_handling import customException


class Embedder:
    """
    Stateless embedder wrapper.
    Owns the model, exposes embedding generation.
    """

    def __init__(self):
        try:
            logger.info("initializing embedding model")

            self.model = HuggingFaceEmbeddings(
                model_name="all-MiniLM-L6-v2",
                model_kwargs={
                    "device": "cuda" if torch.cuda.is_available() else "cpu"
                },
                encode_kwargs={
                    "normalize_embeddings": True
                }
            )

        except Exception as e:
            logger.error("failed to initialize embedding model")
            raise customException(e, sys)

    def embed_query(self, text: str) -> List[float]:
        """
        Embed a single query text.
        """
        try:
            return self.model.embed_query(text)
        except Exception as e:
            logger.error(f"failed to embed query: {str(e)}")
            raise customException(e, sys)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed multiple documents/texts.
        """
        try:
            logger.info("embedding generation started")
            embeddings = self.model.embed_documents(texts)
            return embeddings
        except Exception as e:
            logger.error("embedding generation failed")
            raise customException(e, sys)


