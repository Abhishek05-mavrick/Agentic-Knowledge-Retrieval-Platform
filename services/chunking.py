import sys
import os
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from utility.logger import logger
from utility.error_handling import customException
from dotenv import load_dotenv
load_dotenv()


def re_te_sp(documents: list[Document]) -> list[Document]:
    try:
        logger.info("chunking started")
        text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=120,
        length_function=len,
        is_separator_regex=False,
        )
        texts = text_splitter.split_documents(documents)
        return texts
    except Exception as e:
        logger.info("chunking error has occured")
        raise customException(e,sys)

