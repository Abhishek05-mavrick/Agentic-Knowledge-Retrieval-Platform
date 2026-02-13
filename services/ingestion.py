from typing import List, Union
from pathlib import Path
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import requests
from bs4 import BeautifulSoup
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, YoutubeLoader
from groq import Groq

from utility.logger import logger
from utility.error_handling import customException


def clean_text(text: str) -> str:
    """Keep only non-empty lines with more than 3 characters"""
    lines = [line.strip() for line in text.splitlines() if len(line.strip()) > 3]
    return "\n".join(lines)


def yt_extractor_robust(
    url: str,
    add_video_info: bool = True,
) -> List[Document]:
    try:
        logger.info("starting yt extraction")
        loader = YoutubeLoader.from_youtube_url(
            url,
            add_video_info=add_video_info
        )
        docs = loader.load()
         

        
        for doc in docs:
            doc.page_content = clean_text(doc.page_content)
            doc.metadata["type"]="youtube"
            # if len(doc.page_content)<5:
            #     doc.metadata["tarnscript_Avail"]="False" 

        return docs
        
    except Exception as e:
        logger.error(f"yt extraction failed for {url}: {str(e)}")
        return [Document(
            page_content=f"Error extracting YouTube transcript: {str(e)}",
            metadata={"source": url, "error": True}
        )]


def pdf_extractor(file_path: Union[str, Path]) -> List[Document]:
    try:
        logger.info(f"starting pdf extraction: {file_path}")
        loader = PyPDFLoader(str(file_path))
        docs = loader.load()

        
        for i, doc in enumerate(docs, 1):
            if "page" not in doc.metadata:
                doc.metadata["page"] = i
            if "source" not in doc.metadata:
                doc.metadata["source"] = str(file_path)
            if "type" not in doc.metadata:
                doc.metadata["type"]="pdf"

        
        for doc in docs:
            doc.page_content = clean_text(doc.page_content)

        return docs

    except Exception as e:
        logger.error(f"pdf extraction failed for {file_path}: {str(e)}")
        raise customException(f"PDF loading failed: {str(e)}") from e


def web_extractor(url: str) -> List[Document]:
    try:
        logger.info(f"starting web extraction: {url}")
        r = requests.get(url, timeout=12)
        r.raise_for_status()

        soup = BeautifulSoup(r.text, "html.parser")
        text = soup.get_text(separator="\n", strip=True)
        text = clean_text(text) 

        return [Document(
            page_content=text,
            metadata={
                "source": url,
                "title": soup.title.string.strip() if soup.title else None,
                "type":"web"
            }
        )]

    except Exception as e:
        logger.error(f"web extraction failed for {url}: {str(e)}")
        raise customException(f"Web page loading failed: {str(e)}") from e


def audio_extractor(file_path: Union[str, Path]) -> List[Document]:
    try:
        logger.info(f"starting audio extraction: {file_path}")
        client = Groq()

        full_path = os.path.abspath(file_path)

        with open(full_path, "rb") as file:
            transcription = client.audio.transcriptions.create(
                file=(os.path.basename(full_path), file),
                model="whisper-large-v3",
                temperature=0,
                response_format="text",
            )

        
        content = clean_text(transcription)

        return [Document(
            page_content=content,
            metadata={
                "source": str(file_path),
                "type": "audio",
                "transcription_model": "groq/whisper-large-v3"
            }
        )]

    except Exception as e:
        logger.error(f"audio extraction failed for {file_path}: {str(e)}")
        raise customException(f"Audio transcription failed: {str(e)}") from e