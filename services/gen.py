import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import HumanMessagePromptTemplate
from utility.logger import logger
from langchain_core.messages import HumanMessage, SystemMessage
from utility.error_handling import customException
from services.retriever import (
    prompt_retriever,
    render_advanced_rag_prompt_v1
)
load_dotenv()


llm = ChatGroq(
model="qwen/qwen3-32b",
temperature=0,
max_tokens=2000,
reasoning_format="hidden",
timeout=None,
max_retries=2,
)
class MyAgent:
    def llm_init(self):
        self.llm=ChatGroq(
    model="qwen/qwen3-32b",
    temperature=0,
    max_tokens=2000,
    reasoning_format="hidden",
    timeout=None,
    max_retries=2,
    )


def get_answer(query: str) -> str:
    try:
        logger.info("Starting RAG generation flow")
        
        # 1. RETRIEVAL
        documents = prompt_retriever(query=query)
        
        if not documents:
            logger.info("No documents found")
            return "I couldn't find relevant information in the documents. Can I help you with something else?"
        
        logger.info(f"Documents retrieved: {len(documents)}")
        
        # 2. CONTEXT PREPARATION
        context_parts = []
        for i, doc in enumerate(documents):
            source = doc.metadata.get("source", "Unknown Source")
            content = doc.page_content.strip()
            context_parts.append(f"[{i+1}] Source: {source}\n{content}")
        
        final_context = "\n\n---\n\n".join(context_parts)
        
        # 3. PROMPT RENDERING
        prompt = render_advanced_rag_prompt_v1(
            user_request=query,
            context=final_context
        )
        
        # 4. LLM INVOCATION
        logger.info("Invoking LLM")
        response = llm.invoke([HumanMessage(content=prompt)])
        
        # Ensure we return a string
        if hasattr(response, 'content'):
            return str(response.content)
        else:
            return str(response)
            
    except Exception as e:
        logger.error(f"Generation failed: {str(e)}")
        raise customException(e, sys)