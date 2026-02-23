from __future__ import annotations

from dataclasses import dataclass
import importlib
import os
from pathlib import Path
import re
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from services.retriever import prompt_retriever, render_advanced_rag_prompt_v1
from utility.logger import logger

load_dotenv()
os.environ['LANGGRAPH_PROJECT_NAME']='mutli-model-rag'  # Set a default project name for LangGraph

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


@dataclass
class GraphConfig:
    llm_model: str = os.getenv("GROQ_MODEL", "qwen/qwen3-32b")
    temperature: float = float(os.getenv("GROQ_TEMPERATURE", "0"))
    max_tokens: int = int(os.getenv("GROQ_MAX_TOKENS", "2000"))
    top_k: int = int(os.getenv("RAG_TOP_K", "4"))


def _strip_think_content(text: str) -> str:
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.IGNORECASE | re.DOTALL)
    cleaned = re.sub(r"<think>.*?(?=<think>|$)", "", cleaned, flags=re.IGNORECASE | re.DOTALL)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


@tool("retriever_tool")
def retriever_tool(query: str, k: int = 4) -> str:
    """Retrieve relevant context chunks from the vector store for a user query."""
    docs = prompt_retriever(query=query, k=k)
    if not docs:
        return ""

    chunks: list[str] = []
    for index, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "Unknown Source")
        content = doc.page_content.strip()
        chunks.append(f"[{index}] Source: {source}\n{content}")
    return "\n\n---\n\n".join(chunks)


def _init_checkpointer():
    backend = os.getenv("LANGGRAPH_MEMORY_BACKEND", "sqlite").strip().lower()

    if backend == "postgres":
        postgres_uri = os.getenv("LANGGRAPH_POSTGRES_URI")
        if not postgres_uri:
            raise ValueError("LANGGRAPH_POSTGRES_URI is required when LANGGRAPH_MEMORY_BACKEND=postgres")

        postgres_module = importlib.import_module("langgraph.checkpoint.postgres")
        AsyncPostgresSaver = getattr(postgres_module, "AsyncPostgresSaver")
        
        # Note: AsyncPostgresSaver should be used within an async context
        # For now, use a simpler synchronous approach or fallback to SQLite
        logger.warning("PostgreSQL checkpointer requires async context; falling back to SQLite")
        backend = "sqlite"

    # Use SQLite (default or fallback)
    sqlite_path = os.getenv("LANGGRAPH_SQLITE_PATH", "langgraph_memory.sqlite")
    sqlite_file = str(Path(sqlite_path).resolve())

    import sqlite3
    from langgraph.checkpoint.sqlite import SqliteSaver
    
    conn = sqlite3.connect(sqlite_file, check_same_thread=False)
    saver = SqliteSaver(conn=conn)
    logger.info(f"LangGraph SQLite checkpointer initialized: {sqlite_file}")
    return saver


class SimpleRAGLangGraphAgent:
    def __init__(self):
        self.config = GraphConfig()
        self.llm = ChatGroq(
            model=self.config.llm_model,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            timeout=None,
            max_retries=2,
        )
        self.graph = self._compile_graph()

    def _compile_graph(self):
        tool_node = ToolNode([retriever_tool])

        def retrieve_call_node(state: AgentState):
            user_message = next(
                msg.content for msg in reversed(state["messages"]) if isinstance(msg, HumanMessage)
            )
            return {
                "messages": [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "id": "retriever-call",
                                "name": "retriever_tool",
                                "args": {"query": user_message, "k": self.config.top_k},
                                "type": "tool_call",
                            }
                        ],
                    )
                ]
            }

        def answer_node(state: AgentState):
            user_message = next(
                msg.content for msg in reversed(state["messages"]) if isinstance(msg, HumanMessage)
            )
            context = ""
            for msg in reversed(state["messages"]):
                if isinstance(msg, ToolMessage):
                    context = str(msg.content)
                    break

            if not context:
                return {
                    "messages": [
                        AIMessage(
                            content="I couldn't find relevant information in the documents. Can I help you with something else?"
                        )
                    ]
                }

            prompt = render_advanced_rag_prompt_v1(user_request=user_message, context=context)
            response = self.llm.invoke([HumanMessage(content=prompt)])
            cleaned_response = _strip_think_content(str(response.content))

            return {"messages": [AIMessage(content=cleaned_response)]}

        graph_builder = StateGraph(AgentState)
        graph_builder.add_node("retrieve_call", retrieve_call_node)
        graph_builder.add_node("retrieve_tool", tool_node)
        graph_builder.add_node("answer", answer_node)
        graph_builder.add_edge(START, "retrieve_call")
        graph_builder.add_edge("retrieve_call", "retrieve_tool")
        graph_builder.add_edge("retrieve_tool", "answer")
        graph_builder.add_edge("answer", END)

        checkpointer = _init_checkpointer()
        return graph_builder.compile(checkpointer=checkpointer)

    def chat(self, message: str, thread_id: str) -> str:
        result = self.graph.invoke(
            {"messages": [HumanMessage(content=message)]},
            config={"configurable": {"thread_id": thread_id}},
        )
        return _strip_think_content(str(result["messages"][-1].content))

    def get_thread_messages(self, thread_id: str) -> list[dict]:
        state = self.graph.get_state(config={"configurable": {"thread_id": thread_id}})
        values = state.values or {}
        messages = values.get("messages", [])

        history: list[dict] = []
        for msg in messages:
            if isinstance(msg, HumanMessage):
                history.append({"role": "user", "content": str(msg.content)})
            elif isinstance(msg, AIMessage):
                if str(msg.content).strip():
                    history.append({"role": "assistant", "content": _strip_think_content(str(msg.content))})
        return history

    def generate_thread_title(self, first_user_message: str) -> str:
        prompt = (
            "Generate a short chat title (max 6 words) for this user query. "
            "Return only the title text, no quotes or punctuation extras.\n\n"
            f"Query: {first_user_message}"
        )
        response = self.llm.invoke([HumanMessage(content=prompt)])
        title = str(response.content).strip().strip('"').strip()
        if not title:
            return "New Chat"
        return title[:80]
