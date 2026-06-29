import os

import numpy as np
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage

from ingestion import MODEL

load_dotenv()

llm = ChatAnthropic(
    model="claude-sonnet-4-6", api_key=os.environ["ANTHROPIC_API_KEY"], max_tokens=1024
)

SYSTEM_PROMPT = """You are a helpful study assistant.
18 Answer ONLY based on the provided context.
19 If the answer is not in the context, say exactly:
20 "I could not find this information in the document."
21 Be concise and precise."""


def retrieve(query: str, index, chunks: list[str], k: int = 3) -> list[str]:
    """
    Enbed query and return k nearest chunks from FAISS index.
    """
    q_vec = MODEL.encode([query], show_progress_bar=False)
    q_vec = np.array(q_vec, dtype="float32")
    _, indices = index.search(q_vec, k)
    return [chunks[i] for i in indices[0] if i < len(chunks)]


def generate_answer(query: str, context_chunks: list[str]) -> str:
    """
    Build prompt with retrieved context and invoke via LangChain
    """
    context = "\n\n---\n\n".join(context_chunks)
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f"Context: \n{context}\n\nQuestion: {query}"),
    ]
    response = llm.invoke(messages)
    return response.content
