import os

import numpy as np
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from ingestion import MODEL

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.environ["GOOGLE_API_KEY"],
    max_tokens=4096,
)

SYSTEM_PROMPT = """You are a helpful study assistant.
Answer ONLY based on the provided context.
If the answer is not in the context, say exactly:
"I could not find this information in the document."
"""


def retrieve(query: str, index, chunks: list[str], k: int = 3) -> list[str]:
    """
    Embed query and return k nearest chunks from FAISS index.
    """
    print(f"[rag] retrieving top-{k} chunks for query: {query[:60]!r}")
    q_vec = MODEL.encode([query], show_progress_bar=False)
    q_vec = np.array(q_vec, dtype="float32")
    _, indices = index.search(q_vec, k)
    result = [chunks[i] for i in indices[0] if i < len(chunks)]
    print(f"[rag] retrieved {len(result)} chunks")
    return result


def generate_answer(query: str, context_chunks: list[str]) -> str:
    """
    Build prompt with retrieved context and invoke via LangChain
    """
    context = "\n\n---\n\n".join(context_chunks)
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f"Context: \n{context}\n\nQuestion: {query}"),
    ]
    print("[rag] calling Gemini for final answer...")
    response = llm.invoke(messages)
    print("[rag] answer generated")
    return response.content if isinstance(response.content, str) else str(response.content)
