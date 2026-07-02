import faiss
import fitz
import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer


@st.cache_resource
def _load_model() -> SentenceTransformer:
    return SentenceTransformer("all-MiniLM-L6-v2")


MODEL = _load_model()


def extract_text(pdf_bytes: bytes) -> str:
    """
    Extract raw text from an in-memory PDF
    """
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    text = "\n".join(str(page.get_text()) for page in doc)
    print(f"[ingestion] extracted {len(text)} chars from PDF")
    return text


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """
    Sliding-window world-level chunking with overlap
    """
    chunks, i = [], 0
    words = text.split()

    while i < len(words):
        chunk = " ".join(words[i : i + chunk_size])
        chunks.append(chunk)
        i += chunk_size - overlap

    chunks = [c for c in chunks if len(c.strip()) > 20]
    print(
        f"[ingestion] created {len(chunks)} chunks (size={chunk_size}, overlap={overlap})"
    )
    return chunks


def build_index(chunks: list[str]):
    """
    Embed all chunks and an in-memory FAISS flat L2 index
    """
    embeddings = MODEL.encode(chunks, show_progress_bar=False)
    embeddings = np.array(embeddings, dtype="float32")
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    print(f"[ingestion] indexed {len(chunks)} vectors (dim={embeddings.shape[1]})")
    return index, embeddings
