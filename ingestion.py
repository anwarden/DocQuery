import faiss
import fitz
import numpy as np
from sentence_transformers import SentenceTransformer

MODEL = SentenceTransformer("all-MiniLM-L6-v2")


def extract_text(pdf_bytes: bytes) -> str:
    """
    Extract raw text from an in-memory PDF
    """
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    return "\n".join(page.get_text() for page in doc)


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

    return [c for c in chunks if len(c.strip()) > 20]


def build_index(chunks: list[str]):
    """
    Embed all chunks and an in-memory FAISS flat L2 index
    """
    embeddings = MODEL.encode(chunks, show_progress_bar=False)
    embeddings = np.array(embeddings, dtype="float32")
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index, embeddings
