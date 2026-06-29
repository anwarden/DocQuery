from ingestion import build_index, chunk_text, extract_text
from rag import generate_answer, retrieve

pdf_path = "example.pdf"

with open(pdf_path, "rb") as f:
    pdf_bytes = f.read()

text = extract_text(pdf_bytes)
chunks = chunk_text(text)
index, _ = build_index(chunks)

print("[OK] ingestor is working...")

query = "What is the main topic of this document"
hits = retrieve(
    query,
    index,
    chunks,
)

answer = generate_answer(query, hits)

print(f"\nQ: {query}")
print(f"A: {answer}")
