from ingestion import build_index, chunk_text, extract_text
from rag import generate_answer, retrieve

print("=== Ingestion ===")
pdf_path = "example.pdf"

with open(pdf_path, "rb") as f:
    pdf_bytes = f.read()

text = extract_text(pdf_bytes)
chunks = chunk_text(text)
index, _ = build_index(chunks)

print("[OK] ingestor is working...")

query = """What is the main topic of this document ?"""

print("\n=== Query ===")
hits = retrieve(
    query,
    index,
    chunks,
)

answer = generate_answer(query, hits)

print("\n=== Result ===")
print(f"Q: {query}")
print(f"A: {answer}")

print("\nMost relevant chunks:")

for hit in hits:
    print(f" -> {hit[:300]}")
