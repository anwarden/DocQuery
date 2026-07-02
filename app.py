import streamlit as st

from ingestion import build_index, chunk_text, extract_text
from rag import generate_answer, retrieve

st.set_page_config(page_title="RAG Study assistant", page_icon="book")
st.title("AI Study Assistant")
st.caption("Ask questions about your uploaded PDF document.")

#   Side bar : Upload + Indexing
with st.sidebar:
    st.header("Upload Document")
    uploaded = st.file_uploader("PDF only (text-based)", type=["pdf"])

    if uploaded:
        # Re-Index if a new file is added.
        if (
            "filename" not in st.session_state
            or st.session_state["filename"] != uploaded.name
        ):
            with st.spinner("Indexing document, please wait"):
                pdf_bytes = uploaded.read()
                text = extract_text(pdf_bytes)

                if not text.strip():
                    pass

                chunks = chunk_text(text)
                index, _ = build_index(chunks)

                # Persist new data in the Stramlit session_state
                st.session_state["index"] = index
                st.session_state["chunks"] = chunks
                st.session_state["filename"] = uploaded.name

            st.success(f"Indexed {len(st.session_state['chunks'])} chunks.")

    if "index" in st.session_state:
        st.info(f"Active document: {st.session_state['filename']}")
        if st.button("Clear document"):
            for key in ["index", "chunks", "filename"]:
                st.session_state.pop(key, None)
            st.rerun()


# Main area : Chat

if "index" not in st.session_state:
    st.info("Upload a PDF in the sidebar to get started.")
else:
    query = st.chat_input("Ask a question about your document...")
    if query:
        with st.chat_message("user"):
            st.write(query)
        with st.chat_message("Assistant"):
            with st.spinner("Reading source document, answering ..."):
                hits = retrieve(
                    query, st.session_state["index"], st.session_state["chunks"], k=3
                )

                answer = generate_answer(query, hits)
                st.write(answer)

                with st.expander("View retrieved source chunks"):
                    for i, chunk in enumerate(hits, 1):
                        st.markdown(f"**chunk {i}: {chunk[:300]}...")
