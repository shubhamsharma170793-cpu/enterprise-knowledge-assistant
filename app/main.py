import os
import streamlit as st
from ingestion import extract_text_from_pdf, chunk_text
from embeddings import generate_embeddings
from vectorstore import build_faiss_index
from qa import retrieve_chunks, generate_answer   # <-- NEW


# ---------------------------
# App Title
# ---------------------------
st.title("📄 Enterprise Knowledge Assistant (MVP)")

st.write("Upload a PDF, process it, and ask questions about its content.")

# ---------------------------
# File Upload
# ---------------------------
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file:
    # Save uploaded file temporarily
    temp_path = os.path.join("temp.pdf")
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # ---------------------------
    # Phase 2: Extract Text
    # ---------------------------
    text = extract_text_from_pdf(temp_path)

    st.subheader("📑 Extracted Text (Preview)")
    st.write(text[:1000])   # preview first 1000 characters

    # ---------------------------
    # Phase 3: Chunking
    # ---------------------------
    chunks = chunk_text(text)
    st.write(f"📦 Total Chunks Created: {len(chunks)}")

    st.subheader("🧩 Chunked Text (First 3 Chunks)")
    for i, chunk in enumerate(chunks[:3], start=1):
        st.write(f"**Chunk {i}:** {chunk}")

    # ---------------------------
    # Phase 4: Embeddings
    # ---------------------------
    embeddings = generate_embeddings(chunks)
    st.success(f"✅ Generated {len(embeddings)} embeddings.")

    # ---------------------------
    # Phase 5: Vector Store (FAISS)
    # ---------------------------
    index = build_faiss_index(embeddings)


    # Input for user queries
    user_query = st.text_input("🔍 Ask a question about the document:")

    if user_query:
        # Phase 6: Retrieve top chunks
        retrieved = retrieve_chunks(user_query, index, chunks, top_k=3)

        # Phase 7: Generate answer using Hugging Face model
        answer = generate_answer(user_query, retrieved)

        # Show final answer
        st.subheader("🤖 Answer")
        st.write(answer)

        # Optional: show supporting chunks
        st.subheader("📄 Supporting Chunks")
        for i, r in enumerate(retrieved, start=1):
            st.write(f"**Chunk {i}:** {r}")
