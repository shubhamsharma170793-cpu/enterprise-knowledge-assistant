# app/main.py
import os
import streamlit as st
from ingestion import extract_text_from_pdf, chunk_text
from embeddings import generate_embeddings
from vectorstore import build_faiss_index
from qa import retrieve_chunks, generate_answer

# ---------------------------
# App Title
# ---------------------------
st.title("📄 Enterprise Knowledge Assistant (MVP)")
st.write("Upload a PDF, process it, and ask questions about its content.")

# ---------------------------
# 🎨 Sidebar
# ---------------------------
with st.sidebar:
    st.header("⚙️ Settings")
    # Control chunk size for splitting text
    chunk_size = st.slider("Chunk Size", 200, 1000, 500, step=100)
    # Control how many chunks to retrieve
    top_k = st.slider("Top K Results", 1, 5, 3)

    st.markdown("---")
    st.caption("👨‍💻 Built by Shubham Sharma")
    st.markdown("[🌐 GitHub Repo](https://github.com/shubhamsharma170793-cpu/enterprise-knowledge-assistant)")

# ---------------------------
# 📂 File Upload
# ---------------------------
uploaded_file = st.file_uploader("📂 Upload a PDF document", type="pdf")

if uploaded_file:
    # Save uploaded file temporarily
    temp_path = os.path.join("temp.pdf")
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # ---------------------------
    # Phase 2: Extract Text
    # ---------------------------
    text = extract_text_from_pdf(temp_path)
    st.success("✅ Text extracted successfully!")

    st.subheader("📖 Extracted Text (Preview)")
    st.write(text[:1000])   # preview first 1000 characters

    # ---------------------------
    # Phase 3: Chunking
    # ---------------------------
    chunks = chunk_text(text, chunk_size=chunk_size)
    st.info(f"📦 Created {len(chunks)} chunks")

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
    st.info("📚 Vector store ready!")

    # ---------------------------
    # Phase 6–7: Query + Answer
    # ---------------------------
    st.subheader("🔍 Ask a Question")
    user_query = st.text_input("Type your question here:")

    if user_query:
        with st.spinner("🤖 Thinking..."):
            # Retrieve relevant chunks
            retrieved = retrieve_chunks(user_query, index, chunks, top_k=top_k)
            # Generate final answer
            answer = generate_answer(user_query, retrieved)

        # Show final answer
        st.markdown("### 🤖 Assistant’s Answer")
        st.success(answer)

        # Supporting chunks
        with st.expander("📄 Supporting Chunks"):
            for i, r in enumerate(retrieved, start=1):
                st.markdown(f"**Chunk {i}:** {r}")
