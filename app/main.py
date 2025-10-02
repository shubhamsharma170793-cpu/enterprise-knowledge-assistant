# app/main.py
import os
import streamlit as st
from ingestion import extract_text_from_pdf, chunk_text
from embeddings import generate_embeddings
from vectorstore import build_faiss_index
from qa import retrieve_chunks, generate_answer

# ---------------------------
# App Config
# ---------------------------
st.set_page_config(page_title="Enterprise Knowledge Assistant", layout="wide")
st.title("📄 Enterprise Knowledge Assistant")

st.write("Upload a PDF **or try the sample file** to explore this assistant.")

# ---------------------------
# Sidebar (Settings + Credits)
# ---------------------------
with st.sidebar:
    st.image("app/logo.png", width=150)   # ✅ Add logo here at top
    st.header("⚙️ Settings")
    chunk_size = st.slider("Chunk Size", 200, 1000, 500, step=100)
    top_k = st.slider("Top K Results", 1, 5, 3)

    st.markdown("---")
    st.caption("👨‍💻 Built by Shubham Sharma")
    st.markdown("[📂 GitHub Repo](https://github.com/shubhamsharma170793-cpu/enterprise-knowledge-assistant)")

# ---------------------------
# File Upload Section (Main area)
# ---------------------------
uploaded_file = st.file_uploader("📂 Upload your PDF", type="pdf")

if st.button("📘 Try with Sample PDF"):
    uploaded_file = open("app/sample.pdf", "rb")
    st.session_state["use_sample"] = True
else:
    st.session_state["use_sample"] = False

# ---------------------------
# File Handling + Processing
# ---------------------------
if uploaded_file:
    if st.session_state.get("use_sample", False):
        temp_path = os.path.join("app", "sample.pdf")
        st.info("Using **default sample.pdf** ✅")
    else:
        temp_path = os.path.join("temp.pdf")
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

    # Phase 2: Extract Text
    text = extract_text_from_pdf(temp_path)
    st.subheader("📑 Extracted Text (Preview)")
    st.write(text[:1000])

    # Phase 3: Chunking
    chunks = chunk_text(text, chunk_size=chunk_size)
    st.write(f"📦 Total Chunks Created: {len(chunks)}")

    st.subheader("🧩 Chunked Text (First 3 Chunks)")
    for i, chunk in enumerate(chunks[:3], start=1):
        st.write(f"**Chunk {i}:** {chunk}")

    # Phase 4: Embeddings
    embeddings = generate_embeddings(chunks)
    st.success(f"✅ Generated {len(embeddings)} embeddings.")

    # Phase 5: Vector Store (FAISS)
    index = build_faiss_index(embeddings)

    # Phase 6 & 7: Q&A
    user_query = st.text_input("🔍 Ask a question about the document:")

    if user_query:
        retrieved = retrieve_chunks(user_query, index, chunks, top_k=top_k)
        answer = generate_answer(user_query, retrieved)

        # Final Answer
        st.subheader("🤖 Assistant’s Answer")
        st.write(answer)

        # Supporting Chunks
        st.subheader("📄 Supporting Chunks")
        for i, r in enumerate(retrieved, start=1):
            st.write(f"**Chunk {i}:** {r}")
