# app/main.py
import os, json
import streamlit as st
from ingestion import extract_text_from_pdf, chunk_text
from embeddings import generate_embeddings
from vectorstore import build_faiss_index
from qa import retrieve_chunks, generate_answer
import faiss

# ---------------------------
# App Config
# ---------------------------
st.set_page_config(page_title="Enterprise Knowledge Assistant", layout="wide")
st.title("📄 Enterprise Knowledge Assistant")
st.caption("Select a document from the library or upload your own, then ask questions.")

# ---------------------------
# Sidebar (Settings + Docs + Credits)
# ---------------------------
with st.sidebar:
    st.image("app/logo.png", width=150)
    st.header("⚙️ Settings")
    chunk_size = st.slider("Chunk Size", 200, 1000, 500, step=100)
    top_k = st.slider("Top K Results", 1, 5, 3)

    st.markdown("---")
    st.header("📚 Document Library")
    # 🆕 Default is "-- Select --"
    doc_choice = st.radio(
        "Choose a document:",
        ["-- Select --", "Sample PDF", "Upload Custom PDF"],
        index=0
    )

    st.markdown("---")
    st.caption("👨‍💻 Built by Shubham Sharma")
    st.markdown("[📂 GitHub Repo](https://github.com/shubhamsharma170793-cpu/enterprise-knowledge-assistant)")

# ---------------------------
# Document Handling
# ---------------------------
text, chunks, index = None, None, None

if doc_choice == "-- Select --":
    st.info("⬅️ Please choose **Sample PDF** or **Upload Custom PDF** from the sidebar to continue.")

elif doc_choice == "Sample PDF":
    temp_path = os.path.join("app", "sample.pdf")
    st.success("📘 Sample PDF selected")
    text = extract_text_from_pdf(temp_path)
    chunks = chunk_text(text, chunk_size=chunk_size)
    embeddings = generate_embeddings(chunks)
    index = build_faiss_index(embeddings)

elif doc_choice == "Upload Custom PDF":
    uploaded_file = st.file_uploader("📂 Upload your PDF", type="pdf")
    if uploaded_file:
        temp_path = "temp.pdf"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success("✅ Document uploaded and processed!")

        text = extract_text_from_pdf(temp_path)
        chunks = chunk_text(text, chunk_size=chunk_size)
        embeddings = generate_embeddings(chunks)
        index = build_faiss_index(embeddings)

# ---------------------------
# Document Preview
# ---------------------------
if chunks:
    st.subheader("📑 Document Preview")
    st.text_area("Extracted text (first 1000 chars)", text[:1000], height=150)
    st.caption(f"📦 {len(chunks)} chunks created")

# ---------------------------
# Query Section
# ---------------------------
if index and chunks:
    st.markdown("---")
    st.subheader("🤖 Ask a Question")

    user_query = st.text_input("🔍 Your question about the document:")
    if user_query:
        retrieved = retrieve_chunks(user_query, index, chunks, top_k=top_k)
        answer = generate_answer(user_query, retrieved)

        st.markdown("### ✅ Assistant’s Answer")
        st.write(answer)

        with st.expander("📄 Supporting Chunks"):
            for i, r in enumerate(retrieved, start=1):
                st.markdown(f"**Chunk {i}:** {r}")
