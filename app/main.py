# app/main.py
import os
import streamlit as st
from ingestion import extract_text_from_pdf, chunk_text
from embeddings import generate_embeddings
from vectorstore import build_faiss_index
from qa import retrieve_chunks, generate_answer

# ---------------------------
# Page Config
# ---------------------------
st.set_page_config(page_title="Enterprise Knowledge Assistant", layout="wide")

# ---------------------------
# Sidebar (Navigation)
# ---------------------------
st.sidebar.title("⚡ Enterprise Knowledge Assistant")
st.sidebar.markdown("Upload a PDF or try the sample file to see the assistant in action.")

# Upload option in sidebar
uploaded_file = st.sidebar.file_uploader("📂 Upload your PDF", type="pdf")

# ✅ Sample PDF option
if st.sidebar.button("📘 Try with Sample PDF"):
    uploaded_file = open("app/sample.pdf", "rb")  # sample.pdf must exist
    st.session_state["use_sample"] = True
else:
    st.session_state["use_sample"] = False

# ---------------------------
# Main App Area
# ---------------------------
st.title("📄 Enterprise Knowledge Assistant (MVP)")
st.markdown("Ask questions about your documents with **AI-powered answers**.")

# ---------------------------
# File Handling
# ---------------------------
if uploaded_file:
    # If user chose sample → use directly, else save uploaded
    if st.session_state.get("use_sample", False):
        temp_path = os.path.join("app", "sample.pdf")
        st.info("✅ Using **default sample.pdf**")
    else:
        temp_path = os.path.join("temp.pdf")
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success("✅ PDF uploaded successfully!")

    # ---------------------------
    # Phase 2: Extract Text
    # ---------------------------
    text = extract_text_from_pdf(temp_path)

    st.subheader("📑 Extracted Text (Preview)")
    st.write(text[:1000])   # show first 1000 chars only

    # ---------------------------
    # Phase 3: Chunking
    # ---------------------------
    chunks = chunk_text(text)
    st.write(f"📦 Total Chunks Created: **{len(chunks)}**")

    st.subheader("🧩 Chunked Text (First 3 Chunks)")
    for i, chunk in enumerate(chunks[:3], start=1):
        st.markdown(f"**Chunk {i}:** {chunk}")

    # ---------------------------
    # Phase 4: Embeddings
    # ---------------------------
    embeddings = generate_embeddings(chunks)
    st.success(f"✅ Generated {len(embeddings)} embeddings.")
