import streamlit as st
from ingestion import extract_text_from_pdf
import os

# App title
st.title("📄 Enterprise Knowledge Assistant (MVP)")

st.write("Upload a PDF to extract and view its text")

# Upload button
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file:
    # Save uploaded file temporarily
    temp_path = os.path.join("temp.pdf")
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Extract text
    text = extract_text_from_pdf(temp_path)

    # Show extracted text
    st.subheader("Extracted Text:")
    st.write(text[:1000])  # show first 1000 characters for readability

