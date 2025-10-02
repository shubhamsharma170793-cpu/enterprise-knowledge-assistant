# app/qa.py
from sentence_transformers import SentenceTransformer
from vectorstore import search_faiss
from transformers import pipeline

# Load embedding model (for query encoding)
_query_model = SentenceTransformer("all-MiniLM-L6-v2")

# Load Hugging Face generative model (FLAN-T5-small is free & CPU friendly)
_answer_model = pipeline("text2text-generation", model="google/flan-t5-small")


def retrieve_chunks(query: str, index, chunks: list, top_k: int = 3):
    """Encode query and return top_k matching chunks from FAISS index."""
    query_embedding = _query_model.encode([query], convert_to_numpy=True)[0]
    return search_faiss(query_embedding, index, chunks, top_k)


def generate_answer(query: str, retrieved_chunks: list) -> str:
    """Generate a natural language answer using Hugging Face model."""
    if not retrieved_chunks:
        return "Sorry, I could not find relevant information in the document."

    # Combine retrieved chunks into a single context
    context = " ".join(retrieved_chunks)

    # ✅ Stronger prompt to guide the model
    prompt = f"""
You are a helpful assistant. 
Read the context carefully and answer the question with clear, step-by-step instructions. 
Do not just repeat the context; explain in your own words.

Context:
{context}

Question:
{query}

Answer (in numbered steps):
"""

    # Run the Hugging Face pipeline
    result = _answer_model(prompt, max_length=300, clean_up_tokenization_spaces=True)
    return result[0]["generated_text"].strip()
