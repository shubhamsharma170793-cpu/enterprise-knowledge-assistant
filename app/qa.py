# app/qa.py
from sentence_transformers import SentenceTransformer
from vector import search_faiss
from transformers import pipeline

# Load embedding model (for query encoding)
_query_model = SentenceTransformer("all-MiniLM-L6-v2")

# Load Hugging Face generative model (FLAN-T5-small is lightweight, CPU friendly)
_answer_model = pipeline("text2text-generation", model="google/flan-t5-small")


def retrieve_chunks(query: str, index, chunks: list, top_k: int = 3):
    """Encode query and return top_k matching chunks from FAISS index."""
    query_embedding = _query_model.encode([query], convert_to_numpy=True)[0]
    return search_faiss(query_embedding, index, chunks, top_k)


def generate_answer(query: str, retrieved_chunks: list) -> str:
    """Generate a natural language answer using Hugging Face model."""
    if not retrieved_chunks:
        return "Sorry, I could not find relevant information in the document."

    # Combine retrieved chunks into a context
    context = " ".join(retrieved_chunks)

    # Build prompt
    prompt = f"Answer the following question based on the context:\n\nContext: {context}\n\nQuestion: {query}\n\nAnswer:"

    # Run the Hugging Face pipeline
    result = _answer_model(prompt, max_length=200, clean_up_tokenization_spaces=True)
    return result[0]["generated_text"].strip()
