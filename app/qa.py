# app/qa.py
from sentence_transformers import SentenceTransformer
from vector import search_faiss

_query_model = SentenceTransformer("all-MiniLM-L6-v2")

def retrieve_chunks(query: str, index, chunks: list, top_k: int = 3):
    """Encode user query and return top_k matching chunks."""
    query_embedding = _query_model.encode([query], convert_to_numpy=True)[0]
    return search_faiss(query_embedding, index, chunks, top_k)

