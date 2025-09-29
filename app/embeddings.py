from sentence_transformers import SentenceTransformer   # 📌 library
import numpy as np                                      # 📌 library (for math)

# 📌 variable: load a free embedding model (runs locally in Codespaces)
model = SentenceTransformer('all-MiniLM-L6-v2')

def generate_embeddings(chunks: list) -> list:
    """
    📌 Function: Generate embeddings for a list of text chunks using Hugging Face.

    Args:
        chunks (list): List of text chunks.

    Returns:
        list: List of embedding vectors (one per chunk).
    """
    embeddings = model.encode(chunks, convert_to_numpy=True)   # numpy array
    return embeddings.tolist()   # convert to plain Python list


# ----------------------------
# OLD OPENAI EMBEDDINGS CODE
# (kept for reference only)
# ----------------------------

# from openai import OpenAI   # 📌 library + class
# client = OpenAI()           # 📌 variable (needs API key in env)

# def generate_embeddings(chunks: list, model: str = "text-embedding-3-small") -> list:
#     embeddings = []   # 📌 variable: holds all vectors
#     for chunk in chunks:
#         response = client.embeddings.create(   # 📌 function (method) call
#             input=chunk,
#             model=model
#         )
#         embeddings.append(response.data[0].embedding)  # vector = list of floats
#     return embeddings
