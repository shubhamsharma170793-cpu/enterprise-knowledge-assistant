import faiss            # 📌 Library: specialized for vector search
import numpy as np      # 📌 Library: handles numeric arrays efficiently


def build_faiss_index(embeddings: list):
    """
    📌 Build a FAISS index from embeddings.

    Args:
        embeddings (list): List of embedding vectors (lists of floats).

    Returns:
        index (faiss.IndexFlatL2): FAISS index for similarity search.
    """
    # Step 1: Get vector size (e.g., 384 for Hugging Face embeddings)
    dimension = len(embeddings[0])

    # Step 2: Create FAISS index that uses Euclidean (L2) distance
    index = faiss.IndexFlatL2(dimension)

    # Step 3: Add all embeddings to the index (convert to float32 array first)
    index.add(np.array(embeddings).astype('float32'))

    return index


def search_faiss(query_embedding: list, index, chunks: list, top_k: int = 3):
    """
    📌 Search FAISS index to find most similar chunks.

    Args:
        query_embedding (list): Embedding for user query.
        index: FAISS index object.
        chunks (list): Original text chunks.
        top_k (int): Number of results to return (default = 3).

    Returns:
        list: Top matching text chunks.
    """
    # Step 1: Turn query embedding into shape (1, dim) numpy array
    query_vector = np.array([query_embedding]).astype('float32')

    # Step 2: Search FAISS for nearest neighbors
    distances, indices = index.search(query_vector, top_k)

    # Step 3: Map result indices back to actual text chunks
    return [chunks[i] for i in indices[0]]

