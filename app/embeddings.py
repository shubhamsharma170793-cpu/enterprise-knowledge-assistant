from openai import OpenAI   # 📌 library + class
client = OpenAI()           # 📌 variable (instance of class)

def generate_embeddings(chunks: list, model: str = "text-embedding-3-small") -> list:
    """
    📌 Function: Generate embeddings for a list of text chunks using OpenAI.

    Args:
        chunks (list): List of text chunks.
        model (str): Embedding model to use (default = 'text-embedding-3-small').

    Returns:
        list: List of embedding vectors (one per chunk).
    """
    embeddings = []   # 📌 variable: holds all vectors
    for chunk in chunks:
        response = client.embeddings.create(   # 📌 function (method) call
            input=chunk,
            model=model
        )
        embeddings.append(response.data[0].embedding)  # vector = list of floats
    return embeddings

