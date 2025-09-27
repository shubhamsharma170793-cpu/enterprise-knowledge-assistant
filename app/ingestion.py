import fitz  # PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter


def extract_text_from_pdf(file_path: str) -> str:
    """
    Extracts text from a PDF file using PyMuPDF.
    
    Args:
        file_path (str): Path to the PDF file.
    
    Returns:
        str: The extracted text from the PDF.
    """
    text = ""
    # Open the PDF
    with fitz.open(file_path) as pdf:
        # Loop through each page
        for page in pdf:
            # Extract text from that page
            text += page.get_text()
    return text




def chunk_text(text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> list:
    """
    Splits extracted text into smaller overlapping chunks.

    Args:
        text (str): The full extracted text.
        chunk_size (int): Max characters per chunk.
        chunk_overlap (int): Overlap between chunks.

    Returns:
        list: List of text chunks.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return splitter.split_text(text)

