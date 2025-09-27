import fitz  # PyMuPDF

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

