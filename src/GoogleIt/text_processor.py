"""
GoogleIt Text Processor Module

This module provides functions for processing text documents, including converting PDF to DOCX, reading paragraphs from DOCX, dividing paragraphs into chunks, and extracting text and paragraphs from PDF.

Usage:
    - Import the module: `from GoogleIt import text_processor`
    - Use the provided functions for text processing tasks.

Example:
    ```python
    pdf_path = "path/to/input.pdf"
    docx_path = "path/to/output.docx"

    # Convert PDF to DOCX
    text_processor.pdf_to_docx(pdf_file=pdf_path, docx_file=docx_path)

    # Read paragraphs from DOCX
    paragraphs = text_processor.read_document_paragraphs(filename=docx_path)

    # Divide paragraphs into chunks
    chunked_paragraphs = text_processor.get_chunks(paragraphs=paragraphs, chunk_size=10, overlap_size=2)

    # Extract text and paragraphs from PDF
    pdf_text, pdf_paragraphs = text_processor.extract_text_from_pdf(pdf_path=pdf_path, docx_path=docx_path)
    ```

Functions:
    - `pdf_to_docx(pdf_file: str, docx_file: str) -> None`:
        Converts a PDF file to a DOCX file.

    - `read_document_paragraphs(filename: str) -> List[str]`:
        Reads paragraphs from a document (DOCX file).

    - `get_chunks(paragraphs: List[str], chunk_size: int = 10, overlap_size: int = 2) -> List[str]`:
        Divides a list of paragraphs into chunks.

    - `extract_text_from_pdf(pdf_path: str, docx_path: str = "converted_document.docx") -> Tuple[str, List[str]]`:
        Extracts text and paragraphs from a PDF file.

        Returns a tuple containing the extracted text and a list of paragraphs.

Note:
    - The `get_chunks` function requires passing the list of paragraphs to the function.
    - The module includes an example at the end demonstrating the use of the `extract_text_from_pdf` function.

"""


import pdf2docx
from docx import Document as Docs
from typing import List, Tuple


def pdf_to_docx(pdf_file: str, docx_file: str) -> None:
    """
    Convert a PDF file to a DOCX file.

    Parameters:
    - pdf_file (str): The path to the input PDF file.
    - docx_file (str): The path to the output DOCX file.
    """
    pdf2docx.parse(pdf_file, docx_file)


def read_document_paragraphs(filename: str) -> List[str]:
    """
    Read paragraphs from a document (DOCX file).

    Parameters:
    - filename (str): The path to the DOCX file.

    Returns:
    List[str]: A list of paragraphs.
    """
    document = Docs(filename)
    paragraphs: List[str] = []
    for paragraph in document.paragraphs:
        if paragraph.text.strip():
            paragraphs.append(paragraph.text)
    return paragraphs


def get_chunks(paragraphs: List[str], chunk_size: int = 10, overlap_size: int = 2) -> List[str]:
    """
    Divide a list of paragraphs into chunks.

    Parameters:
    - paragraphs (List[str]): The list of paragraphs.
    - chunk_size (int): The size of each chunk (default is 10).
    - overlap_size (int): The size of overlap between chunks (default is 2).

    Returns:
    List[str]: A list of chunked paragraphs.
    """
    chunked_paragraphs: List[str] = []

    start_idx = 0
    while start_idx + chunk_size <= len(paragraphs):
        end_idx = start_idx + chunk_size
        chunk = " ".join(paragraphs[start_idx:end_idx])
        chunked_paragraphs.append(chunk)
        start_idx += chunk_size - overlap_size

    else:
        chunk = " ".join(paragraphs[start_idx:])
        chunked_paragraphs.append(chunk)

    return chunked_paragraphs


def extract_text_from_pdf(pdf_path: str, docx_path: str = "converted_document.docx") -> Tuple[str, List[str]]:
    """
    Extract text and paragraphs from a PDF file.

    Parameters:
    - pdf_path (str): The path to the input PDF file.
    - docx_path (str): The path to the output DOCX file (default is "converted_document.docx").

    Returns:
    Tuple[str, List[str]]: A tuple containing the extracted text and a list of paragraphs.
    """
    pdf_to_docx(pdf_path, docx_path)
    paragraphs = read_document_paragraphs(docx_path)
    pdf_text = " ".join(paragraphs)

    return pdf_text, paragraphs
