# GoogleIt Python Package

The `GoogleIt` package provides a set of tools for querying Google search results, retrieving URLs, downloading content, preprocessing text, extracting domain names from URLs, combining PDF files, and extracting relevant content based on cosine similarity.

## Installation

```bash
pip install GoogleIt
```

## Usage

```python
from GoogleIt.googleit import GoogleIt

# Create an instance of the GoogleIt class
google_it = GoogleIt(api_key='your_api_key_here')

# Perform a query and retrieve information
query = "How does photosynthesis work?"
response = google_it.get(query=query, urls_count=5)
print(response)
```

## Modules

1. [`converter.py` Documentation](#converterpy-documentation) - Provides functionality for converting HTML or websites to PDF.

2. [`palm_model.py` Documentation](#palm_modelpy-documentation) - Wrapper class for interacting with the Google Palm 2 language model.

3. [`text_processor.py` Documentation](#text_processorpy-documentation) - Functions for processing text documents, including converting PDF to DOCX, reading paragraphs from DOCX, dividing paragraphs into chunks, and extracting text and paragraphs from PDF.

4. [`googleit.py` Documentation](#googleitpy-documentation) - Main module encapsulating the GoogleIt class, which provides functionality for querying, retrieving URLs, downloading content, preprocessing text, and more.


## `converter.py` Documentation

GoogleIt Converter Module

This module provides functionality to convert HTML files or websites into PDF format using Selenium.

### Usage:
    - Import the module: `from GoogleIt import converter`
    - Call the `convert` function with appropriate parameters.

 ### Example:

```python
converter.convert(source='https://example.com', target='output.pdf', timeout=5)
```

### Functions:
    - `convert(source: str, target: str, timeout: int = 2, print_options: dict = {}) -> None`:
        Converts a given HTML file or website into PDF.

        Parameters:
            - `source` (str): Source HTML file or website link.
            - `target` (str): Target location to save the PDF.
            - `timeout` (int, optional): Timeout in seconds. Default is set to 2 seconds.
            - `print_options` (dict, optional): Options for PDF printing. Refer to https://vanilla.aslushnikov.com/?Page.printToPDF for available options.

        Raises:
            - Exception: If an error occurs during PDF conversion.

### Note:
    This module relies on the Selenium library and requires a compatible WebDriver (e.g., ChromeDriver) to be installed.


## `palm_model.py` Documentation

GoogleIt Palm Model Module

This module provides a wrapper class for interacting with the Google Palm 2 language model.

### Usage:
    - Import the module: `from GoogleIt import palm_model`
    - Create an instance of `Palm2Model`.
    - Initialize the model using the `init` method with a valid API key.
    - Use the `query` method to generate answers based on a reference document and user's question.

### Example:
```python
palm = palm_model.Palm2Model()
palm.init(api_key='your_api_key_here')
document = "The reference document containing relevant information."
question = "What is the capital of France?"
answer = palm.query(document, question)
print(answer)
```

### Classes:
    - `Palm2Model`:
        - Wrapper class for the Google Palm 2 language model.
        - Methods:
            - `__init__(self) -> None`: Initializes the Palm2Model instance.
            - `init(self, api_key: str) -> None`: Initializes the Palm 2 language model using the provided API key.
            - `make_prompt(self, query: str, relevant_passage: str) -> str`: Generates a prompt for the Palm 2 language model.
            - `query(self, document: str, question: str) -> str`: Queries the Palm 2 language model for an answer.

### Attributes:
    - `model` (Palm2Model attribute): The initialized Palm 2 language model.

### Raises:
    - `ValueError`: If the language model is not initialized before calling the `query` method.

### Note:
    This module requires a valid API key for authentication.


## `text_processor.py` Documentation

GoogleIt Text Processor Module

This module provides functions for processing text documents, including converting PDF to DOCX, reading paragraphs from DOCX, dividing paragraphs into chunks, and extracting text and paragraphs from PDF.

### Usage:
    - Import the module: `from GoogleIt import text_processor`
    - Use the provided functions for text processing tasks.

### Example:
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

### Functions:
    - `pdf_to_docx(pdf_file: str, docx_file: str) -> None`:
        Converts a PDF file to a DOCX file.

    - `read_document_paragraphs(filename: str) -> List[str]`:
        Reads paragraphs from a document (DOCX file).

    - `get_chunks(paragraphs: List[str], chunk_size: int = 10, overlap_size: int = 2) -> List[str]`:
        Divides a list of paragraphs into chunks.

    - `extract_text_from_pdf(pdf_path: str, docx_path: str = "converted_document.docx") -> Tuple[str, List[str]]`:
        Extracts text and paragraphs from a PDF file.

        Returns a tuple containing the extracted text and a list of paragraphs.

### Note:
    - The `get_chunks` function requires passing the list of paragraphs to the function.
    - The module includes an example at the end demonstrating the use of the `extract_text_from_pdf` function.


## `googleit.py` Documentation

GoogleIt Module

This module provides the `GoogleIt` class, which encapsulates functionality for performing queries, retrieving top URLs from Google search results, downloading content from URLs, preprocessing text, extracting domain names from URLs, combining PDF files, and extracting relevant content based on cosine similarity.

### Usage:
    - Import the module: `from GoogleIt.googleit import GoogleIt`
    - Create an instance of the `GoogleIt` class with a valid API key.
    - Use the provided methods for various tasks.

### Example:
```python
google_it = GoogleIt(api_key='your_api_key_here')
query = "How does photosynthesis work?"
response = google_it.get(query=query, urls_count=5)
print(response)
```

### Classes:
    - `GoogleIt`:
        - A class that provides functionality for querying, retrieving URLs, downloading content, preprocessing text, and more.
        - Methods:
            - `__init__(self, api_key: str) -> None`: Initializes the `GoogleIt` instance with the provided API key.
            - `save_url_to_pdf(self, url: str, pdf_path: str) -> None`: Downloads content from a URL and saves it as a PDF file.
            - `preprocess_text(self, text: str) -> str`: Preprocesses text by converting it to lowercase, tokenizing, and removing stopwords and punctuation.
            - `get_domain_name(self, url: str) -> str`: Extracts the domain name from a given URL.
            - `get_top_urls(self, query: str, urls_count: int = 5) -> Tuple[list[str], list[str]]`: Retrieves top URLs from Google search results based on a given query.
            - `combine_pdf(self, folder_path: str) -> str`: Combines multiple PDF files into a single merged PDF.
            - `extract_relevant_content(self, input_text: str, main_document: str, threshold: float = 0.2) -> str`: Extracts relevant content from the input text based on cosine similarity.
            - `with_document(self, query: str, google_doc: str, pdf_path: str) -> str`: Processes a query using a provided PDF document and a Google document.
            - `without_document(self, query: str, paragraphs: list[str]) -> str`: Processes a query without a provided PDF document.
            - `get(self, query: str, pdf_path: str | None = None, urls_count: int = 5) -> str`: Main function to retrieve information based on a query, optionally using a PDF document.

### Attributes:
    - `model` (GoogleIt attribute): An instance of the `Palm2Model` class for natural language processing.

### Note:
    This module requires the `Palm2Model` class from the `palm_model` module for natural language processing.


**Note:**
- Replace `'your_api_key_here'` with your actual Google API key.
