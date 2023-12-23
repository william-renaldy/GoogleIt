"""
GoogleIt Module

This module provides the `GoogleIt` class, which encapsulates functionality for performing queries, retrieving top URLs from Google search results, downloading content from URLs, preprocessing text, extracting domain names from URLs, combining PDF files, and extracting relevant content based on cosine similarity.

Usage:
    - Import the module: `from GoogleIt.googleit import GoogleIt`
    - Create an instance of the `GoogleIt` class with a valid API key.
    - Use the provided methods for various tasks.

Example:
```python
google_it = GoogleIt(api_key='your_api_key_here', model = "Palm2")
query = "How does photosynthesis work?"
response = google_it.get(query=query, urls_count=5)
print(response)
```

Classes:
    - `GoogleIt`:
        - A class that provides functionality for querying, retrieving URLs, downloading content, preprocessing text, and more.
        - Methods:
            - `__init__(self, api_key: str, model: str = "Palm2") -> None`: Initializes the `GoogleIt` instance with the provided API key and a specified language model.
            - `save_url_to_pdf(self, url: str, pdf_path: str) -> None`: Downloads content from a URL and saves it as a PDF file.
            - `preprocess_text(self, text: str) -> str`: Preprocesses text by converting it to lowercase, tokenizing, and removing stopwords and punctuation.
            - `get_domain_name(self, url: str) -> str`: Extracts the domain name from a given URL.
            - `get_top_urls(self, query: str, urls_count: int = 5) -> Tuple[list[str], list[str]]`: Retrieves top URLs from Google search results based on a given query.
            - `combine_pdf(self, folder_path: str) -> str`: Combines multiple PDF files into a single merged PDF.
            - `extract_relevant_content(self, input_text: str, main_document: str, threshold: float = 0.2) -> str`: Extracts relevant content from the input text based on cosine similarity.
            - `with_document(self, query: str, google_doc: str, pdf_path: str) -> str`: Processes a query using a provided PDF document and a Google document.
            - `without_document(self, query: str, paragraphs: list[str]) -> str`: Processes a query without a provided PDF document.
            - `get(self, query: str, pdf_path: str | None = None, urls_count: int = 5) -> str`: Main function to retrieve information based on a query, optionally using a PDF document.

Attributes:
    - `model` (GoogleIt attribute): An instance of the model class for natural language processing.

Note:
    This module requires the `Palm2Model` class and `GeminiModel` from the `models` module for natural language processing.
"""


import os
import re
import shutil
from PyPDF2 import PdfMerger
from bs4 import BeautifulSoup
import requests
import converter
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from models import Palm2Model, GeminiModel
from text_processor import extract_text_from_pdf, get_chunks
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download necessary NLTK resources
nltk.download('stopwords')
nltk.download('punkt')

class GoogleIt:
    """
    GoogleIt class for retrieving information using Google search and document processing.

    Attributes:
        model: An instance of the underlying language model (Palm2Model or GeminiModel).

    Methods:
        __init__: Initializes the GoogleIt instance with the provided API key and model.
        save_url_to_pdf: Downloads content from a URL and saves it as a PDF file.
        preprocess_text: Preprocesses text by converting it to lowercase, tokenizing, and removing stopwords and punctuation.
        get_domain_name: Extracts the domain name from a given URL.
        get_top_urls: Retrieves top URLs from Google search results based on a given query.
        combine_pdf: Combines multiple PDF files into a single merged PDF.
        extract_relevant_content: Extracts relevant content from the input text based on cosine similarity with the main document.
        with_document: Processes a query using a provided PDF document and a Google document.
        without_document: Processes a query without a provided PDF document.
        get: Main function to retrieve information based on a query, optionally using a PDF document.
    """

    def __init__(self, api_key: str, model: str = "Palm2") -> None:
        """
        Initializes the GoogleIt instance with the provided API key and a specified language model.

        Args:
            api_key (str): The API key for initializing the underlying language model.
            model (str): The backend language model to use, either "Palm2" or "GeminiPro" (default is "Palm2").
        
        Raises:
            ValueError: If an invalid value for `model` is provided.

        Returns:
            None
        """
        # Validate and set the language model
        if model == "Palm2":
            self.model = Palm2Model()
        elif model == "GeminiPro":
            self.model = GeminiModel()
        else:
            raise ValueError("Invalid value for `model`. Available models are: [Palm2, GeminiPro]")

        # Initialize the language model with the provided API key
        self.model.init(api_key=api_key)

    def save_url_to_pdf(self, url: str, pdf_path: str) -> None:
        """
        Downloads content from a URL and saves it as a PDF file.

        Parameters:
            url (str): The URL to download content from.
            pdf_path (str): The path to save the resulting PDF file.
        """
        converter.convert(url, pdf_path)

    def preprocess_text(self, text: str) -> str:
        """
        Preprocesses text by converting it to lowercase, tokenizing, and removing stopwords and punctuation.

        Parameters:
            text (str): The input text to preprocess.

        Returns:
            str: The preprocessed text.
        """
        tokens = word_tokenize(text.lower())
        stop_words = set(stopwords.words('english'))
        tokens = [token for token in tokens if token.isalnum() and token not in stop_words]
        return ' '.join(tokens)

    def get_domain_name(self, url: str) -> str:
        """
        Extracts the domain name from a given URL.

        Parameters:
            url (str): The input URL.

        Returns:
            str: The extracted domain name.
        """
        match = re.search(r'(https?://)?(www\.)?(.+?)\.(.+?)', url)
        if match:
            return match.group(3)
        else:
            return None

    def get_top_urls(self, query: str, urls_count: int = 5) -> tuple[list[str], list[str]]:
        """
        Retrieves top URLs from Google search results based on a given query.

        Parameters:
            query (str): The search query.
            urls_count (int): The number of URLs to retrieve (default is 5).

        Returns:
            tuple[list[str], list[str]]: A tuple containing lists of URLs and corresponding domain names.
        """
        results = 5 * urls_count
        page = requests.get(f"https://www.google.com/search?q={query}&num={results}")
        soup = BeautifulSoup(page.content, "lxml")
        links = soup.findAll("a")
        c = 0

        urls: list[str] = []
        domain_list: list[str] = []
        for link in links:
            link_href = link.get('href')
            if "url?q=" in link_href and not "webcache" in link_href:
                l = link.get('href').split("?q=")[1].split("&sa=U")[0]
                domain = self.get_domain_name(l)
                if domain is not None:
                    if domain not in domain_list:
                        c += 1
                        domain_list.append(domain)
                        urls.append(l)

                    if c == urls_count:
                        break

        return urls, domain_list

    def combine_pdf(self, folder_path: str) -> str:
        """
        Combines multiple PDF files into a single merged PDF.

        Parameters:
            folder_path (str): The path to the folder containing PDF files.

        Returns:
            str: The path to the merged PDF file.
        """
        merged_pdf_path = "merged.pdf"
        merger = PdfMerger()

        for pdf_file in os.listdir(folder_path):
            if pdf_file.endswith(".pdf"):
                merger.append(os.path.join(folder_path, pdf_file))

        merger.write(merged_pdf_path)
        return merged_pdf_path

    def extract_relevant_content(self, input_text: str, main_document: str, threshold: float = 0.2) -> str:
        """
        Extracts relevant content from the input text based on cosine similarity with the main document.

        Parameters:
            input_text (str): The input text to analyze.
            main_document (str): The main document for comparison.
            threshold (float): The similarity threshold (default is 0.2).

        Returns:
            str: The relevant content if similarity is above the threshold; otherwise, None.
        """
        preprocessed_input = self.preprocess_text(input_text)
        preprocessed_main_doc = self.preprocess_text(main_document)

        # Vectorize the text using TF-IDF
        vectorizer = TfidfVectorizer()
        vectors = vectorizer.fit_transform([preprocessed_input, preprocessed_main_doc])

        # Calculate the cosine similarity between the vectors
        similarity_score = cosine_similarity(vectors[0], vectors[1])[0][0]

        # Extract relevant content based on the threshold
        if similarity_score >= threshold:
            return input_text
        else:
            return None

    def with_document(self, query: str, google_doc: str, pdf_path: str) -> str:
        """
        Processes a query using a provided PDF document and a Google document.

        Parameters:
            query (str): The query to process.
            google_doc (str): The Google document content.
            pdf_path (str): The path to the PDF document.

        Returns:
            str: The response to the query.
        """
        input_doc = extract_text_from_pdf(pdf_path=pdf_path)[0]
        chunks = google_doc.split("\n")
        docs = ""
        relevant_chunks: list[str] = []

        for chunk in chunks:
            relevant_content = self.extract_relevant_content(chunk, input_doc, threshold=0.2)
            if relevant_content:
                relevant_chunks.append(relevant_content)

        docs = "".join(relevant_chunks)[:49000]

        response = self.model.query(document=docs, question=query)

        return response

    def without_document(self, query: str, paragraphs: list[str]) -> str:
        """
        Processes a query without a provided PDF document.

        Parameters:
            query (str): The query to process.
            paragraphs (list[str]): List of document paragraphs.

        Returns:
            str: The response to the query.
        """
        chunks = get_chunks(paragraphs=paragraphs)
        document = "".join(chunks)[:49000]
        response = self.model.query(document=document, question=query)
        return response

    def get(self, query: str, pdf_path: str | None = None, urls_count: int = 5) -> str:
        """
        Main function to retrieve information based on a query, optionally using a PDF document.

        Parameters:
            query (str): The query to process.
            pdf_path (str | None): The path to the PDF document (optional).
            urls_count (int): The number of URLs to consider (default is 5).

        Returns:
            str: The response to the query.
        """
        urls, domains = self.get_top_urls(query=query, urls_count=urls_count)
        folder_path = "PDFFiles"

        try:
            shutil.rmtree(folder_path, ignore_errors=True)
        except FileNotFoundError:
            pass

        # Create an empty folder with the same name
        os.mkdir(folder_path)

        for url, domain in zip(urls, domains):
            pdf_path = "PDFFiles/" + domain + ".pdf"
            self.save_url_to_pdf(url=url, pdf_path=pdf_path)

        combined_path = self.combine_pdf("PDFFiles")

        pdf_text, paragraphs = extract_text_from_pdf(pdf_path=combined_path)

        if pdf_path is not None:
            response = self.with_document(query=query, google_doc=pdf_text, pdf_path=pdf_path)
        else:
            response = self.without_document(query=query, paragraphs=paragraphs)

        return response

if __name__ == "__main__":
    m = GoogleIt("AIzaSyAXCVMFirOACW80v3w4lNEruWBAskWbeiw", model = "Palm2")
    r = m.get("How does photosynthesis work?", urls_count=2)
    print(r)