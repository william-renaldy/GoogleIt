"""
GoogleIt Palm Model Module

This module provides a wrapper class for interacting with the Google Palm 2 language model.

Usage:
    - Import the module: `from GoogleIt import palm_model`
    - Create an instance of `Palm2Model`.
    - Initialize the model using the `init` method with a valid API key.
    - Use the `query` method to generate answers based on a reference document and user's question.

Example:
    ```python
    palm = palm_model.Palm2Model()
    palm.init(api_key='your_api_key_here')
    document = "The reference document containing relevant information."
    question = "What is the capital of France?"
    answer = palm.query(document, question)
    print(answer)
    ```

Classes:
    - `Palm2Model`:
        - Wrapper class for the Google Palm 2 language model.
        - Methods:
            - `__init__(self) -> None`: Initializes the Palm2Model instance.
            - `init(self, api_key: str) -> None`: Initializes the Palm 2 language model using the provided API key.
            - `make_prompt(self, query: str, relevant_passage: str) -> str`: Generates a prompt for the Palm 2 language model.
            - `query(self, document: str, question: str) -> str`: Queries the Palm 2 language model for an answer.

Attributes:
    - `model` (Palm2Model attribute): The initialized Palm 2 language model.

Raises:
    - `ValueError`: If the language model is not initialized before calling the `query` method.

Note:
    This module requires a valid API key for authentication.

"""

import google.generativeai as palm
import textwrap


class Palm2Model:
    """
    Wrapper class for interacting with the Google Palm 2 language model.

    Attributes:
    - model: The initialized Palm 2 language model.
    """

    def __init__(self) -> None:
        """Initialize the Palm2Model instance."""
        self.model = None

    def init(self, api_key: str) -> None:
        """
        Initialize the Palm 2 language model using the provided API key.

        Parameters:
        - api_key (str): The API key for authentication.
        """
        palm.configure(api_key=api_key)
        models = [m for m in palm.list_models() if 'generateText' in m.supported_generation_methods]

        if not models:
            raise ValueError("No models with text generation support found.")

        self.model = models[0]

    def make_prompt(self, query: str, relevant_passage: str) -> str:
        """
        Generate a prompt for the Palm 2 language model.

        Parameters:
        - query (str): The user's question.
        - relevant_passage (str): The relevant passage for context.

        Returns:
        str: The formatted prompt for the language model.
        """
        escaped = relevant_passage.replace("'", "").replace('"', "").replace("\n", " ")
        prompt = textwrap.dedent(f"""You are a helpful and informative bot that answers questions using text from the reference passage included below. \
            Be sure to respond in a complete sentence, being comprehensive, including all relevant background information. \
            However, you are talking to a non-technical audience, so be sure to break down complicated concepts and \
            strike a friendly and conversational tone. \
            The length of the response should be relevant to the prompt. Provide longer responses only if asked
            If the passage is irrelevant to the answer, you may ignore it.
            Redraft the response properly with proper sentence formation. Make sure the response length is reasonable and readable
            QUESTION: '{query}'
            PASSAGE: '{escaped}'

            ANSWER:
            """)

        return prompt

    def query(self, document: str, question: str) -> str:
        """
        Query the Palm 2 language model for an answer.

        Parameters:
        - document (str): The reference document for context.
        - question (str): The user's question.

        Returns:
        str: The generated answer from the language model.
        """
        if self.model is None:
            raise ValueError("The language model is not initialized. Call init() with the API key first.")

        prompt = self.make_prompt(question, document)

        temperature = 0.2
        answer = palm.generate_text(prompt=prompt,
                                    model=self.model,
                                    candidate_count=3,
                                    temperature=temperature,
                                    max_output_tokens=100)

        return answer.candidates[0]['output']