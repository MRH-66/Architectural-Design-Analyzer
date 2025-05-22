# ArchitecturalRAGSystem/src/embedding/gemini_embedder.py
import google.generativeai as genai
from typing import List, Optional, Union
import time  # For potential retries with backoff
import os  # For loading environment variables
# Note: The genai module is assumed to be installed and configured correctly.
# If you are using a config file or environment variables, ensure they are loaded.


# It's good practice to import your config if needed, though for this module,
# the model name might be passed in as an argument.
# from src.config import Config
# cfg = Config() # If you need global config access here


class GeminiEmbedder:
    """
    A class to handle text embedding generation using the Gemini API.
    """

    def __init__(self, model_name: str, api_key: Optional[str] = None):
        """
        Initializes the GeminiEmbedder.

        Args:
            model_name (str): The name of the Gemini embedding model to use
                              (e.g., "models/embedding-001").
            api_key (Optional[str]): The Google API Key. If None, it assumes
                                     genai.configure() has been called elsewhere
                                     or Application Default Credentials are set up.
        """
        self.model_name = model_name
        if api_key:
            genai.configure(api_key=api_key)
        # It's assumed genai.configure() has been called if api_key is None,
        # typically in config.py or a main script.

    def embed_texts(self,
                    texts: List[str],
                    task_type: str = "RETRIEVAL_DOCUMENT",
                    batch_size: int = 100,  # Gemini embed_content can handle up to 100 content items
                    max_retries: int = 3,
                    initial_backoff: float = 1.0
                    ) -> List[Optional[List[float]]]:
        """
        Generates embeddings for a list of text strings in batches.

        Args:
            texts (List[str]): A list of text strings to embed.
            task_type (str): The type of task for the embedding.
                             "RETRIEVAL_DOCUMENT" for documents to be stored.
                             "RETRIEVAL_QUERY" for query text.
                             Other types include "SEMANTIC_SIMILARITY", "CLASSIFICATION", "CLUSTERING".
            batch_size (int): How many texts to send to the API in a single call.
            max_retries (int): Maximum number of retries for API calls.
            initial_backoff (float): Initial backoff time in seconds for retries.

        Returns:
            List[Optional[List[float]]]: A list of embeddings. Each embedding is a list of floats.
                                         Returns None for an item if embedding failed for that item after retries.
        """
        if not texts:
            return []

        all_embeddings: List[Optional[List[float]]] = [None] * len(texts)

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            current_retry = 0
            while current_retry < max_retries:
                try:
                    print(
                        f"  Embedding batch {i//batch_size + 1} (size: {len(batch_texts)}) with model '{self.model_name}'...")
                    # The `embed_content` method directly supports batching if `content` is a list of strings.
                    result = genai.embed_content(
                        model=self.model_name,
                        content=batch_texts,  # Pass the list of texts
                        task_type=task_type
                    )
                    # result['embedding'] will be a list of embeddings, one for each text in batch_texts
                    batch_embeddings = result['embedding']

                    # Place embeddings into the correct positions in all_embeddings
                    for j, embedding in enumerate(batch_embeddings):
                        all_embeddings[i + j] = embedding
                    print(
                        f"    Successfully embedded batch {i//batch_size + 1}.")
                    break  # Success, exit retry loop for this batch
                except Exception as e:
                    current_retry += 1
                    print(
                        f"    Error embedding batch {i//batch_size + 1}, attempt {current_retry}/{max_retries}: {e}")
                    if current_retry >= max_retries:
                        print(
                            f"    Failed to embed batch {i//batch_size + 1} after {max_retries} retries.")
                        # The corresponding entries in all_embeddings will remain None
                        break  # Exit retry loop for this batch, move to next batch
                    backoff_time = initial_backoff * (2 ** (current_retry - 1))
                    print(f"    Retrying in {backoff_time:.2f} seconds...")
                    time.sleep(backoff_time)

        return all_embeddings

    def embed_text(self,
                   text: str,
                   task_type: str = "RETRIEVAL_DOCUMENT"
                   ) -> Optional[List[float]]:
        """
        Generates embedding for a single text string.

        Args:
            text (str): The text string to embed.
            task_type (str): The type of task for the embedding.

        Returns:
            Optional[List[float]]: The embedding as a list of floats, or None if failed.
        """
        embeddings_list = self.embed_texts(
            texts=[text], task_type=task_type, batch_size=1)
        return embeddings_list[0] if embeddings_list else None


# --- Example Usage (can be run directly for testing this module) ---
if __name__ == '__main__':
    print("Testing GeminiEmbedder...")
    # This assumes your GOOGLE_API_KEY is set as an environment variable
    # and genai.configure() has effectively been called by the OS resolving it.
    # If not, you might need to call genai.configure() here too, or pass the key.

    # Load API key from .env for direct testing of this module
    from dotenv import load_dotenv
    project_root_for_test = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", ".."))
    dotenv_path_for_test = os.path.join(project_root_for_test, '.env')
    if os.path.exists(dotenv_path_for_test):
        load_dotenv(dotenv_path_for_test)
        print(f".env loaded from: {dotenv_path_for_test}")
    else:
        print(
            f"Warning: .env file not found at {dotenv_path_for_test}. API key might not be loaded for direct test.")

    API_KEY_FOR_TEST = os.getenv("GOOGLE_API_KEY")
    if not API_KEY_FOR_TEST:
        print("GOOGLE_API_KEY not found in environment. Please set it for testing.")
    else:
        # Configure GenAI if not already done (idempotent)
        try:
            genai.configure(api_key=API_KEY_FOR_TEST)
            print("Gemini API configured for testing.")
        except Exception as e:
            print(f"Error configuring Gemini for testing: {e}")

        MODEL_NAME_FOR_TEST = "models/embedding-001"  # Use a valid embedding model
        embedder = GeminiEmbedder(model_name=MODEL_NAME_FOR_TEST)

        sample_texts_for_test = [
            "The standard kitchen countertop height is 36 inches.",
            "Minimum hallway width for accessibility is 36 inches.",
            "Neufert provides detailed dimensions for various room types.",
            "This is a very short text.",  # Test with short texts
            "This is a slightly longer text designed to test the embedding capabilities of the model for sentences that contain more contextual information and specific details relevant to architectural design standards."
        ]

        print("\nTesting single text embedding:")
        single_embedding = embedder.embed_text(
            sample_texts_for_test[0], task_type="RETRIEVAL_DOCUMENT")
        if single_embedding:
            print(
                f"  Embedding for '{sample_texts_for_test[0][:30]}...': Dimension {len(single_embedding)}, First 3 values: {single_embedding[:3]}")
        else:
            print(f"  Failed to embed '{sample_texts_for_test[0][:30]}...'")

        print("\nTesting batch text embedding:")
        batch_embeddings = embedder.embed_texts(
            sample_texts_for_test, task_type="RETRIEVAL_DOCUMENT")

        for i, (text, embedding) in enumerate(zip(sample_texts_for_test, batch_embeddings)):
            if embedding:
                print(
                    f"  Embedding for '{text[:30]}...': Dimension {len(embedding)}, First 3 values: {embedding[:3]}")
            else:
                print(f"  Failed to embed '{text[:30]}...'")

        print("\nTesting query embedding:")
        query_embedding = embedder.embed_text(
            "What are kitchen dimensions?", task_type="RETRIEVAL_QUERY")
        if query_embedding:
            print(
                f"  Embedding for query: Dimension {len(query_embedding)}, First 3 values: {query_embedding[:3]}")
        else:
            print(f"  Failed to embed query.")

        print("\nGeminiEmbedder test finished.")
