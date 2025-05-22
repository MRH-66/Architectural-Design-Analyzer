# ArchitecturalRAGSystem/src/config.py
import os
from dotenv import load_dotenv
from uuid import UUID  # For type hinting

# Calculate the project root directory dynamically and correctly
# __file__ is the path to the current script (src/config.py)
# os.path.dirname(__file__) is the directory of the current script (src/)
# os.path.abspath(os.path.join(os.path.dirname(__file__), "..")) goes one level up to ArchitecturalRAGSystem/
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
dotenv_path = os.path.join(project_root, '.env')

# Attempt to load environment variables from .env file
if os.path.exists(dotenv_path):
    load_dotenv(dotenv_path)
    print(f"Loaded .env file from: {dotenv_path}")
else:
    print(
        f"Warning: .env file not found at {dotenv_path}. API key and other env vars may not be loaded.")


class Config:
    """
    Centralized configuration settings for the RAG system.
    """
    # --- API Keys ---
    GOOGLE_API_KEY: str = os.getenv(
        "GOOGLE_API_KEY", "YOUR_GOOGLE_API_KEY_FALLBACK_IF_NOT_IN_ENV")

    # --- Paths ---
    PROJECT_ROOT: str = project_root
    DATA_PATH: str = os.path.join(PROJECT_ROOT, "data")
    CHROMA_DB_PATH: str = os.path.join(PROJECT_ROOT, "chroma_db_store_v1")
    OUTPUT_JSON_PATH: str = os.path.join(PROJECT_ROOT, "output_jsons")

    # --- ChromaDB Settings ---
    COLLECTION_NAME: str = "architectural_standards_v1"

    # --- Gemini Model Names ---
    GEMINI_EMBEDDING_MODEL: str = "models/embedding-001"
    GEMINI_SYNTHESIS_MODEL: str = "models/gemini-2.5-flash-preview-04-17"
    GEMINI_REQUIREMENT_EXTRACTION_MODEL: str = "models/gemini-2.5-flash-preview-04-17"
    GEMINI_MULTIMODAL_IMAGE_ANALYSIS_MODEL: str = "models/gemini-2.5-flash-preview-04-17"

    # --- Data Ingestion & Chunking Settings ---
    NAMESPACE_UUID_BOOK_CONTENT: UUID = UUID(
        'c274dd16-0f1a-4e3a-9a91-77061ff49c7a')  # Replace with your own generated UUID
    CHUNK_MIN_LENGTH: int = 50
    CHUNK_TARGET_SIZE: int = 500
    CHUNK_OVERLAP: int = 50
    INGESTION_BATCH_SIZE: int = 50

    # --- RAG Retrieval Settings ---
    RAG_NUM_RETRIEVED_CHUNKS: int = 5

    # --- Logging ---
    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = os.path.join(PROJECT_ROOT, "app.log")

    # --- Book Specifics ---
    BOOKS_TO_PROCESS: list[str] = [  # Storing just filenames, will be joined with DATA_PATH
        "Time-Saver_Standards.pdf",
        "Modern_Construction_Handbook.pdf",
        "Neufert_Architects_Data.pdf"
    ]

    def __init__(self):
        if not self.GOOGLE_API_KEY or "YOUR_GOOGLE_API_KEY" in self.GOOGLE_API_KEY:
            print(
                "WARNING: GOOGLE_API_KEY is not set correctly in .env or is using a placeholder.")

        os.makedirs(self.OUTPUT_JSON_PATH, exist_ok=True)
        os.makedirs(self.CHROMA_DB_PATH, exist_ok=True)


# For testing this file directly:
if __name__ == "__main__":
    print("Running config.py directly for testing...")
    cfg = Config()
    print(f"Project Root: {cfg.PROJECT_ROOT}")
    print(f"Data Path: {cfg.DATA_PATH}")
    print(f"Chroma DB Path: {cfg.CHROMA_DB_PATH}")
    print(f"Collection Name: {cfg.COLLECTION_NAME}")
    api_key_status = "Yes" if cfg.GOOGLE_API_KEY and 'FALLBACK' not in cfg.GOOGLE_API_KEY else "No or Fallback"
    print(f"Google API Key Loaded: {api_key_status}")
    print(f"Output JSON Path Exists: {os.path.exists(cfg.OUTPUT_JSON_PATH)}")
    print(f"Chroma DB Path Exists: {os.path.exists(cfg.CHROMA_DB_PATH)}")
    print(f"Books to Process: {cfg.BOOKS_TO_PROCESS}")
    print("Config test finished.")
