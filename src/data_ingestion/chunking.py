# ArchitecturalRAGSystem/src/data_ingestion/chunking.py
import uuid
from typing import List, Dict, Any, Optional
from langchain_text_splitters import RecursiveCharacterTextSplitter

# from src.config import Config # Will be used when called from an orchestrator script


class AdvancedTextChunker:
    """
    Uses RecursiveCharacterTextSplitter for more effective text chunking.
    """

    def __init__(self,
                 chunk_target_size: int = 500,
                 chunk_overlap: int = 50,
                 separators: Optional[List[str]] = None,
                 id_namespace_uuid: uuid.UUID = uuid.UUID(
                     '00000000-0000-0000-0000-000000000000'),
                 # Min length to add metadata like length
                 min_chunk_length_for_metadata: int = 20
                 ):
        """
        Initializes the AdvancedTextChunker.

        Args:
            chunk_target_size (int): The target size (in characters) for each chunk.
            chunk_overlap (int): The number of characters to overlap between chunks.
            separators (Optional[List[str]]): A list of strings by which to split the text recursively.
                                              If None, uses default Langchain separators.
            id_namespace_uuid (uuid.UUID): Namespace UUID for generating deterministic chunk IDs.
            min_chunk_length_for_metadata (int): Smallest chunk length considered for length metadata.
        """
        self.chunk_target_size = chunk_target_size
        self.chunk_overlap = chunk_overlap
        # Default separators are good, but you can customize if needed for your specific documents
        # e.g., separators=["\n\n\n", "\n\n", "\n", ". ", "? ", "! ", " ", ""]
        self.separators = separators
        self.id_namespace_uuid = id_namespace_uuid
        self.min_chunk_length_for_metadata = min_chunk_length_for_metadata

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_target_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            is_separator_regex=False,  # Set to True if your separators are regex
            separators=self.separators  # Pass None to use defaults
        )

    def pages_to_chunks(self,
                        pages_data: List[Dict[str, Any]],
                        source_document_name: str
                        ) -> List[Dict[str, Any]]:
        """
        Chunks text from extracted pages using RecursiveCharacterTextSplitter.

        Args:
            pages_data (List[Dict[str, Any]]): A list of dictionaries, where each
                dictionary represents a page and contains 'page_number' and 'text_content'.
            source_document_name (str): The name of the source document (e.g., PDF filename).

        Returns:
            List[Dict[str, Any]]: A list of chunk dictionaries, each containing
                                  'id', 'text', and 'metadata'.
        """
        all_chunks_data: List[Dict[str, Any]] = []
        print(
            f"\nChunking document: '{source_document_name}' using RecursiveCharacterTextSplitter...")

        total_chunks_generated = 0
        for page_idx, page in enumerate(pages_data):
            page_text = page.get("text_content", "")
            page_number = page.get("page_number", 0)

            if not page_text.strip():
                # print(f"  Page {page_number}: No text content, skipping.")
                continue

            # Use the text_splitter to get Document objects (or just strings if preferred)
            # The splitter returns a list of strings by default.
            split_texts = self.text_splitter.split_text(page_text)

            # print(f"  Page {page_number}: Original length {len(page_text)}, split into {len(split_texts)} sub-chunks.")

            for chunk_seq_on_page, chunk_text in enumerate(split_texts):
                cleaned_chunk_text = chunk_text.strip()
                if not cleaned_chunk_text:  # Should not happen with Langchain splitter usually
                    continue

                # Create a deterministic ID
                id_content_string = f"{source_document_name}_p{page_number}_chunk{chunk_seq_on_page+1}_{cleaned_chunk_text[:50]}"
                chunk_id = str(uuid.uuid5(
                    self.id_namespace_uuid, id_content_string))

                metadata = {
                    "source_document": source_document_name,
                    "original_page_number": page_number,
                    "chunk_sequence_on_page": chunk_seq_on_page + 1,
                }
                if len(cleaned_chunk_text) >= self.min_chunk_length_for_metadata:
                    metadata["chunk_length_chars"] = len(cleaned_chunk_text)

                all_chunks_data.append({
                    "id": chunk_id,
                    "text": cleaned_chunk_text,
                    "metadata": metadata
                })
                total_chunks_generated += 1

        print(
            f"Generated {total_chunks_generated} chunks from '{source_document_name}'.")
        return all_chunks_data


# --- Example Usage (can be run directly for testing this module) ---
if __name__ == '__main__':
    print("Testing AdvancedTextChunker with RecursiveCharacterTextSplitter...")

    import sys
    import os
    project_root_for_test = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", ".."))
    sys.path.insert(0, project_root_for_test)

    from src.config import Config
    # Assuming pdf_parser.py is in the same dir
    from src.data_ingestion.pdf_parser import PDFParser

    cfg = Config()

    # --- Test Data Preparation ---
    if not cfg.BOOKS_TO_PROCESS:
        print("No books configured in Config.BOOKS_TO_PROCESS to test chunking.")
        exit()

    # Use the first configured book
    test_pdf_filename = cfg.BOOKS_TO_PROCESS[0]
    test_pdf_path = os.path.join(cfg.DATA_PATH, test_pdf_filename)

    if not os.path.exists(test_pdf_path):
        print(
            f"Test PDF not found: {test_pdf_path}. Please place it in '{cfg.DATA_PATH}'.")
        exit()

    pdf_parser = PDFParser()
    print(
        f"\nExtracting pages from '{test_pdf_filename}' for chunking test...")

    all_pages_from_pdf = pdf_parser.extract_text_from_pdf(test_pdf_path)

    if not all_pages_from_pdf:
        print(
            f"Could not extract any pages from '{test_pdf_filename}'. Aborting chunker test.")
        exit()

    # Test with first 3 pages for speed
    sample_pages_for_chunking = all_pages_from_pdf[:3]
    print(
        f"Using {len(sample_pages_for_chunking)} sample pages for chunking test.")

    # Initialize the chunker with settings from Config
    # You can also define custom separators if the defaults aren't working well for your PDFs
    # e.g., custom_separators = ["\n\n\n", "\n\n", "\n", ". ", "; ", ", ", " ", ""]
    chunker = AdvancedTextChunker(
        chunk_target_size=cfg.CHUNK_TARGET_SIZE,  # e.g., 500
        chunk_overlap=cfg.CHUNK_OVERLAP,       # e.g., 50
        # separators=custom_separators, # Optional
        id_namespace_uuid=cfg.NAMESPACE_UUID_BOOK_CONTENT
    )

    # Perform chunking
    generated_chunks = chunker.pages_to_chunks(
        pages_data=sample_pages_for_chunking,
        source_document_name=test_pdf_filename
    )

    if generated_chunks:
        print(
            f"\nSuccessfully generated {len(generated_chunks)} chunks using RecursiveCharacterTextSplitter.")
        print("Sample of first 5 chunks (if available):")
        for i, chunk in enumerate(generated_chunks[:5]):
            print(f"  --- Chunk {i+1} ---")
            print(f"  ID: {chunk['id']}")
            print(f"  Metadata: {chunk['metadata']}")
            print(f"  Text Length: {len(chunk['text'])}")
            print(f"  Text (first 150 chars): {chunk['text'][:150]}...")
            if len(chunk['text']) > 150:
                print(f"  Text (last 50 chars): ...{chunk['text'][-50:]}")
    else:
        print("No chunks were generated from the sample pages.")

    print("\nAdvancedTextChunker test finished.")
