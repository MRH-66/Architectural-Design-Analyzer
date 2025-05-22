# ArchitecturalRAGSystem/src/data_ingestion/pdf_parser.py
import fitz  # PyMuPDF
import os
from typing import List, Dict, Any


class PDFParser:
    """
    Handles parsing of PDF files to extract text content.
    """

    def __init__(self):
        pass  # No specific initialization needed for now

    def extract_text_from_pdf(self, pdf_path: str) -> List[Dict[str, Any]]:
        """
        Extracts text content from each page of a PDF file.

        Args:
            pdf_path (str): The full path to the PDF file.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries, where each dictionary
                                  represents a page and contains 'page_number'
                                  and 'text_content'. Returns an empty list if
                                  the PDF cannot be opened or has no text.
        """
        if not os.path.exists(pdf_path):
            print(f"Error: PDF file not found at '{pdf_path}'")
            return []

        all_pages_data: List[Dict[str, Any]] = []
        try:
            doc = fitz.open(pdf_path)
            print(
                f"Processing PDF: '{os.path.basename(pdf_path)}', Pages: {len(doc)}")
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                text = page.get_text("text")  # Extract plain text

                # Clean up common PDF text extraction artifacts
                cleaned_text = text.strip()
                # Replace multiple spaces/newlines with a single one (optional, can affect layout meaning)
                # cleaned_text = " ".join(cleaned_text.split())

                if cleaned_text:  # Only add if there's actual text after stripping
                    all_pages_data.append({
                        "source_pdf": os.path.basename(pdf_path),
                        "page_number": page_num + 1,
                        "text_content": cleaned_text
                    })
            doc.close()
            print(
                f"Successfully extracted text from {len(all_pages_data)} pages of '{os.path.basename(pdf_path)}'.")
        except Exception as e:
            print(f"Error processing PDF '{pdf_path}': {e}")
            return []  # Return empty list on error

        return all_pages_data

    # Future methods for more advanced parsing could go here:
    # - extract_tables_from_page(page_object)
    # - identify_image_blocks_on_page(page_object)
    # - extract_text_with_font_info(page_object)


# --- Example Usage (can be run directly for testing this module) ---
if __name__ == '__main__':
    print("Testing PDFParser...")

    # For direct testing, we need to know where the 'data' folder is relative to this script
    # This assumes the script is run with 'python -m src.data_ingestion.pdf_parser'
    # from the project root.
    project_root_for_test = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", ".."))
    sample_pdf_dir = os.path.join(project_root_for_test, "data")

    # Try to find one of the books to test with
    # Make sure you have at least one of your PDFs in the 'data' folder
    # For this test, we'll just try to pick the first one from a predefined list
    # In a real scenario, your ingestion script would iterate through specific files.

    # Using Config to get book names and data path
    import sys
    # Add project root to path for importing config
    sys.path.insert(0, project_root_for_test)
    from src.config import Config
    cfg = Config()

    # Use the first book from the config for testing
    if cfg.BOOKS_TO_PROCESS:
        # e.g., "Time-Saver_Standards.pdf"
        test_pdf_filename = cfg.BOOKS_TO_PROCESS[0]
        test_pdf_path = os.path.join(cfg.DATA_PATH, test_pdf_filename)

        parser = PDFParser()
        if os.path.exists(test_pdf_path):
            print(f"\nAttempting to parse: {test_pdf_path}")
            extracted_data = parser.extract_text_from_pdf(test_pdf_path)

            if extracted_data:
                print(
                    f"\nSuccessfully extracted data from {len(extracted_data)} pages.")
                print("Sample from first extracted page:")
                print(f"  Source: {extracted_data[0]['source_pdf']}")
                print(f"  Page: {extracted_data[0]['page_number']}")
                print(
                    f"  Text (first 300 chars): {extracted_data[0]['text_content'][:300]}...")
                if len(extracted_data) > 1:
                    print("\nSample from second extracted page (if exists):")
                    print(f"  Source: {extracted_data[1]['source_pdf']}")
                    print(f"  Page: {extracted_data[1]['page_number']}")
                    print(
                        f"  Text (first 300 chars): {extracted_data[1]['text_content'][:300]}...")
            else:
                print(
                    f"No data extracted from '{test_pdf_filename}' or an error occurred.")
        else:
            print(
                f"Test PDF not found: {test_pdf_path}. Please place it in the '{cfg.DATA_PATH}' directory.")
    else:
        print("No books configured in Config.BOOKS_TO_PROCESS to test PDF parsing.")

    print("\nPDFParser test finished.")
