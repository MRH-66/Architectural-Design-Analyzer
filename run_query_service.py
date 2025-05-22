# ArchitecturalRAGSystem/run_query_service.py
import os
import json
import time
import argparse  # For command-line arguments
from typing import Dict, Any, List, Optional

# Import necessary classes from your src modules
from src.config import Config
from src.rag_pipeline.requirement_extractor import RequirementExtractor
from src.rag_pipeline.query_generator import QueryGenerator
from src.embedding.gemini_embedder import GeminiEmbedder
from src.vector_store.chroma_manager import ChromaManager
from src.rag_pipeline.synthesizer import Synthesizer  # Import the Synthesizer


def run_full_rag_pipeline(conversation_json_path: str, output_dir: str) -> Optional[Dict[str, Any]]:
    """
    Runs the FULL RAG pipeline from user conversation to final synthesized JSON.

    Args:
        conversation_json_path (str): Path to the input user conversation JSON file.
        output_dir (str): Directory to save the final synthesized output JSON.

    Returns:
        Optional[Dict[str, Any]]: The final synthesized JSON data, or None if an error occurs.
    """
    cfg = Config()  # Load configuration
    print("--- Starting Full RAG Pipeline ---")
    pipeline_start_time = time.time()

    # --- 1. Load User Conversation ---
    if not os.path.exists(conversation_json_path):
        print(
            f"Error: Conversation JSON file not found at '{conversation_json_path}'")
        return None

    try:
        with open(conversation_json_path, 'r', encoding='utf-8') as f:
            user_conversation_data = json.load(f)
        print(
            f"Successfully loaded conversation from: {conversation_json_path}")
    except Exception as e:
        print(f"Error loading conversation JSON: {e}")
        return None

    # --- Initialize All RAG Components ---
    print("\n--- Initializing RAG Components ---")
    requirement_extractor = RequirementExtractor(
        model_name=getattr(
            cfg, "GEMINI_REQUIREMENT_EXTRACTION_MODEL", "models/gemini-2.5-flash-preview-04-17"),
        api_key=cfg.GOOGLE_API_KEY
    )
    query_generator = QueryGenerator(
        use_llm_for_generation=False)  # Using rule-based

    gemini_embedder = GeminiEmbedder(
        model_name=cfg.GEMINI_EMBEDDING_MODEL,
        api_key=cfg.GOOGLE_API_KEY
    )
    chroma_manager = ChromaManager(
        path=cfg.CHROMA_DB_PATH,
        collection_name=cfg.COLLECTION_NAME
    )
    synthesizer = Synthesizer(  # Initialize the Synthesizer
        model_name=getattr(cfg, "GEMINI_SYNTHESIS_MODEL",
                           "models/gemini-2.5-flash-preview-04-17"),
        api_key=cfg.GOOGLE_API_KEY
    )

    # Check if models initialized correctly
    if not all([requirement_extractor.model, gemini_embedder.model_name, chroma_manager.collection, synthesizer.model]):  # Simple check
        print("Error: One or more RAG components failed to initialize properly. Exiting.")
        return None

    if chroma_manager.count() == 0:
        print(
            f"Warning: ChromaDB collection '{cfg.COLLECTION_NAME}' is empty. RAG will have no context.")
        # Proceeding, but synthesis will rely only on general knowledge and user reqs.

    # --- 2. Extract Requirements ---
    print("\n--- Step 1: Extracting User Requirements ---")
    req_extract_start = time.time()
    extracted_requirements = requirement_extractor.extract_requirements(
        user_conversation_data)
    print(
        f"Requirement extraction took: {time.time() - req_extract_start:.2f}s")
    if not extracted_requirements:
        print("Failed to extract requirements. Exiting pipeline.")
        return None

    # --- 3. Generate RAG Queries ---
    print("\n--- Step 2: Generating RAG Queries ---")
    rag_queries = query_generator.generate_queries(extracted_requirements)
    if not rag_queries:
        print("No RAG queries generated. Synthesis will rely on general knowledge.")
        # Don't exit, allow synthesis to proceed without retrieved context if desired

    # --- 4. Retrieve Context for Queries ---
    print("\n--- Step 3: Retrieving Context from ChromaDB ---")
    retrieval_start_time = time.time()
    all_retrieved_contexts: Dict[str, List[Dict[str, Any]]] = {}

    # Consider if you want to limit queries for production or make it configurable
    # For now, let's process all generated queries if any exist
    if rag_queries:
        print(
            f"Processing {len(rag_queries)} RAG queries for context retrieval...")
        for i, query_text in enumerate(rag_queries):
            # print(f"  Querying for: '{query_text}' ({i+1}/{len(rag_queries)})") # Can be verbose
            query_embedding_list = gemini_embedder.embed_texts(
                texts=[query_text], task_type="RETRIEVAL_QUERY"
            )

            if not query_embedding_list or not query_embedding_list[0]:
                all_retrieved_contexts[query_text] = []
                continue

            query_embedding = query_embedding_list[0]
            retrieved_docs = chroma_manager.query_collection(
                query_embeddings=[
                    query_embedding], n_results=cfg.RAG_NUM_RETRIEVED_CHUNKS
            )

            current_query_contexts = []
            if retrieved_docs and retrieved_docs.get('ids') and retrieved_docs['ids'][0]:
                for j in range(len(retrieved_docs['ids'][0])):
                    current_query_contexts.append({
                        "id": retrieved_docs['ids'][0][j],
                        "text": retrieved_docs['documents'][0][j],
                        "metadata": retrieved_docs['metadatas'][0][j],
                        "distance": retrieved_docs['distances'][0][j]
                    })
            all_retrieved_contexts[query_text] = current_query_contexts
    else:
        print("No RAG queries to process for context retrieval.")
    print(f"Context retrieval took: {time.time() - retrieval_start_time:.2f}s")

    # --- 5. Synthesize Final Output ---
    print("\n--- Step 4: Synthesizing Final Output ---")
    synthesis_start_time = time.time()
    final_output_json = synthesizer.synthesize_output(
        extracted_requirements,
        all_retrieved_contexts  # Pass the retrieved contexts
    )
    print(f"Synthesis took: {time.time() - synthesis_start_time:.2f}s")

    if not final_output_json:
        print("Failed to synthesize final output. Exiting pipeline.")
        return None

    # --- 6. Save Final Output ---
    # Create a unique filename based on the input conversation file
    base_input_filename = os.path.splitext(
        os.path.basename(conversation_json_path))[0]
    output_filename = f"final_synthesized_output_{base_input_filename}.json"
    output_filepath = os.path.join(output_dir, output_filename)

    try:
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        with open(output_filepath, 'w', encoding='utf-8') as f:
            json.dump(final_output_json, f, indent=2, ensure_ascii=False)
        print(
            f"\nSuccessfully saved final synthesized output to: {output_filepath}")
    except Exception as e:
        print(f"Error saving final synthesized output: {e}")
        # Still return the JSON data even if saving failed

    pipeline_end_time = time.time()
    print(f"\n--- Full RAG Pipeline Finished ---")
    print(
        f"Total execution time: {pipeline_end_time - pipeline_start_time:.2f} seconds.")
    return final_output_json


if __name__ == "__main__":
    # --- Setup Command-Line Argument Parsing ---
    parser = argparse.ArgumentParser(
        description="Run the Architectural RAG pipeline.")
    parser.add_argument(
        "input_json_path",
        type=str,
        help="Path to the input user conversation JSON file."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=Config().OUTPUT_JSON_PATH,  # Default to path from config
        help="Directory to save the final synthesized output JSON file."
    )

    args = parser.parse_args()

    # --- Run the Pipeline ---
    if not os.path.exists(args.input_json_path):
        print(
            f"Error: Input conversation JSON file not found at '{args.input_json_path}'")
    else:
        final_result = run_full_rag_pipeline(
            args.input_json_path, args.output_dir)
        if final_result:
            print("\n--- Final Synthesized Output (Snippet) ---")
            # Print a small part of the result for confirmation
            # For brevity, just print the project_summary_assessment keys
            if "project_summary_assessment" in final_result:
                print(json.dumps(
                    final_result["project_summary_assessment"], indent=2))
            else:
                print("Project summary assessment not found in final output.")
            print("-----------------------------------------")
        else:
            print("Pipeline execution failed to produce a final result.")
