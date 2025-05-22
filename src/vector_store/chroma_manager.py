# ArchitecturalRAGSystem/src/vector_store/chroma_manager.py
import chromadb
from chromadb.api.models.Collection import Collection as ChromaCollection # For type hinting
from typing import List, Dict, Any, Optional, Union
import os
import uuid

# from src.config import Config # We'll likely pass config values or the instance in

class ChromaManager:
    """
    Manages interactions with a ChromaDB vector store.
    """
    def __init__(self, path: str, collection_name: str):
        """
        Initializes the ChromaManager and connects to or creates a collection.

        Args:
            path (str): The file system path to persist ChromaDB data.
            collection_name (str): The name of the collection to use.
        """
        self.path = path
        self.collection_name = collection_name
        self.client = chromadb.PersistentClient(path=self.path)
        try:
            self.collection: ChromaCollection = self.client.get_collection(name=self.collection_name)
            print(f"ChromaDB: Collection '{self.collection_name}' loaded. Contains {self.collection.count()} items.")
        except Exception: # Replace with more specific ChromaDB exception if available
            print(f"ChromaDB: Collection '{self.collection_name}' not found. Creating new collection.")
            self.collection = self.client.create_collection(name=self.collection_name)
            print(f"ChromaDB: Collection '{self.collection_name}' created.")

    def add_documents(self,
                      ids: List[str],
                      embeddings: List[List[float]],
                      metadatas: List[Dict[str, Any]],
                      documents: List[str], # The actual text content
                      batch_size: int = 100
                      ) -> None:
        """
        Adds documents (with their embeddings and metadata) to the ChromaDB collection in batches.

        Args:
            ids (List[str]): A list of unique IDs for the documents.
            embeddings (List[List[float]]): A list of vector embeddings.
            metadatas (List[Dict[str, Any]]): A list of metadata dictionaries.
            documents (List[str]): A list of the actual text content for each document.
            batch_size (int): How many documents to add in a single call to ChromaDB.
        """
        if not (len(ids) == len(embeddings) == len(metadatas) == len(documents)):
            print("Error: Lengths of ids, embeddings, metadatas, and documents must match.")
            return

        if not ids:
            print("No documents to add.")
            return
        
        num_added_successfully = 0
        for i in range(0, len(ids), batch_size):
            batch_ids = ids[i:i + batch_size]
            batch_embeddings = embeddings[i:i + batch_size]
            batch_metadatas = metadatas[i:i + batch_size]
            batch_documents = documents[i:i + batch_size]

            try:
                # Check for existing IDs in this batch to avoid errors if upsert isn't default
                # or to explicitly manage updates vs. adds if needed.
                # For now, Chroma's add will typically upsert if ID exists, or you can use upsert().
                # Let's assume add() handles this or we manage deduplication before calling.
                print(f"  Adding batch of {len(batch_ids)} items to ChromaDB collection '{self.collection_name}'...")
                self.collection.add(
                    ids=batch_ids,
                    embeddings=batch_embeddings,
                    metadatas=batch_metadatas,
                    documents=batch_documents
                )
                num_added_successfully += len(batch_ids)
                print(f"    Successfully added batch to ChromaDB.")
            except Exception as e:
                print(f"    Error adding batch to ChromaDB: {e}")
                # Potentially log failed IDs or implement more granular retry
        
        print(f"Finished adding documents. Total added in this call: {num_added_successfully}.")
        print(f"Collection '{self.collection_name}' now contains {self.collection.count()} items.")


    def query_collection(self,
                         query_embeddings: List[List[float]],
                         n_results: int = 5,
                         where_filter: Optional[Dict[str, Any]] = None,
                         where_document_filter: Optional[Dict[str, Any]] = None, # For $contains on documents
                         include: List[str] = ['metadatas', 'documents', 'distances']
                         ) -> Optional[Dict[str, Any]]:
        """
        Queries the ChromaDB collection for similar documents.

        Args:
            query_embeddings (List[List[float]]): A list of query embeddings.
            n_results (int): The number of results to return per query embedding.
            where_filter (Optional[Dict[str, Any]]): Metadata filter.
            where_document_filter (Optional[Dict[str, Any]]): Document content filter.
            include (List[str]): List of fields to include in the results.

        Returns:
            Optional[Dict[str, Any]]: The query results, or None if an error occurs.
        """
        if not query_embeddings:
            print("Error: No query embeddings provided.")
            return None
        try:
            results = self.collection.query(
                query_embeddings=query_embeddings,
                n_results=n_results,
                where=where_filter,
                where_document=where_document_filter,
                include=include
            )
            return results
        except Exception as e:
            print(f"Error querying ChromaDB collection: {e}")
            return None

    def get_document_by_id(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves a document by its ID."""
        try:
            result = self.collection.get(ids=[doc_id], include=['metadatas', 'documents'])
            if result and result['ids']:
                return {
                    "id": result['ids'][0],
                    "document": result['documents'][0] if result['documents'] else None,
                    "metadata": result['metadatas'][0] if result['metadatas'] else None
                }
            return None
        except Exception as e:
            print(f"Error getting document by ID '{doc_id}': {e}")
            return None

    def count(self) -> int:
        """Returns the number of items in the collection."""
        return self.collection.count()

    def clear_collection(self) -> None:
        """Deletes all items from the collection. Use with caution!"""
        print(f"Warning: Clearing all {self.count()} items from collection '{self.collection_name}'!")
        # For safety, you might want a confirmation step here in a real app.
        # A simple way to clear is to delete and recreate, if allowed by use case.
        # Or iterate and delete, though less efficient for full clear.
        current_count = self.collection.count()
        if current_count > 0:
            all_items = self.collection.get(limit=current_count) # Get all item IDs
            if all_items['ids']:
                self.collection.delete(ids=all_items['ids'])
                print(f"Successfully deleted {len(all_items['ids'])} items.")
            else:
                print("No items found to delete, though count was > 0. This is unexpected.")
        print(f"Collection '{self.collection_name}' now contains {self.collection.count()} items.")


# --- Example Usage (can be run directly for testing this module) ---
if __name__ == '__main__':
    print("Testing ChromaManager...")
    
    # For direct testing, we need to load config to get paths
    import sys
    # Add project root to sys.path to allow importing src.config
    project_root_for_test = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    sys.path.insert(0, project_root_for_test)
    from src.config import Config

    cfg = Config() # Load configuration

    # Initialize ChromaManager
    chroma_manager = ChromaManager(path=cfg.CHROMA_DB_PATH, collection_name=cfg.COLLECTION_NAME)
    initial_count = chroma_manager.count()
    print(f"Initial item count: {initial_count}")

    # Sample data for testing
    sample_ids = [str(uuid.uuid4()) for _ in range(3)]
    sample_embeddings = [[0.1 * i, 0.2 * i, -0.1 * i] for i in range(1, 4)] # Dummy embeddings (replace with real ones)
    sample_metadatas = [
        {"source": "test_book", "page": i+1, "topic": "general"} for i in range(3)
    ]
    sample_documents = [
        "This is test document one about architecture.",
        "Another test document focusing on design principles.",
        "The final test document for ChromaDB integration."
    ]

    # Add documents
    print("\nTesting add_documents...")
    # To avoid re-adding same static test data, check if already added
    # A better test would use truly unique IDs each run or clear collection
    test_item_exists = chroma_manager.get_document_by_id(sample_ids[0])
    if not test_item_exists:
        chroma_manager.add_documents(
            ids=sample_ids,
            embeddings=sample_embeddings, # Use dummy embeddings for this test
            metadatas=sample_metadatas,
            documents=sample_documents
        )
    else:
        print("Sample documents likely already exist. Skipping add for this test run.")


    # Query collection (using one of the dummy embeddings as a query)
    print("\nTesting query_collection...")
    if sample_embeddings:
        query_results = chroma_manager.query_collection(
            query_embeddings=[sample_embeddings[0]], # Query with the first dummy embedding
            n_results=2
        )
        if query_results and query_results.get('ids') and query_results['ids'][0]:
            print("Query Results:")
            for i in range(len(query_results['ids'][0])):
                print(f"  ID: {query_results['ids'][0][i]}, Distance: {query_results['distances'][0][i]:.4f}, Doc: {query_results['documents'][0][i][:30]}...")
        else:
            print("  No results from query or error occurred.")

    # Get a document by ID
    print("\nTesting get_document_by_id...")
    if sample_ids:
        doc = chroma_manager.get_document_by_id(sample_ids[0])
        if doc:
            print(f"  Retrieved doc by ID '{sample_ids[0]}': {doc['document'][:50]}...")
        else:
            print(f"  Could not retrieve doc by ID '{sample_ids[0]}'")

    # Count after operations
    print(f"\nItem count after operations: {chroma_manager.count()}")

    # Example of clearing (use with caution - uncomment to test)
    # print("\nTesting clear_collection...")
    # chroma_manager.clear_collection()
    # print(f"Item count after clearing: {chroma_manager.count()}")

    print("\nChromaManager test finished.")