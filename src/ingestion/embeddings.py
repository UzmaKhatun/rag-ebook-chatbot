"""
Embeddings Module
Handles embedding generation and ChromaDB vector store setup
"""
from typing import List, Optional
import chromadb
from chromadb.config import Settings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from src.config import Config
from src.ingestion.pdf_processor import PDFProcessor


class EmbeddingManager:
    """Manage embeddings and ChromaDB operations"""
    
    def __init__(self):
        """Initialize embedding model and ChromaDB client"""
        print(f"🔧 Initializing Embedding Manager...")
        
        # Initialize embedding model
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=Config.EMBEDDING_MODEL,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        print(f"✅ Loaded embedding model: {Config.EMBEDDING_MODEL}")
        
        # Initialize ChromaDB client
        self.chroma_client = chromadb.PersistentClient(
            path=Config.CHROMA_PERSIST_DIRECTORY,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        print(f"✅ ChromaDB initialized at: {Config.CHROMA_PERSIST_DIRECTORY}")
        
        self.vectorstore = None
    
    def create_vectorstore(self, documents: List[Document], force_recreate: bool = False) -> Chroma:
        """
        Create or load ChromaDB vector store
        
        Args:
            documents: List of Document objects to embed
            force_recreate: If True, delete existing collection and recreate
            
        Returns:
            Chroma vectorstore instance
        """
        try:
            # Check if collection exists
            existing_collections = [col.name for col in self.chroma_client.list_collections()]
            collection_exists = Config.CHROMA_COLLECTION_NAME in existing_collections
            
            if collection_exists and force_recreate:
                print(f"🗑️  Deleting existing collection: {Config.CHROMA_COLLECTION_NAME}")
                self.chroma_client.delete_collection(Config.CHROMA_COLLECTION_NAME)
                collection_exists = False
            
            if collection_exists:
                print(f"📂 Loading existing collection: {Config.CHROMA_COLLECTION_NAME}")
                self.vectorstore = Chroma(
                    client=self.chroma_client,
                    collection_name=Config.CHROMA_COLLECTION_NAME,
                    embedding_function=self.embedding_model
                )
                print(f"✅ Loaded {self.vectorstore._collection.count()} existing documents")
            else:
                print(f"🆕 Creating new collection: {Config.CHROMA_COLLECTION_NAME}")
                print(f"📊 Embedding {len(documents)} documents...")
                
                self.vectorstore = Chroma.from_documents(
                    documents=documents,
                    embedding=self.embedding_model,
                    client=self.chroma_client,
                    collection_name=Config.CHROMA_COLLECTION_NAME,
                    persist_directory=Config.CHROMA_PERSIST_DIRECTORY
                )
                
                print(f"✅ Successfully created vectorstore with {len(documents)} documents")
            
            return self.vectorstore
            
        except Exception as e:
            raise Exception(f"Error creating vectorstore: {str(e)}")
    
    def get_vectorstore(self) -> Optional[Chroma]:
        """
        Get existing vectorstore instance
        
        Returns:
            Chroma vectorstore or None if not initialized
        """
        if self.vectorstore is None:
            # Try to load existing collection
            try:
                existing_collections = [col.name for col in self.chroma_client.list_collections()]
                if Config.CHROMA_COLLECTION_NAME in existing_collections:
                    self.vectorstore = Chroma(
                        client=self.chroma_client,
                        collection_name=Config.CHROMA_COLLECTION_NAME,
                        embedding_function=self.embedding_model
                    )
                    print(f"✅ Loaded existing vectorstore")
            except Exception as e:
                print(f"⚠️  Could not load vectorstore: {str(e)}")
        
        return self.vectorstore
    
    def add_documents(self, documents: List[Document]) -> None:
        """
        Add new documents to existing vectorstore
        
        Args:
            documents: List of Document objects to add
        """
        if self.vectorstore is None:
            raise ValueError("Vectorstore not initialized. Call create_vectorstore first.")
        
        try:
            print(f"➕ Adding {len(documents)} new documents...")
            self.vectorstore.add_documents(documents)
            print(f"✅ Successfully added documents")
        except Exception as e:
            raise Exception(f"Error adding documents: {str(e)}")
    
    def get_collection_stats(self) -> dict:
        """
        Get statistics about the current collection
        
        Returns:
            Dictionary with collection statistics
        """
        if self.vectorstore is None:
            return {"status": "not_initialized"}
        
        try:
            collection = self.vectorstore._collection
            count = collection.count()
            
            return {
                "status": "active",
                "collection_name": Config.CHROMA_COLLECTION_NAME,
                "document_count": count,
                "embedding_model": Config.EMBEDDING_MODEL,
                "persist_directory": Config.CHROMA_PERSIST_DIRECTORY
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    def reset_vectorstore(self) -> None:
        """Delete all data and reset vectorstore"""
        try:
            print(f"🗑️  Resetting vectorstore...")
            self.chroma_client.delete_collection(Config.CHROMA_COLLECTION_NAME)
            self.vectorstore = None
            print(f"✅ Vectorstore reset complete")
        except Exception as e:
            print(f"⚠️  Error resetting vectorstore: {str(e)}")


def setup_vectorstore(force_recreate: bool = False) -> Chroma:
    """
    Complete setup: Process PDF and create vectorstore
    
    Args:
        force_recreate: If True, recreate vectorstore from scratch
        
    Returns:
        Chroma vectorstore instance
    """
    print("\n" + "="*60)
    print("🚀 Starting Vectorstore Setup")
    print("="*60 + "\n")
    
    # Step 1: Process PDF
    print("Step 1: Processing PDF...")
    processor = PDFProcessor()
    documents = processor.process()
    
    # Step 2: Create embeddings and vectorstore
    print("\nStep 2: Creating embeddings and vectorstore...")
    embedding_manager = EmbeddingManager()
    vectorstore = embedding_manager.create_vectorstore(documents, force_recreate)
    
    # Step 3: Show stats
    print("\nStep 3: Vectorstore Statistics")
    stats = embedding_manager.get_collection_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print("\n" + "="*60)
    print("✅ Vectorstore Setup Complete!")
    print("="*60 + "\n")
    
    return vectorstore


if __name__ == "__main__":
    # Test the embedding manager
    vectorstore = setup_vectorstore(force_recreate=True)
    
    # Test a simple query
    print("\n🔍 Testing similarity search...")
    results = vectorstore.similarity_search("What is Agentic AI?", k=2)
    
    print(f"\nFound {len(results)} results:")
    for i, doc in enumerate(results, 1):
        print(f"\n--- Result {i} ---")
        print(f"Content: {doc.page_content[:200]}...")
        print(f"Metadata: {doc.metadata}")