"""
Vector Database Setup Script
One-time script to process PDF and create ChromaDB vector store
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ingestion.embeddings import setup_vectorstore
from src.config import Config
import argparse


def main():
    """Main setup function"""
    parser = argparse.ArgumentParser(
        description="Setup ChromaDB vector store from Agentic AI PDF"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force recreate vectorstore (delete existing)"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run test queries after setup"
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("🚀 VECTOR DATABASE SETUP")
    print("="*70 + "\n")
    
    print("📋 Configuration:")
    print(f"  PDF: {Config.PDF_PATH}")
    print(f"  Collection: {Config.CHROMA_COLLECTION_NAME}")
    print(f"  Persist Directory: {Config.CHROMA_PERSIST_DIRECTORY}")
    print(f"  Embedding Model: {Config.EMBEDDING_MODEL}")
    print(f"  Chunk Size: {Config.CHUNK_SIZE}")
    print(f"  Chunk Overlap: {Config.CHUNK_OVERLAP}")
    print(f"  Force Recreate: {args.force}")
    print()
    
    # Confirm if force recreate
    if args.force:
        confirm = input("⚠️  This will DELETE existing vectorstore. Continue? (yes/no): ")
        if confirm.lower() != "yes":
            print("❌ Setup cancelled")
            return
    
    try:
        # Setup vectorstore
        vectorstore = setup_vectorstore(force_recreate=args.force)
        
        # Test queries if requested
        if args.test:
            print("\n" + "="*70)
            print("🧪 RUNNING TEST QUERIES")
            print("="*70 + "\n")
            
            test_queries = [
                "What is Agentic AI?",
                "What are the key components of an Agentic AI system?",
                "How do multi-agent systems work?"
            ]
            
            for i, query in enumerate(test_queries, 1):
                print(f"\n{i}. Query: {query}")
                print("-" * 70)
                
                results = vectorstore.similarity_search(query, k=2)
                
                for j, doc in enumerate(results, 1):
                    print(f"\n  Result {j}:")
                    print(f"  Page: {doc.metadata.get('page', 'Unknown')}")
                    print(f"  Content: {doc.page_content[:200]}...")
        
        print("\n" + "="*70)
        print("✅ SETUP COMPLETE!")
        print("="*70 + "\n")
        print("Next steps:")
        print("  1. Create a .env file (copy from .env.example)")
        print("  2. Add your GROQ_API_KEY to .env")
        print("  3. Run: streamlit run app/streamlit_app.py")
        print()
        
    except Exception as e:
        print(f"\n❌ Setup failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()