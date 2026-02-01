"""
Vector Store Retrieval Module
Handles querying ChromaDB for relevant documents
"""
from typing import List, Dict, Any, Tuple
from langchain_chroma import Chroma
from langchain_core.documents import Document
from src.config import Config
from src.ingestion.embeddings import EmbeddingManager


class VectorStoreRetriever:
    """Retrieve relevant documents from ChromaDB"""
    
    def __init__(self, vectorstore: Chroma = None):
        """
        Initialize retriever
        
        Args:
            vectorstore: Optional pre-initialized Chroma vectorstore
        """
        if vectorstore:
            self.vectorstore = vectorstore
        else:
            # Load existing vectorstore
            embedding_manager = EmbeddingManager()
            self.vectorstore = embedding_manager.get_vectorstore()
            
            if self.vectorstore is None:
                raise ValueError(
                    "No vectorstore found. Please run setup_vectordb.py first."
                )
        
        print("✅ VectorStore Retriever initialized")
    
    def retrieve(
        self, 
        query: str, 
        k: int = None,
        filter_dict: Dict[str, Any] = None
    ) -> List[Document]:
        """
        Retrieve relevant documents for a query
        
        Args:
            query: Search query
            k: Number of documents to retrieve (default: Config.TOP_K_RESULTS)
            filter_dict: Optional metadata filters
            
        Returns:
            List of relevant Document objects
        """
        k = k or Config.TOP_K_RESULTS
        
        try:
            if filter_dict:
                results = self.vectorstore.similarity_search(
                    query,
                    k=k,
                    filter=filter_dict
                )
            else:
                results = self.vectorstore.similarity_search(query, k=k)
            
            return results
            
        except Exception as e:
            print(f"⚠️  Error retrieving documents: {str(e)}")
            return []
    
    def retrieve_with_scores(
        self,
        query: str,
        k: int = None,
        filter_dict: Dict[str, Any] = None
    ) -> List[Tuple[Document, float]]:
        """
        Retrieve documents with similarity scores
        
        Args:
            query: Search query
            k: Number of documents to retrieve
            filter_dict: Optional metadata filters
            
        Returns:
            List of tuples (Document, similarity_score)
        """
        k = k or Config.TOP_K_RESULTS
        
        try:
            if filter_dict:
                results = self.vectorstore.similarity_search_with_score(
                    query,
                    k=k,
                    filter=filter_dict
                )
            else:
                results = self.vectorstore.similarity_search_with_score(query, k=k)
            
            return results
            
        except Exception as e:
            print(f"⚠️  Error retrieving documents with scores: {str(e)}")
            return []
    
    def retrieve_filtered_by_threshold(
        self,
        query: str,
        k: int = None,
        similarity_threshold: float = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve documents above similarity threshold
        
        Args:
            query: Search query
            k: Number of documents to retrieve
            similarity_threshold: Minimum similarity score (0-1)
            
        Returns:
            List of dicts with document content, metadata, and scores
        """
        k = k or Config.TOP_K_RESULTS
        threshold = similarity_threshold or Config.SIMILARITY_THRESHOLD
        
        # Get results with scores
        results_with_scores = self.retrieve_with_scores(query, k=k * 2)  # Get more to filter
        
        # Filter by threshold and format
        filtered_results = []
        for doc, score in results_with_scores:
            # Convert distance to similarity (ChromaDB returns distance, lower is better)
            # For L2 distance, we invert it to get similarity
            similarity = 1 / (1 + score)  # Simple conversion
            
            if similarity >= threshold:
                filtered_results.append({
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "similarity_score": round(similarity, 4)
                })
            
            # Stop if we have enough results
            if len(filtered_results) >= k:
                break
        
        return filtered_results
    
    def format_context(self, documents: List[Document]) -> str:
        """
        Format retrieved documents into context string
        
        Args:
            documents: List of retrieved documents
            
        Returns:
            Formatted context string
        """
        if not documents:
            return "No relevant context found."
        
        context_parts = []
        for i, doc in enumerate(documents, 1):
            page_info = f"[Page {doc.metadata.get('page', 'Unknown')}]"
            context_parts.append(f"Context {i} {page_info}:\n{doc.page_content}\n")
        
        return "\n".join(context_parts)
    
    def get_retriever(self):
        """
        Get LangChain retriever interface
        
        Returns:
            LangChain Retriever object
        """
        return self.vectorstore.as_retriever(
            search_kwargs={"k": Config.TOP_K_RESULTS}
        )


class RetrievalResult:
    """Structured retrieval result with metadata"""
    
    def __init__(self, query: str, documents: List[Dict[str, Any]]):
        """
        Initialize retrieval result
        
        Args:
            query: Original search query
            documents: List of document dicts with content, metadata, scores
        """
        self.query = query
        self.documents = documents
        self.num_results = len(documents)
    
    def get_context(self) -> str:
        """Get formatted context from all documents"""
        if not self.documents:
            return "No relevant information found in the knowledge base."
        
        context_parts = []
        for i, doc in enumerate(self.documents, 1):
            page = doc['metadata'].get('page', 'Unknown')
            score = doc.get('similarity_score', 0)
            context_parts.append(
                f"[Source {i} - Page {page} - Relevance: {score:.2%}]\n"
                f"{doc['content']}\n"
            )
        
        return "\n---\n".join(context_parts)
    
    def get_sources(self) -> List[str]:
        """Get list of source pages"""
        sources = []
        for doc in self.documents:
            page = doc['metadata'].get('page', 'Unknown')
            sources.append(f"Page {page}")
        return sources
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format"""
        return {
            "query": self.query,
            "num_results": self.num_results,
            "documents": self.documents,
            "context": self.get_context(),
            "sources": self.get_sources()
        }


if __name__ == "__main__":
    # Test the retriever
    print("Testing VectorStore Retriever...\n")
    
    retriever = VectorStoreRetriever()
    
    # Test query
    test_query = "What is Agentic AI?"
    print(f"Query: {test_query}\n")
    
    # Retrieve with scores
    results = retriever.retrieve_filtered_by_threshold(test_query, k=3)
    
    print(f"Found {len(results)} results:\n")
    for i, result in enumerate(results, 1):
        print(f"--- Result {i} ---")
        print(f"Similarity: {result['similarity_score']:.4f}")
        print(f"Page: {result['metadata'].get('page')}")
        print(f"Content: {result['content'][:200]}...")
        print()
    
    # Test RetrievalResult
    retrieval_result = RetrievalResult(test_query, results)
    print(f"\n📊 Retrieval Statistics:")
    print(f"Query: {retrieval_result.query}")
    print(f"Results: {retrieval_result.num_results}")
    print(f"Sources: {', '.join(retrieval_result.get_sources())}")