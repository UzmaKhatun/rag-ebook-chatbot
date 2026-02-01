"""
LangGraph RAG Pipeline
Orchestrates the retrieval-augmented generation workflow
"""
from typing import TypedDict, List, Dict, Any
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END #Graph
from src.config import Config
from src.retrieval.vector_store import VectorStoreRetriever, RetrievalResult
from src.rag.prompts import (
    SYSTEM_PROMPT,
    get_rag_prompt,
    NO_CONTEXT_RESPONSE,
    GREETING_RESPONSE
)


class RAGState(TypedDict):
    """State for RAG workflow"""
    question: str
    context: str
    retrieved_docs: List[Dict[str, Any]]
    answer: str
    sources: List[str]
    confidence: float
    error: str


class RAGPipeline:
    """LangGraph-based RAG Pipeline"""
    
    def __init__(self):
        """Initialize RAG pipeline components"""
        print("🔧 Initializing RAG Pipeline...")
        
        # Initialize LLM (Groq)
        self.llm = ChatGroq(
            groq_api_key=Config.GROQ_API_KEY,
            model_name=Config.LLM_MODEL,
            temperature=Config.LLM_TEMPERATURE,
            max_tokens=Config.LLM_MAX_TOKENS
        )
        print(f"✅ LLM initialized: {Config.LLM_MODEL}")
        
        # Initialize retriever
        self.retriever = VectorStoreRetriever()
        print("✅ Retriever initialized")
        
        # Build graph
        self.graph = self._build_graph()
        print("✅ RAG Pipeline ready")
    
    def _build_graph(self): #-> Graph:
        """
        Build LangGraph workflow
        
        Returns:
            Compiled StateGraph
        """
        workflow = StateGraph(RAGState)
        
        # Add nodes
        workflow.add_node("retrieve", self._retrieve_node)
        workflow.add_node("generate", self._generate_node)
        workflow.add_node("format", self._format_node)
        
        # Add edges
        workflow.set_entry_point("retrieve")
        workflow.add_edge("retrieve", "generate")
        workflow.add_edge("generate", "format")
        workflow.add_edge("format", END)
        
        return workflow.compile()
    
    def _retrieve_node(self, state: RAGState) -> RAGState:
        """
        Retrieve relevant documents
        
        Args:
            state: Current RAG state
            
        Returns:
            Updated state with retrieved documents
        """
        question = state["question"]
        
        # Check for greetings
        greetings = ["hello", "hi", "hey", "greetings"]
        if any(greeting in question.lower() for greeting in greetings) and len(question.split()) <= 3:
            state["context"] = ""
            state["retrieved_docs"] = []
            state["sources"] = []
            state["answer"] = GREETING_RESPONSE
            return state
        
        try:
            # Retrieve documents with scores
            docs = self.retriever.retrieve_filtered_by_threshold(
                query=question,
                k=Config.TOP_K_RESULTS,
                similarity_threshold=Config.SIMILARITY_THRESHOLD
            )
            
            if not docs:
                state["context"] = ""
                state["retrieved_docs"] = []
                state["sources"] = []
                state["answer"] = NO_CONTEXT_RESPONSE
            else:
                # Format context
                retrieval_result = RetrievalResult(question, docs)
                state["context"] = retrieval_result.get_context()
                state["retrieved_docs"] = docs
                state["sources"] = retrieval_result.get_sources()
            
        except Exception as e:
            state["error"] = f"Retrieval error: {str(e)}"
            state["context"] = ""
            state["retrieved_docs"] = []
        
        return state
    
    def _generate_node(self, state: RAGState) -> RAGState:
        """
        Generate answer using LLM
        
        Args:
            state: Current RAG state
            
        Returns:
            Updated state with generated answer
        """
        # Skip if answer already set (greeting or no context)
        if state.get("answer"):
            return state
        
        question = state["question"]
        context = state["context"]
        
        try:
            # Create messages
            messages = [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=get_rag_prompt(context, question))
            ]
            
            # Generate response
            response = self.llm.invoke(messages)
            state["answer"] = response.content
            
        except Exception as e:
            state["error"] = f"Generation error: {str(e)}"
            state["answer"] = "I apologize, but I encountered an error generating the response."
        
        return state
    
    def _format_node(self, state: RAGState) -> RAGState:
        """
        Format final response with metadata
        
        Args:
            state: Current RAG state
            
        Returns:
            Final formatted state
        """
        # Calculate confidence based on retrieval scores
        if state["retrieved_docs"]:
            avg_score = sum(doc["similarity_score"] for doc in state["retrieved_docs"]) / len(state["retrieved_docs"])
            state["confidence"] = round(avg_score, 4)
        else:
            state["confidence"] = 0.0
        
        return state
    
    def query(self, question: str) -> Dict[str, Any]:
        """
        Process a question through the RAG pipeline
        
        Args:
            question: User question
            
        Returns:
            Dictionary with answer, context, sources, and confidence
        """
        # Initialize state
        initial_state: RAGState = {
            "question": question,
            "context": "",
            "retrieved_docs": [],
            "answer": "",
            "sources": [],
            "confidence": 0.0,
            "error": ""
        }
        
        # Run graph
        final_state = self.graph.invoke(initial_state)
        
        # Format response
        return {
            "question": question,
            "answer": final_state["answer"],
            "context_chunks": final_state["retrieved_docs"],
            "sources": final_state["sources"],
            "confidence": final_state["confidence"],
            "num_chunks": len(final_state["retrieved_docs"]),
            "error": final_state.get("error", "")
        }
    
    def chat(self, question: str, chat_history: List[Dict[str, str]] = None) -> Dict[str, Any]:
        """
        Chat interface with conversation history
        
        Args:
            question: User question
            chat_history: List of previous messages
            
        Returns:
            Response dictionary
        """
        # For now, just use standard query
        # Can be extended to include chat history in context
        return self.query(question)


if __name__ == "__main__":
    # Test the RAG pipeline
    print("\n" + "="*60)
    print("Testing RAG Pipeline")
    print("="*60 + "\n")
    
    pipeline = RAGPipeline()
    
    # Test questions
    test_questions = [
        "What is Agentic AI?",
        "How does Agentic AI differ from traditional AI?",
        "What are the key components of an Agentic AI system?"
    ]
    
    for question in test_questions:
        print(f"\n{'='*60}")
        print(f"Q: {question}")
        print(f"{'='*60}\n")
        
        result = pipeline.query(question)
        
        print(f"Answer:\n{result['answer']}\n")
        print(f"Sources: {', '.join(result['sources'])}")
        print(f"Confidence: {result['confidence']:.2%}")
        print(f"Chunks used: {result['num_chunks']}")