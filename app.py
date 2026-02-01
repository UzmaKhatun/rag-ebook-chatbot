"""
Streamlit Chat Interface for RAG Agentic AI Chatbot
"""
import streamlit as st
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.rag.graph import RAGPipeline
from src.config import Config
import time


# Page configuration
st.set_page_config(
    page_title="Agentic AI eBook Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
    }
    .stTextInput > div > div > input {
        background-color: #262730;
        color: #fafafa;
    }
    .chat-message {
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: column;
    }
    .chat-message.user {
        background-color: #1e3a5f;
        border-left: 5px solid #4a9eff;
        color: #ffffff;
    }
    .chat-message.assistant {
        background-color: #1e3d2f;
        border-left: 5px solid #4CAF50;
        color: #ffffff;
    }
    .source-box {
        background-color: #3d2e1e;
        padding: 0.5rem;
        border-radius: 0.3rem;
        margin-top: 0.5rem;
        font-size: 0.85rem;
        color: #ffd699;
    }
    .confidence-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 1rem;
        font-size: 0.8rem;
        font-weight: 600;
        margin-top: 0.5rem;
    }
    .confidence-high {
        background-color: #2d5f2e;
        color: #a5d6a7;
    }
    .confidence-medium {
        background-color: #5f4d2d;
        color: #ffeb99;
    }
    .confidence-low {
        background-color: #5f2d2d;
        color: #ffb3b3;
    }
    /* Make text more visible */
    .stMarkdown, p, span, div {
        color: #fafafa !important;
    }
    /* Sidebar styling */
    .css-1d391kg, .css-1lcbmhc {
        background-color: #262730;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def initialize_pipeline():
    """Initialize RAG pipeline (cached)"""
    try:
        with st.spinner("🔧 Loading RAG Pipeline..."):
            pipeline = RAGPipeline()
        return pipeline, None
    except Exception as e:
        return None, str(e)


def get_confidence_badge(confidence: float) -> str:
    """Generate HTML for confidence badge"""
    if confidence >= 0.7:
        class_name = "confidence-high"
        label = "High Confidence"
    elif confidence >= 0.4:
        class_name = "confidence-medium"
        label = "Medium Confidence"
    else:
        class_name = "confidence-low"
        label = "Low Confidence"
    
    return f'<span class="confidence-badge {class_name}">📊 {label}: {confidence:.0%}</span>'


def display_chat_message(role: str, content: str, metadata: dict = None):
    """Display a chat message with styling"""
    css_class = "user" if role == "user" else "assistant"
    icon = "👤" if role == "user" else "🤖"
    
    with st.container():
        st.markdown(f"""
            <div class="chat-message {css_class}">
                <div style="font-weight: 600; margin-bottom: 0.5rem; color: #ffffff; font-size: 1.1rem;">
                    {icon} {role.capitalize()}
                </div>
                <div style="color: #ffffff; font-size: 1rem; line-height: 1.6;">
                    {content}
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        # Display metadata for assistant messages
        if role == "assistant" and metadata:
            col1, col2 = st.columns([3, 1])
            
            with col1:
                if metadata.get("sources"):
                    sources_text = ", ".join(metadata["sources"])
                    st.markdown(f"""
                        <div class="source-box">
                            📚 <strong>Sources:</strong> {sources_text}
                        </div>
                    """, unsafe_allow_html=True)
            
            with col2:
                if metadata.get("confidence") is not None:
                    st.markdown(
                        get_confidence_badge(metadata["confidence"]),
                        unsafe_allow_html=True
                    )


def display_context_chunks(chunks: list):
    """Display retrieved context chunks in expandable section"""
    if not chunks:
        return
    
    with st.expander(f"📄 View Retrieved Context ({len(chunks)} chunks)", expanded=False):
        for i, chunk in enumerate(chunks, 1):
            st.markdown(f"**Chunk {i}** - Page {chunk['metadata'].get('page', 'Unknown')} "
                       f"(Relevance: {chunk.get('similarity_score', 0):.2%})")
            st.text_area(
                f"Content {i}",
                chunk['content'],
                height=150,
                key=f"chunk_{i}",
                label_visibility="collapsed"
            )
            st.divider()


def main():
    """Main Streamlit application"""
    
    # Header
    st.title("📚 Agentic AI Chatbot")
    st.markdown("""
        Ask me anything about **Agentic AI** based on the comprehensive eBook: 
        *"Agentic AI: An Executive's Guide to In-depth Understanding of Agentic AI"*
    """)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Pipeline status
        st.subheader("📊 System Status")
        pipeline, error = initialize_pipeline()
        
        if pipeline:
            st.success("✅ RAG Pipeline Active")
            st.info(f"🔧 Model: {Config.LLM_MODEL}")
            st.info(f"🔍 Top-K Results: {Config.TOP_K_RESULTS}")
            st.info(f"📏 Chunk Size: {Config.CHUNK_SIZE}")
        else:
            st.error(f"❌ Pipeline Error: {error}")
            st.stop()
        
        st.divider()
        
        # Controls
        st.subheader("🎛️ Controls")
        
        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
        
        # Display settings
        show_context = st.checkbox("Show Retrieved Context", value=False)
        show_metadata = st.checkbox("Show Metadata", value=True)
        
        st.divider()
        
        # Sample questions
        st.subheader("💡 Sample Questions")
        sample_questions = [
            "What is Agentic AI?",
            "How does Agentic AI differ from traditional AI?",
            "What are the components of an Agentic AI system?",
            "What are multi-agent systems?",
            "How can organizations assess their readiness for Agentic AI?",
            "What are real-world applications of Agentic AI?"
        ]
        
        for q in sample_questions:
            if st.button(q, key=f"sample_{q}", use_container_width=True):
                st.session_state.current_question = q
        
        st.divider()
        
        # About
        with st.expander("ℹ️ About"):
            st.markdown("""
                **RAG Agentic AI Chatbot**
                
                Built with:
                - LangGraph for orchestration
                - ChromaDB for vector storage
                - Groq LLM for generation
                - Streamlit for UI
                
                This chatbot answers questions strictly based on the 
                Agentic AI eBook knowledge base.
            """)
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat history
    for message in st.session_state.messages:
        display_chat_message(
            message["role"],
            message["content"],
            message.get("metadata")
        )
        
        # Show context if enabled
        if show_context and message["role"] == "assistant":
            if "context_chunks" in message.get("metadata", {}):
                display_context_chunks(message["metadata"]["context_chunks"])
    
    # Handle sample question click
    if hasattr(st.session_state, 'current_question'):
        user_input = st.session_state.current_question
        delattr(st.session_state, 'current_question')
    else:
        # Chat input
        user_input = st.chat_input("Ask a question about Agentic AI...")
    
    # Process user input
    if user_input:
        # Add user message to history
        st.session_state.messages.append({
            "role": "user",
            "content": user_input
        })
        
        # Display user message
        display_chat_message("user", user_input)
        
        # Generate response
        with st.spinner("🤔 Thinking..."):
            try:
                response = pipeline.query(user_input)
                
                # Add assistant message to history
                assistant_message = {
                    "role": "assistant",
                    "content": response["answer"],
                    "metadata": {
                        "sources": response["sources"],
                        "confidence": response["confidence"],
                        "num_chunks": response["num_chunks"],
                        "context_chunks": response["context_chunks"] if show_context else []
                    }
                }
                st.session_state.messages.append(assistant_message)
                
                # Display assistant message
                display_chat_message(
                    "assistant",
                    response["answer"],
                    assistant_message["metadata"] if show_metadata else None
                )
                
                # Show context if enabled
                if show_context and response["context_chunks"]:
                    display_context_chunks(response["context_chunks"])
                
            except Exception as e:
                error_message = f"❌ Error: {str(e)}"
                st.error(error_message)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_message
                })
        
        # Rerun to update chat
        st.rerun()
    
    # Footer
    st.markdown("---")
    st.markdown("""
        <div style='text-align: center; color: #666; font-size: 0.85rem;'>
            💡 Powered by LangGraph + ChromaDB + Groq
        </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()