# 🤖 RAG Agentic AI Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that answers questions strictly based on the **"Agentic AI: An Executive's Guide"** eBook using LangGraph, ChromaDB, and Groq LLM.

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![LangChain](https://img.shields.io/badge/LangChain-Latest-green)](https://www.langchain.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30-red)](https://streamlit.io/)

---

## 🎯 Overview

This RAG chatbot provides accurate, grounded answers about Agentic AI by:
- **Retrieving** relevant context from the eBook using ChromaDB vector search
- **Augmenting** user queries with retrieved context
- **Generating** precise answers using Groq's LLM (llama-3.1-70b)
- **Orchestrating** the workflow using LangGraph state machines

### Key Highlights
✅ Answers **strictly based** on the provided eBook  
✅ Cites **page numbers** for transparency  
✅ Shows **confidence scores** and retrieved context  
✅ Built with **LangGraph** for robust orchestration  
✅ Clean, modular, and **production-ready** code  

---

## ✨ Features

### Core Capabilities
- 🔍 **Semantic Search**: ChromaDB vector store with similarity search
- 🧠 **Intelligent Retrieval**: Top-K results with similarity threshold filtering
- 💬 **Natural Conversations**: Context-aware responses using Groq LLM
- 📊 **Confidence Scoring**: Shows reliability of each answer
- 📚 **Source Attribution**: Cites exact page numbers from the eBook
- 🎨 **Interactive UI**: Beautiful Streamlit chat interface

### Technical Features
- **LangGraph Orchestration**: State-based workflow (Retrieve → Generate → Format)
- **Persistent Storage**: ChromaDB with automatic persistence
- **Embeddings**: HuggingFace all-MiniLM-L6-v2 model
- **Chunking Strategy**: Recursive text splitting with overlap
- **Error Handling**: Graceful fallbacks at every stage
- **Caching**: Optimized pipeline initialization

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Streamlit UI                              │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              LangGraph RAG Pipeline                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │   Retrieve   │→ │   Generate   │→ │    Format    │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└──────────┬────────────────────────────────────┬─────────────┘
           │                                    │
           ▼                                    ▼
    ┌──────────────┐                   ┌──────────────┐
    │  ChromaDB    │                   │  Groq LLM    │
    │  Vector DB   │                   │  (llama-3.1) │
    └──────────────┘                   └──────────────┘
```


---

## 🛠️ Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Orchestration** | LangGraph | Workflow state management |
| **Vector Database** | ChromaDB | Persistent vector storage |
| **Embeddings** | HuggingFace (all-MiniLM-L6-v2) | Text-to-vector conversion |
| **LLM** | Groq (llama-3.1-70b-versatile) | Answer generation |
| **PDF Processing** | PyPDF | Document parsing |
| **Text Splitting** | LangChain RecursiveCharacterTextSplitter | Intelligent chunking |
| **UI** | Streamlit | Interactive chat interface |
| **Language** | Python 3.8+ | Core implementation |

---

## 📦 Installation

### Prerequisites
- Python 3.8 or higher
- Groq API key ([Get it here](https://console.groq.com/))
- Git

### Step 1: Clone the Repository
```bash
git clone <your-repo-url>
cd rag-agentic-ai-chatbot
```

### Step 2: Create Virtual Environment
```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Setup Environment Variables
```bash
# Copy the example file
cp .env.example .env

# Edit .env and add your Groq API key
# GROQ_API_KEY=your_groq_api_key_here
```

### Step 5: Add the PDF
```bash
# Place the eBook PDF in the data directory
cp /path/to/Ebook-Agentic-AI.pdf data/
```

### Step 6: Setup Vector Database
```bash
# Run the setup script to process PDF and create ChromaDB
python scripts/setup_vectordb.py

# To force recreate (delete existing):
python scripts/setup_vectordb.py --force

# To run test queries after setup:
python scripts/setup_vectordb.py --test
```

---

## 🚀 Usage

### Run the Streamlit App
```bash
streamlit run app/streamlit_app.py
```

The app will open in your browser at `http://localhost:8501`

### Run Test Queries
```bash
python scripts/test_queries.py
```

### Test Individual Components
```bash
# Test PDF processing
python src/ingestion/pdf_processor.py

# Test embeddings
python src/ingestion/embeddings.py

# Test retrieval
python src/retrieval/vector_store.py

# Test RAG pipeline
python src/rag/graph.py
```

---

## 📁 Project Structure

```
rag-agentic-ai-chatbot/
│
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── .env.example                       # Environment variables template
├── .gitignore                         # Git ignore rules
│
├── data/
│   ├── Ebook-Agentic-AI.pdf          # Source eBook
│   └── chroma_db/                     # ChromaDB storage (auto-created)
│
├── src/
│   ├── config.py                      # Configuration management
│   │
│   ├── ingestion/
│   │   ├── pdf_processor.py          # PDF loading & chunking
│   │   └── embeddings.py             # Embeddings & ChromaDB setup
│   │
│   ├── retrieval/
│   │   └── vector_store.py           # Vector similarity search
│   │
│   ├── rag/
│   │   ├── prompts.py                # LLM prompts & templates
│   │   └── graph.py                  # LangGraph RAG pipeline
│   │
│   └── utils/
│       └── helpers.py                # Utility functions
│
├── app/
│   └── streamlit_app.py              # Streamlit UI
│
├── scripts/
│   ├── setup_vectordb.py             # One-time vectorstore setup
│   └── test_queries.py               # Test sample queries
│
├── tests/
│   └── test_rag.py                   # Unit tests
│
└── docs/
    ├── architecture.md               # Architecture documentation
    └── sample_queries.md             # Sample questions & responses
```

---

## 💬 Sample Queries

Try asking:

1. **Definitions**
   - "What is Agentic AI?"
   - "How does Agentic AI differ from traditional AI?"

2. **Technical Components**
   - "What are the key components of an Agentic AI system?"
   - "Explain the BDI model in Agentic AI"

3. **Multi-Agent Systems**
   - "What are multi-agent systems and their benefits?"
   - "What are the challenges in orchestrating Agentic AI?"

4. **Implementation**
   - "How can organizations assess their readiness for Agentic AI?"
   - "What are real-world applications of Agentic AI in healthcare?"


---

## ⚙️ Configuration

### Environment Variables

Create a `.env` file with:

```env
# Required
GROQ_API_KEY=your_groq_api_key_here

# Optional (defaults shown)
CHROMA_PERSIST_DIRECTORY=./data/chroma_db
CHROMA_COLLECTION_NAME=agentic_ai_ebook
EMBEDDING_MODEL=all-MiniLM-L6-v2
LLM_MODEL=llama-3.1-70b-versatile
LLM_TEMPERATURE=0.1
LLM_MAX_TOKENS=1024
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
TOP_K_RESULTS=5
SIMILARITY_THRESHOLD=0.7
```

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `CHUNK_SIZE` | 1000 | Characters per chunk |
| `CHUNK_OVERLAP` | 200 | Overlap between chunks |
| `TOP_K_RESULTS` | 5 | Number of chunks to retrieve |
| `SIMILARITY_THRESHOLD` | 0.7 | Minimum similarity score (0-1) |
| `LLM_TEMPERATURE` | 0.1 | Lower = more factual |

---

## 🔧 Troubleshooting

### Common Issues

#### 1. "GROQ_API_KEY not found"
```bash
# Make sure .env file exists and contains:
GROQ_API_KEY=your_actual_key_here
```

#### 2. "PDF not found"
```bash
# Ensure PDF is in data directory:
ls data/Ebook-Agentic-AI.pdf

# If not, copy it:
cp /path/to/pdf data/
```

#### 3. "No vectorstore found"
```bash
# Run setup script:
python scripts/setup_vectordb.py
```

#### 4. ChromaDB errors
```bash
# Delete and recreate:
rm -rf data/chroma_db
python scripts/setup_vectordb.py --force
```

#### 5. Dependency issues
```bash
# Upgrade pip and reinstall:
pip install --upgrade pip
pip install -r requirements.txt --upgrade
```

---

## 🧪 Testing

### Run All Tests
```bash
# Test complete pipeline
python scripts/test_queries.py

# Test individual components
python src/ingestion/pdf_processor.py
python src/ingestion/embeddings.py
python src/retrieval/vector_store.py
python src/rag/graph.py
```

### Expected Output
- ✅ 8 sample queries processed
- ✅ Average confidence: ~75-85%
- ✅ All responses grounded in eBook
- ✅ Page citations included

---

## 🎨 UI Features

### Chat Interface
- 💬 Real-time conversation
- 📊 Confidence indicators (High/Medium/Low)
- 📚 Source citations with page numbers
- 🔍 View retrieved context chunks
- 💡 Sample question suggestions
- 🗑️ Clear chat history

### Settings Panel
- Toggle context visibility
- Toggle metadata display
- System status monitoring
- Collection statistics

---

## 🔒 Best Practices

1. **API Keys**: Never commit `.env` to version control
2. **Vector Store**: Backup `chroma_db` directory for production
3. **Chunking**: Adjust `CHUNK_SIZE` based on your document structure
4. **Temperature**: Keep low (0.1-0.3) for factual accuracy
5. **Top-K**: Increase for complex questions, decrease for speed

---

## 🚧 Known Limitations

1. **Context Length**: Limited by LLM max tokens (1024)
2. **Single Document**: Currently supports one PDF
3. **No Multi-turn Memory**: Each query is independent (can be extended)
4. **Language**: English only (based on eBook)

---

## 🔮 Future Enhancements

- [ ] Multi-document support
- [ ] Conversation history/memory
- [ ] Answer verification node
- [ ] Query enhancement preprocessing
- [ ] Multi-language support
- [ ] API endpoint (FastAPI)
- [ ] Docker containerization
- [ ] Cloud deployment guide

---

## 👨‍💻 Author

**Uzma Khatun**  
📧 uzmakhatun0205@gmail.com  
🔗 [LinkedIn](https://www.linkedin.com/in/uzma-khatun-88b990334/)  
🔗 [GitHub](https://github.com/UzmaKhatun)

---

## 📄 License

This project was created as part of an interview assignment for the AI Engineer Intern role.

---

## 🙏 Acknowledgments

- **Konverge AI & Emergence AI** for the comprehensive Agentic AI eBook
- **Groq** for fast LLM inference
- **LangChain** for the RAG framework
- **ChromaDB** for vector storage

---

**Built with ❤️**
