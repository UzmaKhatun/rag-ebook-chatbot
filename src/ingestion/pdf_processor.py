"""
PDF Processing Module
Handles loading and chunking of the Agentic AI eBook
"""
from typing import List, Dict
from pathlib import Path
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from src.config import Config


class PDFProcessor:
    """Process PDF documents for RAG pipeline"""
    
    def __init__(self, pdf_path: str = None):
        """
        Initialize PDF processor
        
        Args:
            pdf_path: Path to PDF file (defaults to Config.PDF_PATH)
        """
        self.pdf_path = Path(pdf_path) if pdf_path else Config.PDF_PATH
        
        if not self.pdf_path.exists():
            raise FileNotFoundError(f"PDF not found at {self.pdf_path}")
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
    
    def load_pdf(self) -> str:
        """
        Load and extract text from PDF
        
        Returns:
            Extracted text from all pages
        """
        try:
            reader = PdfReader(str(self.pdf_path))
            text = ""
            
            for page_num, page in enumerate(reader.pages, start=1):
                page_text = page.extract_text()
                if page_text:
                    text += f"\n\n--- Page {page_num} ---\n\n{page_text}"
            
            print(f"✅ Loaded {len(reader.pages)} pages from PDF")
            return text
            
        except Exception as e:
            raise Exception(f"Error loading PDF: {str(e)}")
    
    def chunk_text(self, text: str) -> List[Document]:
        """
        Split text into chunks with metadata
        
        Args:
            text: Full text to chunk
            
        Returns:
            List of Document objects with chunks and metadata
        """
        try:
            # Split into pages first
            pages = text.split('\n\n--- Page ')
            
            # Process each page separately
            all_documents = []
            chunk_id = 0
            
            for page_text in pages:
                if not page_text.strip():
                    continue
                
                # Extract page number from the start of page_text
                lines = page_text.split('\n', 1)
                if len(lines) > 1 and lines[0].strip().replace(' ---', '').isdigit():
                    page_num = int(lines[0].strip().replace(' ---', ''))
                    page_content = lines[1] if len(lines) > 1 else ""
                else:
                    # First page doesn't have the marker
                    page_num = 1
                    page_content = page_text
                
                # Chunk this page
                page_chunks = self.text_splitter.split_text(page_content)
                
                # Create documents for each chunk
                for chunk in page_chunks:
                    doc = Document(
                        page_content=chunk,
                        metadata={
                            "chunk_id": chunk_id,
                            "page": page_num,
                            "source": "Ebook-Agentic-AI.pdf",
                        }
                    )
                    all_documents.append(doc)
                    chunk_id += 1
            
            # Update total_chunks in all documents
            for doc in all_documents:
                doc.metadata["total_chunks"] = len(all_documents)
            
            print(f"✅ Created {len(all_documents)} chunks from {page_num} pages")
            return all_documents
            
        except Exception as e:
            raise Exception(f"Error chunking text: {str(e)}")
    
    def _extract_page_number(self, chunk: str) -> int:
        """
        Extract page number from chunk text
        
        Args:
            chunk: Text chunk
            
        Returns:
            Page number or 0 if not found
        """
        import re
        match = re.search(r'--- Page (\d+) ---', chunk)
        return int(match.group(1)) if match else 0
    
    def process(self) -> List[Document]:
        """
        Complete processing pipeline: load + chunk
        
        Returns:
            List of chunked documents with metadata
        """
        print(f"📄 Processing PDF: {self.pdf_path.name}")
        text = self.load_pdf()
        documents = self.chunk_text(text)
        print(f"✅ PDF processing complete: {len(documents)} chunks created")
        return documents


if __name__ == "__main__":
    # Test the processor
    processor = PDFProcessor()
    docs = processor.process()
    
    print(f"\n📊 Sample chunk:")
    print(f"Content: {docs[0].page_content[:200]}...")
    print(f"Metadata: {docs[0].metadata}")