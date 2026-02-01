"""
Test Queries Script
Test the RAG pipeline with sample questions
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.rag.graph import RAGPipeline
import json
from datetime import datetime


# Sample test queries covering different aspects
TEST_QUERIES = [
    {
        "id": 1,
        "question": "What is Agentic AI?",
        "category": "Definition"
    },
    {
        "id": 2,
        "question": "How does Agentic AI differ from traditional AI and non-agentic AI?",
        "category": "Comparison"
    },
    {
        "id": 3,
        "question": "What are the key components of an Agentic AI system?",
        "category": "Technical"
    },
    {
        "id": 4,
        "question": "Explain multi-agent systems and their benefits.",
        "category": "Multi-Agent Systems"
    },
    {
        "id": 5,
        "question": "What are the challenges in orchestrating Agentic AI systems?",
        "category": "Challenges"
    },
    {
        "id": 6,
        "question": "How can organizations assess their readiness for Agentic AI?",
        "category": "Implementation"
    },
    {
        "id": 7,
        "question": "What are some real-world applications of Agentic AI in healthcare?",
        "category": "Use Cases"
    },
    {
        "id": 8,
        "question": "What is the BDI model in Agentic AI?",
        "category": "Technical Concepts"
    }
]


def print_separator(char="=", length=80):
    """Print a separator line"""
    print(char * length)


def print_result(query_num: int, query_info: dict, result: dict):
    """Print formatted query result"""
    print_separator()
    print(f"QUERY {query_num}: {query_info['category']}")
    print_separator()
    print(f"\n❓ Question: {query_info['question']}\n")
    
    print(f"📊 Metadata:")
    print(f"  • Chunks Retrieved: {result['num_chunks']}")
    print(f"  • Confidence: {result['confidence']:.2%}")
    print(f"  • Sources: {', '.join(result['sources']) if result['sources'] else 'None'}")
    
    print(f"\n💬 Answer:")
    print(f"{result['answer']}\n")
    
    if result.get('error'):
        print(f"⚠️  Error: {result['error']}\n")
    
    # Show context chunks summary
    if result['context_chunks']:
        print(f"📄 Retrieved Context Chunks:")
        for i, chunk in enumerate(result['context_chunks'], 1):
            print(f"  {i}. Page {chunk['metadata'].get('page', 'N/A')} - "
                  f"Relevance: {chunk.get('similarity_score', 0):.2%}")
    
    print()


def save_results_to_file(results: list, filename: str = "test_results.json"):
    """Save test results to JSON file"""
    output_path = Path(__file__).parent.parent / "docs" / filename
    output_path.parent.mkdir(exist_ok=True)
    
    output_data = {
        "timestamp": datetime.now().isoformat(),
        "total_queries": len(results),
        "results": results
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"💾 Results saved to: {output_path}")


def main():
    """Main test function"""
    print("\n" + "="*80)
    print("🧪 TESTING RAG PIPELINE")
    print("="*80 + "\n")
    
    # Initialize pipeline
    print("🔧 Initializing RAG Pipeline...")
    try:
        pipeline = RAGPipeline()
        print("✅ Pipeline ready\n")
    except Exception as e:
        print(f"❌ Failed to initialize pipeline: {str(e)}")
        sys.exit(1)
    
    # Run test queries
    all_results = []
    
    for i, query_info in enumerate(TEST_QUERIES, 1):
        try:
            result = pipeline.query(query_info['question'])
            
            # Add query info to result
            result['query_id'] = query_info['id']
            result['category'] = query_info['category']
            
            # Print result
            print_result(i, query_info, result)
            
            # Store result
            all_results.append(result)
            
        except Exception as e:
            print(f"\n❌ Error processing query {i}: {str(e)}\n")
            all_results.append({
                'query_id': query_info['id'],
                'question': query_info['question'],
                'category': query_info['category'],
                'error': str(e)
            })
    
    # Summary
    print_separator("=")
    print("📈 TEST SUMMARY")
    print_separator("=")
    print(f"\nTotal Queries: {len(TEST_QUERIES)}")
    print(f"Successful: {sum(1 for r in all_results if not r.get('error'))}")
    print(f"Failed: {sum(1 for r in all_results if r.get('error'))}")
    
    # Average confidence
    confidences = [r['confidence'] for r in all_results if 'confidence' in r]
    if confidences:
        avg_confidence = sum(confidences) / len(confidences)
        print(f"Average Confidence: {avg_confidence:.2%}")
    
    print()
    
    # Save results
    save_results_to_file(all_results)
    
    print("\n" + "="*80)
    print("✅ TESTING COMPLETE")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()