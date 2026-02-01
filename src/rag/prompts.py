"""
Prompt Templates for RAG System
"""

SYSTEM_PROMPT = """You are an expert AI assistant specializing in Agentic AI. Your knowledge comes exclusively from the "Agentic AI: An Executive's Guide to In-depth Understanding of Agentic AI" eBook.

CRITICAL INSTRUCTIONS:
1. Answer questions ONLY using information from the provided context
2. If the context doesn't contain relevant information, clearly state: "I don't have information about that in the provided eBook."
3. DO NOT make up or infer information beyond what's in the context
4. Always cite the page number when providing information
5. Be precise, accurate, and professional
6. If asked about topics not covered in the eBook, politely decline

Your responses should be:
- Direct and concise
- Well-structured with clear explanations
- Grounded in the provided context
- Properly attributed to source pages"""


RAG_PROMPT_TEMPLATE = """Based on the following context from the Agentic AI eBook, please answer the user's question.

CONTEXT:
{context}

USER QUESTION: {question}

INSTRUCTIONS:
- Answer using ONLY the information provided in the context above
- Cite the page number(s) where you found the information
- If the context doesn't contain the answer, say so clearly
- Be specific and accurate
- Structure your answer logically

ANSWER:"""


QUERY_ENHANCEMENT_PROMPT = """Given a user question about Agentic AI, enhance it to make it more specific and searchable while preserving the original intent.

Original Question: {question}

Enhanced Question (more specific and detailed):"""


ANSWER_VERIFICATION_PROMPT = """You are a fact-checker. Verify if the answer is fully grounded in the provided context.

CONTEXT:
{context}

QUESTION: {question}

PROPOSED ANSWER:
{answer}

Verification Instructions:
1. Check if every claim in the answer is supported by the context
2. Identify any hallucinations or unsupported statements
3. Verify page citations are correct

Is the answer fully grounded in the context? (YES/NO)
If NO, list the issues:

VERIFICATION RESULT:"""


CONVERSATIONAL_PROMPT = """You are a helpful AI assistant with expertise in Agentic AI based on the provided eBook.

Previous conversation:
{chat_history}

Current context from eBook:
{context}

User: {question}

Assistant (answer based on context, acknowledge previous conversation if relevant):"""


NO_CONTEXT_RESPONSE = """I apologize, but I don't have information about that specific topic in the Agentic AI eBook that I have access to.

The eBook covers topics such as:
- Introduction to Agentic AI
- Anatomy of Agentic AI Systems
- Multi-Agent Systems
- Orchestrating Agentic AI
- Organizational Readiness
- Practical Applications

Could you rephrase your question or ask about one of these topics?"""


GREETING_RESPONSE = """Hello! I'm an AI assistant specialized in Agentic AI, based on the comprehensive eBook "Agentic AI: An Executive's Guide."

I can help you understand:
- What Agentic AI is and how it differs from traditional AI
- Components and architecture of Agentic AI systems
- Multi-agent systems and orchestration
- Real-world applications across industries
- How to assess your organization's readiness for Agentic AI

What would you like to know about Agentic AI?"""


def get_rag_prompt(context: str, question: str) -> str:
    """
    Generate complete RAG prompt
    
    Args:
        context: Retrieved context from vector store
        question: User question
        
    Returns:
        Formatted prompt string
    """
    return RAG_PROMPT_TEMPLATE.format(context=context, question=question)


def get_system_message() -> dict:
    """
    Get system message for LLM
    
    Returns:
        System message dict
    """
    return {"role": "system", "content": SYSTEM_PROMPT}


def format_chat_history(messages: list) -> str:
    """
    Format chat history for conversational context
    
    Args:
        messages: List of message dicts with 'role' and 'content'
        
    Returns:
        Formatted chat history string
    """
    if not messages:
        return "No previous conversation."
    
    history_parts = []
    for msg in messages[-6:]:  # Last 3 exchanges
        role = "User" if msg["role"] == "user" else "Assistant"
        history_parts.append(f"{role}: {msg['content']}")
    
    return "\n".join(history_parts)