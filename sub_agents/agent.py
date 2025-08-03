import faiss
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from google.adk.agents import Agent
import os
import textwrap
from typing import Optional, List

# --- The Consolidated Intelligent RAG Function ---
def run_intelligent_rag_search(
    query: str, 
    index_path: str, 
    mapping_path: str, 
    content_type: Optional[str] = None, 
    domain: Optional[str] = None
) -> str:
    """
    A self-contained function that loads a knowledge base, performs an intelligent,
    filtered search, and returns the formatted results.

    Args:
        query (str): The user's complaint or question.
        index_path (str): The file path to the FAISS index.
        mapping_path (str): The file path to the JSON mapping file.
        content_type (str, optional): 'tickets' or 'manuals'.
        domain (str, optional): 'HR', 'IT', or 'Payroll'.
    
    Returns:
        A formatted string containing the most relevant search results.
    """
    # --- Part 1: Load the Knowledge Base Service ---
    print(f"\n🔎 [Tool Called] Loading RAG service for query: '{query}'")
    try:
        if not os.path.exists(index_path) or not os.path.exists(mapping_path):
            raise FileNotFoundError(f"Database files '{index_path}' or '{mapping_path}' not found.")
        
        model = SentenceTransformer('all-MiniLM-L6-v2')
        index = faiss.read_index(index_path)
        with open(mapping_path, 'r', encoding='utf-8') as f:
            mapping = json.load(f)
        print("✅ RAG Service is ready.")
    except Exception as e:
        return f"Error loading RAG service: {e}"

    # --- Part 2: Perform the Intelligent Search ---
    print(f"   -> Searching with filters: Type='{content_type}', Domain='{domain}'")
    query_vector = model.encode([query]).astype('float32')
    # Retrieve a larger number of candidates (k=20) for better filtering
    distances, ids = index.search(query_vector, 20)
    
    # --- Part 3: Filter the Results ---
    filtered_results = []
    for i, doc_id in enumerate(ids[0]):
        if doc_id == -1: continue
        result_chunk = mapping.get(str(doc_id))
        if not result_chunk: continue

        metadata = result_chunk.get('metadata', {})
        source_file = metadata.get('source', '').lower()
        
        # Filter by Content Type
        if content_type:
            is_ticket = any(ext in source_file for ext in ['.csv', '.xlsx'])
            is_manual = '.docx' in source_file
            if (content_type == 'tickets' and not is_ticket) or \
               (content_type == 'manuals' and not is_manual):
                continue

        # Filter by Domain
        if domain:
            chunk_domain = str(metadata.get('domain', metadata.get('section', ''))).lower()
            if domain.lower() not in chunk_domain:
                continue
        
        result_chunk['similarity_score'] = 1 - distances[0][i]
        filtered_results.append(result_chunk)

    # --- Part 4: Format and Return the Final Output ---
    if not filtered_results:
        return "No relevant information was found with the specified filters."

    top_results = filtered_results[:3]
    formatted_output = "Found the following relevant information:\n\n"
    for res in top_results:
        source = res.get('metadata', {}).get('source', 'N/A')
        text = res.get('text', 'N/A')
        formatted_output += f"--- \nSource: {source}\nContent: {text}\n---\n"
        
    return formatted_output

# --- Agent Setup and Execution ---
# Note: For the ADK, the tool function itself is what matters.
# The agent doesn't need to know it's a single, consolidated function.

# We need to define a simple wrapper so the agent can call it without
# having to provide the file paths every time.
def intelligent_search_tool_wrapper(query: str, content_type: Optional[str] = None, domain: Optional[str] = None) -> str:
    """
    Intelligently searches the NexaCorp knowledge base with optional filters.

    Args:
        query (str): The user's complaint or question.
        content_type (str, optional): The type of content to search for. Can be either 'tickets' to find past examples or 'manuals' to find official policy. Defaults to searching both.
        domain (str, optional): The specific department to search within. Can be 'HR', 'IT', or 'Payroll'. Defaults to searching all domains.
    """
    DATABASE_DIR = "dataset"
    INDEX_PATH = os.path.join(DATABASE_DIR, "nexa_corp.index")
    MAPPING_PATH = os.path.join(DATABASE_DIR, "nexa_corp_mapping.json")
    
    return run_intelligent_rag_search(
        query=query,
        index_path=INDEX_PATH,
        mapping_path=MAPPING_PATH,
        content_type=content_type,
        domain=domain
    )
root_agent = Agent(
    name="SupportAgent",
    model="gemini-2.0-flash",
    tools=[intelligent_search_tool_wrapper],
    instruction="""
    You are an intelligent NexaCorp support agent. Your goal is to resolve employee issues efficiently.
    1. First, analyze the user's query to determine the topic (e.g., 'VPN', 'salary', 'leave') and the likely domain ('IT', 'Payroll', 'HR').
    2. Decide if you need to find an official policy ('manuals') or a past example ('tickets').
    3. Use the `intelligent_search_tool_wrapper` with the appropriate `query`, `content_type`, and `domain` parameters to get the most accurate information.
    4. Synthesize the tool's output into a clear, helpful, and conversational answer.
    """
)
