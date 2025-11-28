import os
import glob
from typing import List, Dict
from rank_bm25 import BM25Okapi

# Simple in-memory storage for our chunks
CHUNKS = []
BM25_MODEL = None

def load_and_chunk_docs(docs_dir: str = None):
    """
    Loads all .md files from docs_dir, splits them by double newlines (paragraphs),
    and initializes the BM25 model.
    """
    global CHUNKS, BM25_MODEL
    
    CHUNKS = []
    tokenized_corpus = []
    
    # Default to agent/docs directory
    if docs_dir is None:
        docs_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "docs")
    
    # Get all markdown files
    md_files = glob.glob(os.path.join(docs_dir, "*.md"))
    
    for filepath in md_files:
        filename = os.path.basename(filepath)
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
            
        # Split by double newline to get paragraphs
        # Assignment Tip: Keep chunks small [cite: 149]
        raw_chunks = content.split("\n\n")
        
        for i, text in enumerate(raw_chunks):
            text = text.strip()
            if not text: 
                continue
                
            chunk_id = f"{filename}::chunk{i}"
            
            # Store metadata
            CHUNKS.append({
                "id": chunk_id,
                "content": text,
                "source": filename
            })
            
            # Simple tokenization for BM25 (lowercase, split by space)
            tokenized_corpus.append(text.lower().split())

    # Initialize BM25
    if tokenized_corpus:
        BM25_MODEL = BM25Okapi(tokenized_corpus)
        print(f"Indexed {len(CHUNKS)} chunks from {len(md_files)} files.")
    else:
        print("Warning: No documents found to index.")

def retrieve_docs(query: str, top_k: int = 3) -> List[Dict]:
    """
    Returns the top_k most relevant chunks for a given query WITH SCORES.
    """
    global CHUNKS, BM25_MODEL
    
    if BM25_MODEL is None:
        load_and_chunk_docs()
        
    if not CHUNKS:
        return []

    tokenized_query = query.lower().split()
    
    # Get scores
    scores = BM25_MODEL.get_scores(tokenized_query)
    
    # Pair scores with chunks and sort
    scored_chunks = []
    for i, score in enumerate(scores):
        if score > 0: # Filter out irrelevant noise
            chunk_with_score = CHUNKS[i].copy()
            chunk_with_score['score'] = float(score)
            scored_chunks.append((score, chunk_with_score))
            
    # Sort descending by score
    scored_chunks.sort(key=lambda x: x[0], reverse=True)
    
    # Return top K chunks (now including score)
    return [item[1] for item in scored_chunks[:top_k]]