#!/usr/bin/env python3
"""
RAG System Backend - FastAPI Server

A production-ready FastAPI server that provides RAG (Retrieval-Augmented Generation)
functionality for Android applications. Loads FAISS vector store on startup and
provides an API endpoint to augment user queries with relevant context.
"""

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from pydantic import BaseModel, Field
from sentence_transformers import CrossEncoder

# Force CPU mode for Ollama to avoid GPU/ROCm issues
os.environ.setdefault("OLLAMA_NUM_GPU", "0")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global variables to store the vector store, embeddings, and reranker
vectorstore: Optional[FAISS] = None
embeddings: Optional[OllamaEmbeddings] = None
reranker: Optional[CrossEncoder] = None


# Pydantic models for request/response
class AugmentRequest(BaseModel):
    """Request model for the augment endpoint."""
    query: str = Field(..., description="User's question or query string")
    k: int = Field(default=3, ge=1, le=20, description="Number of documents to retrieve (1-20)")


class AugmentResponse(BaseModel):
    """Response model for the augment endpoint."""
    original_query: str = Field(..., description="The original user query")
    context_chunks: List[str] = Field(..., description="Retrieved context chunks with citations from the vector store")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager to load vector store, embeddings, and reranker on startup.
    """
    global vectorstore, embeddings, reranker
    
    vectorstore_path = Path("./vectorstore")
    
    logger.info("=" * 50)
    logger.info("Starting RAG Backend Server")
    logger.info("=" * 50)
    
    # Check if vectorstore exists
    if not vectorstore_path.exists():
        logger.error(f"Vector store not found at {vectorstore_path.absolute()}")
        logger.error("Please run ingest.py first to create the vector store")
        raise FileNotFoundError(
            f"Vector store not found at {vectorstore_path}. "
            "Run ingest.py first to create the vector store."
        )
    
    # Initialize Ollama embeddings
    logger.info("Initializing Ollama embeddings (model: nomic-embed-text)...")
    try:
        embeddings = OllamaEmbeddings(model="nomic-embed-text")
        logger.info("Ollama embeddings initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize Ollama embeddings: {str(e)}")
        raise
    
    # Load FAISS vector store
    logger.info(f"Loading FAISS vector store from {vectorstore_path.absolute()}...")
    try:
        vectorstore = FAISS.load_local(
            str(vectorstore_path),
            embeddings,
            allow_dangerous_deserialization=True
        )
        logger.info(f"Vector store loaded successfully ({vectorstore.index.ntotal} vectors)")
    except Exception as e:
        logger.error(f"Failed to load vector store: {str(e)}")
        raise
    
    # Initialize CrossEncoder reranker
    logger.info("Initializing CrossEncoder reranker (model: cross-encoder/ms-marco-MiniLM-L-6-v2)...")
    try:
        reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        logger.info("CrossEncoder reranker initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize CrossEncoder reranker: {str(e)}")
        raise
    
    logger.info("=" * 50)
    logger.info("Server ready to accept requests")
    logger.info("=" * 50)
    
    yield
    
    # Cleanup (if needed)
    logger.info("Shutting down server...")


# Create FastAPI app with lifespan
app = FastAPI(
    title="RAG Backend API",
    description="Retrieval-Augmented Generation API for Android applications",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware to allow Android app to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your Android app's origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Root endpoint to check if server is running."""
    return {
        "status": "running",
        "message": "RAG Backend API is operational",
        "vectorstore_loaded": vectorstore is not None
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "vectorstore_loaded": vectorstore is not None,
        "embeddings_loaded": embeddings is not None,
        "reranker_loaded": reranker is not None
    }


@app.post("/api/v1/augment", response_model=AugmentResponse)
async def augment_query(request: AugmentRequest):
    """
    Augment a user query with relevant context from the vector store.
    
    This endpoint performs a two-stage retrieval process:
    1. Initial Retrieval: Uses FAISS to fetch top 15 candidates with scores
    2. Reranking: Uses CrossEncoder to rerank and select top k chunks
    3. Returns the final top k chunks with detailed logging at each stage
    
    Args:
        request: AugmentRequest containing the query and optional k parameter
        
    Returns:
        AugmentResponse with original query, context chunks, and suggested prompt
        
    Raises:
        HTTPException: 503 if vector store/embeddings/reranker not loaded, 500 for internal errors
    """
    if vectorstore is None or embeddings is None or reranker is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Vector store, embeddings, or reranker not loaded. Please check server logs."
        )
    
    try:
        query = request.query
        k = request.k
        
        # Step 1: Initial Retrieval (with scores)
        logger.info(f"Searching for top 15 documents for query: {query[:50]}...")
        
        initial_results = vectorstore.similarity_search_with_score(
            query,
            k=15
        )
        
        # Log Stage 1: Before Reranking
        logger.info("--- Stage 1: Initial Retrieval (Top 15 from FAISS) ---")
        for i, (doc, faiss_score) in enumerate(initial_results, start=1):
            metadata_source = doc.metadata.get('source', 'unknown')
            text_preview = doc.page_content[:70] + "..." if len(doc.page_content) > 70 else doc.page_content
            logger.info(f"[Rank {i}] Score: {faiss_score:.4f} | Source: {metadata_source} | Text: \"{text_preview}\"")
        
        # Step 2: Reranking
        # Extract text from documents
        chunk_texts = [doc.page_content for doc, _ in initial_results]
        
        # Prepare pairs for CrossEncoder: [query, chunk_text]
        pairs = [[query, chunk_text] for chunk_text in chunk_texts]
        
        # Get reranker scores
        reranker_scores = reranker.predict(pairs)
        
        # Step 3: Sorting & Slicing
        # Combine documents with their reranker scores
        ranked_results = list(zip(initial_results, reranker_scores))
        
        # Sort by reranker score (descending - higher is better)
        ranked_results.sort(key=lambda x: x[1], reverse=True)
        
        # Take top k chunks
        final_results = ranked_results[:k]
        
        # Log Stage 2: After Reranking
        logger.info(f"--- Stage 2: Reranked Results (Top {k} from Cross-Encoder) ---")
        for i, ((doc, faiss_score), reranker_score) in enumerate(final_results, start=1):
            metadata_source = doc.metadata.get('source', 'unknown')
            text_preview = doc.page_content[:70] + "..." if len(doc.page_content) > 70 else doc.page_content
            logger.info(f"[Rank {i}] Score: {reranker_score:.4f} | Source: {metadata_source} | Text: \"{text_preview}\"")
        
        # Step 4: Format final context chunks with citations
        context_chunks = []
        
        for idx, ((doc, _), _) in enumerate(final_results, start=1):
            chunk_text = doc.page_content
            
            # Extract and clean metadata
            source_path = doc.metadata.get('source', 'unknown')
            page_num = doc.metadata.get('page', 'N/A')
            
            # Clean filename: extract only the filename from full path
            if source_path and source_path != 'unknown':
                filename = os.path.basename(source_path)
            else:
                filename = 'unknown'
            
            # Format chunk with citation metadata
            formatted_chunk = f"[Source {idx}] (File: {filename}, Page: {page_num}):\n{chunk_text}"
            context_chunks.append(formatted_chunk)
        
        logger.info(f"Retrieved {len(context_chunks)} context chunks after reranking")
        
        return AugmentResponse(
            original_query=query,
            context_chunks=context_chunks
        )
        
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )

