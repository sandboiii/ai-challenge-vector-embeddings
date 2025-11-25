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

# Force CPU mode for Ollama to avoid GPU/ROCm issues
os.environ.setdefault("OLLAMA_NUM_GPU", "0")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global variables to store the vector store and embeddings
vectorstore: Optional[FAISS] = None
embeddings: Optional[OllamaEmbeddings] = None


# Pydantic models for request/response
class AugmentRequest(BaseModel):
    """Request model for the augment endpoint."""
    query: str = Field(..., description="User's question or query string")
    k: int = Field(default=3, ge=1, le=20, description="Number of documents to retrieve (1-20)")


class AugmentResponse(BaseModel):
    """Response model for the augment endpoint."""
    original_query: str = Field(..., description="The original user query")
    context_chunks: List[str] = Field(..., description="Retrieved context chunks from the vector store")
    suggested_prompt: str = Field(..., description="Augmented prompt combining context and query")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager to load vector store and embeddings on startup.
    """
    global vectorstore, embeddings
    
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
        "embeddings_loaded": embeddings is not None
    }


@app.post("/api/v1/augment", response_model=AugmentResponse)
async def augment_query(request: AugmentRequest):
    """
    Augment a user query with relevant context from the vector store.
    
    This endpoint:
    1. Converts the user query into an embedding
    2. Searches the FAISS index for the top k most similar chunks
    3. Constructs a suggested prompt combining the retrieved context and the query
    
    Args:
        request: AugmentRequest containing the query and optional k parameter
        
    Returns:
        AugmentResponse with original query, context chunks, and suggested prompt
        
    Raises:
        HTTPException: 404 if vector store not loaded, 500 for internal errors
    """
    if vectorstore is None or embeddings is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Vector store or embeddings not loaded. Please check server logs."
        )
    
    try:
        # Perform similarity search
        logger.info(f"Searching for top {request.k} documents for query: {request.query[:50]}...")
        
        # Use similarity_search_with_score to get results with relevance scores
        results = vectorstore.similarity_search_with_score(
            request.query,
            k=request.k
        )
        
        # Extract text chunks from results
        context_chunks = [doc.page_content for doc, score in results]
        
        # Construct suggested prompt
        context_text = "\n\n".join([
            f"[Context {i+1}]: {chunk}"
            for i, chunk in enumerate(context_chunks)
        ])
        
        suggested_prompt = f"""Context:
{context_text}

Question: {request.query}

Please answer the question based on the provided context."""
        
        logger.info(f"Retrieved {len(context_chunks)} context chunks")
        
        return AugmentResponse(
            original_query=request.query,
            context_chunks=context_chunks,
            suggested_prompt=suggested_prompt
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

