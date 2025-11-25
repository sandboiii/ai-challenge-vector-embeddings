#!/usr/bin/env python3
"""
RAG System Backend - PDF Ingestion Script

This script processes PDF files from a directory, chunks them based on token count,
generates embeddings using a local Ollama model, and saves the results to FAISS.
"""

import json
import logging
import os
from pathlib import Path
from typing import List, Tuple, Dict, Any

import numpy as np
import tiktoken
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.faiss import dependable_faiss_import
from langchain_ollama import OllamaEmbeddings
from pypdf import PdfReader
from tqdm import tqdm

# Force CPU mode for Ollama to avoid GPU/ROCm issues
# Set environment variable before importing/using Ollama
os.environ.setdefault("OLLAMA_NUM_GPU", "0")

# Try different import paths for TokenTextSplitter
try:
    from langchain_text_splitters import TokenTextSplitter
except ImportError:
    try:
        from langchain_community.text_splitter import TokenTextSplitter
    except ImportError:
        from langchain.text_splitter import TokenTextSplitter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_pdfs(documents_dir: Path) -> List[Tuple[str, str]]:
    """
    Load all PDF files from the documents directory.
    
    Args:
        documents_dir: Path to the directory containing PDF files
        
    Returns:
        List of tuples containing (filename, text_content) for each PDF
    """
    # Create directory if it doesn't exist
    documents_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all PDF files
    pdf_files = list(documents_dir.glob("*.pdf"))
    
    if not pdf_files:
        logger.warning(f"No PDF files found in {documents_dir}")
        return []
    
    logger.info(f"Found {len(pdf_files)} PDF file(s) to process")
    
    texts = []
    for pdf_file in pdf_files:
        try:
            logger.info(f"Loading PDF: {pdf_file.name}")
            reader = PdfReader(str(pdf_file))
            
            # Extract text from all pages
            text_content = ""
            for page in reader.pages:
                text_content += page.extract_text() + "\n"
            
            if text_content.strip():
                texts.append((pdf_file.name, text_content))
                logger.info(f"Successfully loaded {pdf_file.name} ({len(text_content)} characters)")
            else:
                logger.warning(f"No text extracted from {pdf_file.name}")
                
        except Exception as e:
            logger.error(f"Error loading PDF {pdf_file.name}: {str(e)}")
            continue
    
    logger.info(f"Successfully loaded {len(texts)} PDF file(s)")
    return texts


def chunk_documents(
    texts: List[Tuple[str, str]], 
    chunk_size: int = 1000, 
    chunk_overlap: int = 100
) -> Tuple[List[str], List[Dict[str, Any]]]:
    """
    Split documents into chunks based on token count.
    
    Args:
        texts: List of (filename, text) tuples
        chunk_size: Maximum number of tokens per chunk
        chunk_overlap: Number of tokens to overlap between chunks
        
    Returns:
        Tuple of (chunks, metadata) where chunks is a list of text chunks
        and metadata is a list of dictionaries with LangChain metadata format
    """
    if not texts:
        logger.warning("No texts provided for chunking")
        return [], []
    
    # Initialize token text splitter
    text_splitter = TokenTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    
    # Initialize tiktoken with the same encoding as TokenTextSplitter (cl100k_base)
    try:
        encoding = tiktoken.get_encoding("cl100k_base")
    except Exception:
        # Fallback to default encoding if cl100k_base is not available
        encoding = tiktoken.get_encoding("gpt2")
    
    all_chunks = []
    all_metadata = []
    
    for filename, text in texts:
        try:
            # Split the text into chunks
            chunks = text_splitter.split_text(text)
            
            # Create metadata for each chunk with tokenized text
            for i, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                
                # Tokenize the chunk text and get actual token strings
                try:
                    token_ids = encoding.encode(chunk)
                    # Decode each token ID back to its string representation
                    # We decode the full sequence and then extract individual tokens
                    # by decoding progressively
                    token_list = []
                    for idx, token_id in enumerate(token_ids):
                        try:
                            # Decode from start up to current token to get the token string
                            if idx == 0:
                                # First token: decode just this token
                                decoded = encoding.decode([token_id])
                                # Get the token string by comparing with previous (empty) decode
                                token_string = decoded
                            else:
                                # Decode up to current position and subtract previous decode
                                decoded_up_to = encoding.decode(token_ids[:idx+1])
                                decoded_prev = encoding.decode(token_ids[:idx])
                                token_string = decoded_up_to[len(decoded_prev):]
                            token_list.append(token_string)
                        except Exception:
                            # Fallback: try direct decode of single token
                            try:
                                token_string = encoding.decode([token_id])
                                token_list.append(token_string)
                            except Exception:
                                # If all else fails, use token ID as fallback
                                token_list.append(f"<token_{token_id}>")
                except Exception as e:
                    logger.warning(f"Error tokenizing chunk {i} from {filename}: {str(e)}")
                    token_list = []
                
                all_metadata.append({
                    "source": filename,
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "tokens": token_list
                })
            
            logger.info(f"Split {filename} into {len(chunks)} chunk(s)")
            
        except Exception as e:
            logger.error(f"Error chunking {filename}: {str(e)}")
            continue
    
    logger.info(f"Total chunks created: {len(all_chunks)}")
    return all_chunks, all_metadata


def create_embeddings_batch(
    chunks: List[str], 
    metadata: List[Dict[str, Any]], 
    output_dir: Path,
    batch_size: int = 32
) -> Tuple[FAISS, List[Dict[str, Any]]]:
    """
    Create embeddings for chunks in batches and build FAISS vector store.
    Saves metadata incrementally after each batch.
    
    Args:
        chunks: List of text chunks to embed
        metadata: List of metadata dictionaries for each chunk
        output_dir: Directory to save metadata incrementally
        batch_size: Number of chunks to process in each batch
        
    Returns:
        Tuple of (FAISS vector store, updated metadata with vectors)
    """
    if not chunks:
        raise ValueError("No chunks provided for embedding")
    
    if len(chunks) != len(metadata):
        raise ValueError("Number of chunks and metadata must match")
    
    # Initialize Ollama embeddings
    # Force CPU mode if GPU/ROCm issues occur
    if "OLLAMA_NUM_GPU" not in os.environ:
        os.environ["OLLAMA_NUM_GPU"] = "0"
        logger.info("Setting OLLAMA_NUM_GPU=0 to force CPU mode")
    
    logger.info("Initializing Ollama embeddings (model: nomic-embed-text)")
    logger.info("Note: If you encounter GPU/ROCm errors, ensure Ollama is running with CPU mode")
    logger.info("Note: Ollama may show warnings about invalid options (mirostat, tfs_z, etc.) - these are harmless for embedding models")
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    
    # Initialize FAISS vector store (will be built incrementally)
    vectorstore = None
    
    # Process chunks in batches with progress bar
    total_chunks = len(chunks)
    num_batches = (total_chunks + batch_size - 1) // batch_size
    
    logger.info(f"Processing {total_chunks} chunks in {num_batches} batch(es) of {batch_size}")
    
    # Initialize lists to collect all processed data
    all_valid_chunks = []
    all_valid_metadata = []
    all_valid_embeddings = []
    
    # Process chunks in batches with progress bar and incremental saving
    logger.info("Generating embeddings and building vector store in batches...")
    with tqdm(total=total_chunks, desc="Processing embeddings", unit="chunks") as pbar:
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, total_chunks)
            
            batch_chunks = chunks[start_idx:end_idx]
            batch_metadata = metadata[start_idx:end_idx]
            
            try:
                # Generate embeddings for this batch
                batch_embeddings = embeddings.embed_documents(batch_chunks)
                
                # Add vectors to metadata and collect valid entries
                batch_valid_chunks = []
                batch_valid_metadata = []
                batch_valid_embeddings = []
                
                for i, (chunk, meta, emb) in enumerate(zip(batch_chunks, batch_metadata, batch_embeddings)):
                    if emb is not None:
                        # Add vector to metadata (convert numpy array to list for JSON)
                        meta_with_vector = meta.copy()
                        meta_with_vector["vector"] = emb.tolist() if isinstance(emb, np.ndarray) else list(emb)
                        
                        batch_valid_chunks.append(chunk)
                        batch_valid_metadata.append(meta_with_vector)
                        batch_valid_embeddings.append(emb)
                
                if not batch_valid_chunks:
                    logger.warning(f"No valid embeddings in batch {batch_idx + 1}/{num_batches}")
                    pbar.update(len(batch_chunks))
                    continue
                
                # Save metadata incrementally after each batch
                save_metadata_incremental(output_dir, batch_valid_metadata)
                
                # Collect data for FAISS (will build at the end to avoid regenerating embeddings)
                # Metadata is saved incrementally above, FAISS will be built once at the end
                
                # Collect for final return
                all_valid_chunks.extend(batch_valid_chunks)
                all_valid_metadata.extend(batch_valid_metadata)
                all_valid_embeddings.extend(batch_valid_embeddings)
                
                pbar.update(len(batch_chunks))
                
            except Exception as e:
                logger.error(f"Error processing batch {batch_idx + 1}/{num_batches}: {str(e)}")
                pbar.update(len(batch_chunks))
                continue
    
    if not all_valid_chunks:
        raise ValueError("No valid embeddings were generated")
    
    # Build FAISS vector store from all collected embeddings
    logger.info(f"Creating FAISS vector store with {len(all_valid_chunks)} chunks...")
    try:
        from langchain_core.documents import Document
        from langchain_community.docstore.in_memory import InMemoryDocstore
        
        faiss = dependable_faiss_import()
        dimension = len(all_valid_embeddings[0])
        
        # Create FAISS index
        index = faiss.IndexFlatL2(dimension)
        
        # Add all vectors to the index
        vectors = np.array(all_valid_embeddings, dtype=np.float32)
        index.add(vectors)
        
        # Create documents
        documents = [
            Document(page_content=chunk, metadata=meta)
            for chunk, meta in zip(all_valid_chunks, all_valid_metadata)
        ]
        
        # Create docstore and populate it
        docstore_dict = {str(i): doc for i, doc in enumerate(documents)}
        docstore = InMemoryDocstore(docstore_dict)
        index_to_docstore_id = {i: str(i) for i in range(len(documents))}
        
        # Create FAISS vector store with properly initialized docstore
        vectorstore = FAISS(embeddings, index, docstore, index_to_docstore_id)
        
    except Exception as e:
        logger.warning(f"Could not create FAISS from pre-computed embeddings: {str(e)}")
        logger.info("Falling back to standard method (embeddings will be regenerated)...")
        # Fallback: create using from_texts (will regenerate embeddings)
        vectorstore = FAISS.from_texts(
            texts=all_valid_chunks,
            embedding=embeddings,
            metadatas=all_valid_metadata
        )
    
    logger.info("Embedding generation completed")
    return vectorstore, all_valid_metadata


def save_metadata_incremental(
    output_dir: Path,
    new_metadata: List[Dict[str, Any]]
) -> None:
    """
    Save metadata incrementally by appending to existing metadata.json.
    
    Args:
        output_dir: Directory containing metadata.json
        new_metadata: New metadata entries to append
    """
    # Create directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    metadata_path = output_dir / "metadata.json"
    
    # Load existing metadata if it exists
    existing_metadata = []
    if metadata_path.exists():
        try:
            with open(metadata_path, 'r', encoding='utf-8') as f:
                existing_metadata = json.load(f)
        except Exception as e:
            logger.warning(f"Could not load existing metadata: {str(e)}. Starting fresh.")
            existing_metadata = []
    
    # Append new metadata
    existing_metadata.extend(new_metadata)
    
    # Save updated metadata
    try:
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(existing_metadata, f, indent=2, ensure_ascii=False)
        logger.info(f"Incremental metadata save: {len(new_metadata)} new entries, {len(existing_metadata)} total entries")
    except Exception as e:
        logger.error(f"Error saving incremental metadata: {str(e)}")
        raise


def save_vectorstore(
    vectorstore: FAISS, 
    output_dir: Path, 
    metadata: List[Dict[str, Any]]
) -> None:
    """
    Save FAISS vector store to disk.
    Note: Metadata is already saved incrementally during batch processing.
    
    Args:
        vectorstore: FAISS vector store to save
        output_dir: Directory to save the vector store
        metadata: List of metadata dictionaries (for logging/verification)
    """
    # Create directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Save FAISS index
        logger.info(f"Saving FAISS vector store to {output_dir}")
        vectorstore.save_local(str(output_dir))
        logger.info("FAISS vector store saved successfully")
        
        # Verify metadata file exists (it should have been saved incrementally)
        metadata_path = output_dir / "metadata.json"
        if metadata_path.exists():
            logger.info(f"Metadata already saved incrementally ({len(metadata)} entries)")
        else:
            logger.warning("Metadata file not found, saving now...")
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            logger.info(f"Metadata saved successfully ({len(metadata)} entries)")
        
    except Exception as e:
        logger.error(f"Error saving vector store: {str(e)}")
        raise


def main() -> None:
    """Main function to orchestrate the PDF ingestion pipeline."""
    logger.info("Starting PDF ingestion pipeline")
    
    # Define paths
    documents_dir = Path("./documents")
    vectorstore_dir = Path("./vectorstore")
    
    # Step 1: Load PDFs
    logger.info("=" * 50)
    logger.info("Step 1: Loading PDF files")
    logger.info("=" * 50)
    texts = load_pdfs(documents_dir)
    
    if not texts:
        logger.error("No PDF files were successfully loaded. Exiting.")
        return
    
    # Step 2: Chunk documents
    logger.info("=" * 50)
    logger.info("Step 2: Chunking documents")
    logger.info("=" * 50)
    chunks, metadata = chunk_documents(texts, chunk_size=1000, chunk_overlap=100)
    
    if not chunks:
        logger.error("No chunks were created. Exiting.")
        return
    
    # Step 3: Create embeddings and build vector store
    logger.info("=" * 50)
    logger.info("Step 3: Generating embeddings")
    logger.info("=" * 50)
    try:
        vectorstore, updated_metadata = create_embeddings_batch(
            chunks, metadata, vectorstore_dir, batch_size=32
        )
    except Exception as e:
        logger.error(f"Failed to create embeddings: {str(e)}")
        return
    
    # Step 4: Save vector store (metadata already saved incrementally)
    logger.info("=" * 50)
    logger.info("Step 4: Saving vector store")
    logger.info("=" * 50)
    try:
        save_vectorstore(vectorstore, vectorstore_dir, updated_metadata)
    except Exception as e:
        logger.error(f"Failed to save vector store: {str(e)}")
        return
    
    logger.info("=" * 50)
    logger.info("PDF ingestion pipeline completed successfully!")
    logger.info("=" * 50)
    logger.info(f"Vector store saved to: {vectorstore_dir.absolute()}")
    logger.info(f"Metadata saved to: {(vectorstore_dir / 'metadata.json').absolute()}")


if __name__ == "__main__":
    main()

