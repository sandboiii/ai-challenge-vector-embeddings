# RAG System Backend - PDF Ingestion Script

A Python script for building a local RAG (Retrieval-Augmented Generation) system backend that processes PDF files, chunks them based on token count, generates embeddings using Ollama, and stores them in a FAISS vector database.

## Features

- **PDF Processing**: Automatically loads all PDF files from a `./documents` directory
- **Token-based Chunking**: Splits documents into chunks of 1000 tokens with 100 token overlap
- **Local Embeddings**: Uses Ollama's `nomic-embed-text` model for generating embeddings
- **Batch Processing**: Processes embeddings in batches with a visual progress bar
- **FAISS Vector Store**: Stores embeddings in a local FAISS index for fast similarity search
- **Metadata Tracking**: Saves metadata mapping chunks to source files

## Prerequisites

- Python 3.8 or higher
- [Ollama](https://ollama.ai/) installed and running
- The `nomic-embed-text` model available in Ollama

### Installing Ollama and the Model

1. Install Ollama from [https://ollama.ai/](https://ollama.ai/)
2. Pull the embedding model:
   ```bash
   ollama pull nomic-embed-text
   ```
3. Verify the model is available:
   ```bash
   ollama list
   ```

## Setup Instructions

### 1. Create a Virtual Environment

```bash
python -m venv venv
```

### 2. Activate the Virtual Environment

**On Linux/Mac:**
```bash
source venv/bin/activate
```

**On Windows:**
```bash
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Prepare PDF Files

Place your PDF files in a `./documents` directory (the script will create this directory if it doesn't exist):

```bash
mkdir -p documents
# Copy your PDF files to the documents directory
cp /path/to/your/*.pdf documents/
```

### 5. Run the Script

```bash
python ingest.py
```

## Usage

The script will:

1. **Load PDFs**: Scan the `./documents` directory and load all PDF files
2. **Chunk Documents**: Split each PDF into chunks of 1000 tokens with 100 token overlap
3. **Generate Embeddings**: Process chunks in batches of 32, showing a progress bar
4. **Save Vector Store**: Save the FAISS index and metadata to `./vectorstore`

### Example Output

```
2024-01-15 10:30:00 - INFO - Starting PDF ingestion pipeline
==================================================
Step 1: Loading PDF files
==================================================
2024-01-15 10:30:00 - INFO - Found 3 PDF file(s) to process
2024-01-15 10:30:00 - INFO - Loading PDF: document1.pdf
2024-01-15 10:30:01 - INFO - Successfully loaded document1.pdf (45230 characters)
...
2024-01-15 10:30:05 - INFO - Successfully loaded 3 PDF file(s)
==================================================
Step 2: Chunking documents
==================================================
2024-01-15 10:30:05 - INFO - Split document1.pdf into 45 chunk(s)
...
2024-01-15 10:30:05 - INFO - Total chunks created: 127
==================================================
Step 3: Generating embeddings
==================================================
2024-01-15 10:30:05 - INFO - Initializing Ollama embeddings (model: nomic-embed-text)
2024-01-15 10:30:05 - INFO - Processing 127 chunks in 4 batch(es) of 32
Processing embeddings: 100%|████████████| 127/127 chunks [02:15<00:00,  0.94chunks/s]
2024-01-15 10:30:07 - INFO - Embedding generation completed
==================================================
Step 4: Saving vector store
==================================================
2024-01-15 10:30:07 - INFO - Saving FAISS vector store to ./vectorstore
2024-01-15 10:30:07 - INFO - FAISS vector store saved successfully
2024-01-15 10:30:07 - INFO - Saving metadata to ./vectorstore/metadata.json
2024-01-15 10:30:07 - INFO - Metadata saved successfully (127 entries)
==================================================
PDF ingestion pipeline completed successfully!
==================================================
```

## Verification Steps

After running the script, verify that everything was created correctly:

### 1. Check Vector Store Directory

```bash
ls -la vectorstore/
```

You should see FAISS index files (typically `index.faiss` and `index.pkl`).

### 2. Check Metadata File

```bash
cat vectorstore/metadata.json | head -20
```

The metadata file should contain a JSON array with entries like:

```json
[
  {
    "source": "document1.pdf",
    "chunk_index": 0,
    "total_chunks": 45
  },
  {
    "source": "document1.pdf",
    "chunk_index": 1,
    "total_chunks": 45
  },
  ...
]
```

### 3. Verify File Counts

Check that the number of entries in `metadata.json` matches the total chunks logged:

```bash
python -c "import json; data = json.load(open('vectorstore/metadata.json')); print(f'Total chunks: {len(data)}')"
```

### 4. Test Loading the Vector Store (Optional)

You can verify the vector store loads correctly:

```python
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings

embeddings = OllamaEmbeddings(model="nomic-embed-text")
vectorstore = FAISS.load_local("./vectorstore", embeddings, allow_dangerous_deserialization=True)
print(f"Vector store loaded with {vectorstore.index.ntotal} vectors")
```

## Configuration

You can modify the following parameters in `ingest.py`:

- **Chunk Size**: Default is 1000 tokens (modify in `chunk_documents()` call)
- **Chunk Overlap**: Default is 100 tokens (modify in `chunk_documents()` call)
- **Batch Size**: Default is 32 chunks per batch (modify in `create_embeddings_batch()` call)
- **Embedding Model**: Default is `nomic-embed-text` (modify in `create_embeddings_batch()`)

## Troubleshooting

### Ollama Connection Error

If you see errors about Ollama not being available:

1. Ensure Ollama is running: `ollama serve` (or start it as a service)
2. Verify the model is available: `ollama list`
3. Test the model: `ollama run nomic-embed-text "test"`

### No PDFs Found

- Ensure PDF files are in the `./documents` directory
- Check file extensions are `.pdf` (lowercase)
- Verify PDF files are not corrupted

### GPU/ROCm Errors

If you encounter ROCm or GPU-related errors like:
```
ROCm error: invalid device function
llama runner process has terminated
```

This indicates Ollama is trying to use GPU but encountering compatibility issues. Force CPU mode:

**Option 1: Set environment variable before running the script**
```bash
export OLLAMA_NUM_GPU=0
python ingest.py
```

**Option 2: Restart Ollama with CPU mode**
```bash
# Stop Ollama
pkill ollama

# Start Ollama with CPU mode
OLLAMA_NUM_GPU=0 ollama serve

# In another terminal, run the script
python ingest.py
```

**Option 3: Set permanently in your shell profile**
Add to `~/.bashrc` or `~/.zshrc`:
```bash
export OLLAMA_NUM_GPU=0
```

### Ollama Server Warnings

If you see warnings in the Ollama server logs like:
```
level=WARN source=types.go:800 msg="invalid option provided" option=mirostat
init: embeddings required but some input tokens were not marked as outputs -> overriding
```

**These warnings are harmless and expected** when using embedding models. They occur because:

1. **Invalid option warnings** (mirostat, mirostat_tau, tfs_z, etc.): These are text generation parameters that don't apply to embedding models. LangChain may pass default parameters, but Ollama correctly ignores them for embeddings.

2. **Token marking warnings**: This is normal behavior for embedding models - Ollama automatically handles token marking for embeddings.

**No action needed** - your embeddings are being generated correctly despite these warnings. They can be safely ignored.

### Memory Issues

If you encounter memory issues with large PDFs:

- Reduce the `batch_size` parameter (e.g., from 32 to 16)
- Process fewer PDFs at a time
- Increase system RAM or use a machine with more memory

## Project Structure

```
.
├── ingest.py              # Main ingestion script
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── documents/            # Input PDF files (created automatically)
└── vectorstore/          # Output FAISS index and metadata (created automatically)
    ├── index.faiss       # FAISS vector index
    ├── index.pkl         # FAISS metadata pickle
    └── metadata.json     # Chunk-to-source mapping
```

## License

This project is provided as-is for educational and development purposes.

