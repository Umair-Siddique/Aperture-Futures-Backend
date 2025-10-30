# Upload Resolution API - Complete Pipeline Overview

## 🔄 Processing Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                        PDF Upload Request                        │
│                  POST /bluelines/resolution/upload               │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                   Step 1: Text Extraction                        │
│  • Uses PyMuPDF (fitz) to extract text from PDF                 │
│  • Removes common footer text                                   │
│  • Returns raw text content                                     │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                   Step 2: Text Chunking                          │
│  • RecursiveCharacterTextSplitter                               │
│  • Chunk size: 2500 characters                                  │
│  • Chunk overlap: 250 characters                                │
│  • Separators: \n\n, \n, ., !                                   │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│      Step 3: Embed & Store Chunks (Pinecone - Temporary)       │
│  • OpenAI text-embedding-3-large (3072 dimensions)              │
│  • Batch processing (10 chunks per API call)                    │
│  • Store in Pinecone temporary namespace (temp_upload_{uuid})   │
│  • Each chunk stored with text in metadata                      │
│  • Purpose: Enable semantic search for metadata extraction      │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│         Step 4: Retrieve Relevant Chunks from Pinecone          │
│  • Query: "Security Council resolution"                         │
│  • Retrieve top 15 most relevant chunks from temp namespace     │
│  • Extract chunk texts from metadata                            │
│  • Combine chunks into context for LLM                          │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                Step 5: Extract Metadata                          │
│  • LLM: meta-llama/llama-4-scout-17b-16e-instruct (Groq)       │
│  • Input: Retrieved chunks as context                           │
│  • Output: Structured JSON metadata                             │
│    - resolution_no                                              │
│    - year                                                       │
│    - theme                                                      │
│    - chapter (VI or VII)                                        │
│    - charter_articles                                           │
│    - entities                                                   │
│    - reporting_cycle                                            │
│    - operative_authority                                        │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                Step 6: Generate Summaries                        │
│  • Process PDF in 2-page chunks                                 │
│  • LLM: meta-llama/llama-4-scout-17b-16e-instruct (Groq)       │
│  • Generate concise summary for each chunk                      │
│  • Combine all summaries into complete summary                  │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│      Step 7: Store in Pinecone (Permanent - Default NS)        │
│  • Generate embedding for complete summary                      │
│  • OpenAI text-embedding-3-large (3072 dimensions)              │
│  • Create unique ID: {filename}_{resolution_no}_{year}          │
│  • Store vector + metadata in Pinecone default namespace        │
│  • Index: Uses UNSC_INDEX_NAME from environment                 │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│               Step 8: Cleanup Temporary Namespace                │
│  • Delete all vectors from temp_upload_{uuid} namespace         │
│  • Ensures no leftover data in Pinecone                         │
│  • Cleanup happens even if processing fails                     │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Step 9: Return Response                      │
│  • JSON with all metadata                                       │
│  • Complete summary                                             │
│  • Pinecone vector_id                                           │
│  • Filename                                                     │
└─────────────────────────────────────────────────────────────────┘
```

## 📊 Data Flow

### Input
```json
POST /bluelines/resolution/upload
Content-Type: multipart/form-data

file: [PDF Binary Data]
```

### Output
```json
{
  "message": "PDF processed and stored successfully",
  "data": {
    "resolution_no": "S/RES/XXXX (YYYY)",
    "year": "YYYY",
    "theme": ["Theme1", "Theme2"],
    "chapter": "Chapter VII",
    "charter_articles": ["Article 39", "Article 41"],
    "entities": ["Entity1", "Entity2"],
    "reporting_cycle": "Details...",
    "operative_authority": "Chapter VII",
    "summary": "Complete summary...",
    "filename": "uploaded_file.pdf",
    "vector_id": "uploaded_file_S_RES_XXXX_(YYYY)_YYYY"
  }
}
```

## 🔧 Technologies Used

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Web Framework** | Flask | API endpoint handling |
| **PDF Processing** | PyMuPDF (fitz) | Text extraction from PDF |
| **Text Splitting** | LangChain RecursiveCharacterTextSplitter | Intelligent text chunking |
| **Temporary Storage** | Pinecone (temp namespace) | Temporary chunk storage for metadata extraction |
| **Embeddings** | OpenAI text-embedding-3-large | Vector generation (3072 dims) |
| **LLM Processing** | Groq (Llama-4-Scout-17b) | Metadata extraction & summarization |
| **Permanent Storage** | Pinecone (default namespace) | Long-term embedding storage & retrieval |

## 🎯 Key Features

✅ **Single API Call** - Upload, process, and store in one request  
✅ **OpenAI Embeddings** - High-quality 3072-dimensional vectors  
✅ **Metadata Extraction** - Structured UN Security Council metadata  
✅ **Intelligent Summarization** - Page-by-page analysis with LLM  
✅ **Pinecone-Based Pipeline** - Uses temporary namespaces for processing  
✅ **Auto Cleanup** - Temporary data automatically removed after processing  
✅ **Batch Processing** - Efficient API usage with batch embedding  
✅ **Error Handling** - Comprehensive validation and error messages  
✅ **Concurrent Upload Safe** - UUID-based namespaces prevent conflicts  

## 🔑 Environment Variables

```env
# OpenAI (for embeddings)
OPENAI_API_KEY=sk-...

# Groq (for LLM processing)
GROQ_API_KEY=gsk_...

# Pinecone (for vector storage)
PINECONE_API_KEY=...
UNSC_INDEX_NAME=unsc-index
PINECONE_ENVIRONMENT=...
```

## 🚀 Quick Start

1. **Start the server**
   ```bash
   python run.py
   ```

2. **Upload a PDF**
   ```bash
   curl -X POST http://localhost:5001/bluelines/resolution/upload \
     -F "file=@resolution.pdf"
   ```

3. **Check health**
   ```bash
   curl http://localhost:5001/bluelines/resolution/health
   ```

## 💡 Notes

- Temporary Pinecone namespace is created per request with unique UUID
- All chunks are stored temporarily in Pinecone for metadata extraction
- Temporary namespace is automatically deleted after successful processing
- Final summary embedding is stored in Pinecone default namespace permanently
- UUID-based namespaces allow safe concurrent uploads
- Processing time: typically 1-3 minutes depending on PDF size
- Cleanup happens even if processing fails to avoid orphaned data

