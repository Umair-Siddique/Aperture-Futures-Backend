# Upload Resolution API Documentation

## Overview
This API endpoint accepts a single PDF file, processes it using OpenAI embeddings (text-embedding-3-large with 3072 dimensions), extracts metadata and summaries, and stores the embedding in Pinecone for retrieval.

## Endpoint
```
POST /bluelines/resolution/upload
```

## Request Format
- **Content-Type**: `multipart/form-data`
- **Required Field**: `file` (PDF file)

## Example Usage

### Using cURL
```bash
curl -X POST http://localhost:5001/bluelines/resolution/upload \
  -F "file=@/path/to/resolution.pdf"
```

### Using Python (requests)
```python
import requests

url = "http://localhost:5001/bluelines/resolution/upload"
files = {'file': open('resolution.pdf', 'rb')}

response = requests.post(url, files=files)
print(response.json())
```

### Using JavaScript (fetch)
```javascript
const formData = new FormData();
formData.append('file', pdfFile); // pdfFile is a File object

fetch('http://localhost:5001/bluelines/resolution/upload', {
  method: 'POST',
  body: formData
})
.then(response => response.json())
.then(data => console.log(data))
.catch(error => console.error('Error:', error));
```

## Response Format

### Success Response (200 OK)
```json
{
  "message": "PDF processed and stored successfully",
  "data": {
    "resolution_no": "S/RES/2758 (2024)",
    "year": "2024",
    "theme": ["Peacekeeping", "Humanitarian"],
    "chapter": "Chapter VII",
    "charter_articles": ["Article 39", "Article 41"],
    "entities": ["UNIFIL", "MINUSMA"],
    "reporting_cycle": "Secretary-General reports every 90 days",
    "operative_authority": "Chapter VII",
    "summary": "Detailed summary of the resolution...",
    "filename": "resolution.pdf",
    "vector_id": "resolution_S_RES_2758_(2024)_2024"
  }
}
```

### Error Responses

#### 400 Bad Request - No file provided
```json
{
  "error": "No file provided"
}
```

#### 400 Bad Request - No file selected
```json
{
  "error": "No file selected"
}
```

#### 400 Bad Request - Wrong file type
```json
{
  "error": "Only PDF files are allowed"
}
```

#### 500 Internal Server Error
```json
{
  "error": "Failed to process PDF: <error details>"
}
```

## Processing Pipeline

### 1. Text Extraction
- Extracts all text from the PDF using PyMuPDF (fitz)
- Removes common footer text

### 2. Text Chunking
- Uses `RecursiveCharacterTextSplitter`
- Chunk size: 2500 characters
- Chunk overlap: 250 characters
- Separators: `\n\n`, `\n`, `.`, `!`

### 3. Temporary Embedding Storage (Pinecone)
- Embeds all chunks using OpenAI API
- Stores in Pinecone temporary namespace (`temp_upload_{unique_id}`)
- Processes in batches of 10 for efficiency
- Each chunk stored with its text in metadata for retrieval

### 4. Metadata Extraction
- Retrieves top 15 relevant chunks from Pinecone temporary namespace
- Uses Groq LLM (meta-llama/llama-4-scout-17b-16e-instruct)
- Extracts structured metadata based on UN Charter context
- Returns: resolution number, year, themes, chapter, charter articles, entities, reporting cycle, operative authority

### 5. Summary Generation
- Processes document in 2-page chunks
- Generates concise summaries maintaining original tone
- Uses Groq LLM for generation
- Combines all summaries into final output

### 6. Permanent Pinecone Storage
- Generates embedding for the complete summary
- Creates unique vector ID: `{filename}_{resolution_no}_{year}`
- Stores in Pinecone default namespace with all metadata
- Returns vector ID in response

### 7. Cleanup
- Deletes all vectors from temporary namespace
- Ensures no leftover temporary data in Pinecone

## Environment Variables Required

```env
# Required for processing
GROQ_API_KEY=your_groq_api_key_here
OPENAI_API_KEY=your_openai_api_key_here

# Required for Pinecone storage
PINECONE_API_KEY=your_pinecone_api_key_here
UNSC_INDEX_NAME=your_pinecone_index_name
PINECONE_ENVIRONMENT=your_pinecone_environment
```

## Technical Details

### Embedding Model
- **Model**: `text-embedding-3-large`
- **Dimensions**: 3072
- **Provider**: OpenAI

### LLM for Metadata & Summaries
- **Model**: `meta-llama/llama-4-scout-17b-16e-instruct`
- **Provider**: Groq
- **Temperature**: 0.6
- **Max Tokens**: 1000 (summaries), 3000 (metadata)

### Vector Database
- **Provider**: Pinecone
- **Index**: Uses `UNSC_INDEX_NAME` from environment
- **Temporary Namespace**: `temp_upload_{uuid}` (auto-cleaned after processing)
- **Permanent Namespace**: Default namespace (empty string)
- **Metadata Stored**: All extracted fields + summary

## Health Check Endpoint
```
GET /bluelines/resolution/health
```

Returns:
```json
{
  "status": "healthy"
}
```

## Notes
- Only PDF files are accepted
- Processing time depends on PDF size and content (typically 1-3 minutes)
- OpenAI API costs apply per embedding call
- The uploaded PDF is not saved to disk
- Temporary namespace is automatically cleaned up after processing
- All chunks are temporarily stored in Pinecone for metadata extraction
- Final embedding is stored in Pinecone default namespace for long-term retrieval
- Each document gets a unique vector ID based on filename, resolution number, and year
- Temporary namespaces use UUID to avoid conflicts between concurrent uploads

