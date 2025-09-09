import os
import json
from pinecone import Pinecone
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# Configuration variables
JSON_FOLDER = './processed_files_2025'
INDEX_NAME =  os.getenv('UNSC_INDEX_NAME')
EMBEDDING_MODEL = 'text-embedding-3-large'
OPENAI_API_KEY =  os.getenv('OPENAI_API_KEY')
PINECONE_API_KEY =  os.getenv('PINECONE_API_KEY')
PINECONE_ENVIRONMENT =  os.getenv('PINECONE_ENVIRONMENT')

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)

# Connect to existing index
index = pc.Index(INDEX_NAME)

# Initialize OpenAI
client = OpenAI(api_key=OPENAI_API_KEY)


# Batch size for embedding requests
BATCH_SIZE = 10  # Adjust this if needed (OpenAI supports up to 2048 tokens per request)

def process_and_upload():
    items = []

    # Load and prepare all JSON data
    for filename in os.listdir(JSON_FOLDER):
        if filename.endswith('.json'):
            filepath = os.path.join(JSON_FOLDER, filename)
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)

            content = data.get('summary', '')
            if not content:
                continue  # Skip empty summaries

            metadata = {
                'resolution_no': data.get('resolution_no'),
                'year': data.get('year'),
                'theme': data.get('theme'),
                'chapter': data.get('chapter'),
                'charter_articles': data.get('charter_articles'),
                'entities': data.get('entities'),
                'reporting_cycle': data.get('reporting_cycle'),
                'operative_authority': data.get('operative_authority'),
                'summary': content,
                'url': data.get('url')
            }
            basename = os.path.splitext(filename)[0]
            items.append({
                'id': f"{basename}_{data['resolution_no']}_{data['year']}",
                'content': content,
                'metadata': metadata
            })

    # Process in batches
    for i in range(0, len(items), BATCH_SIZE):
        batch = items[i:i + BATCH_SIZE]
        contents = [item['content'] for item in batch]

        response = client.embeddings.create(
            input=contents,
            model=EMBEDDING_MODEL
        )

        vectors = []
        for item, embedding_obj in zip(batch, response.data):
            vectors.append({
                'id': item['id'],
                'values': embedding_obj.embedding,
                'metadata': item['metadata']
            })

        index.upsert(vectors)
        print(f"Uploaded batch {i // BATCH_SIZE + 1} with {len(vectors)} items.")

if __name__ == "__main__":
    process_and_upload()





# # Process JSON files
# def process_and_upload():
#     for filename in os.listdir(JSON_FOLDER):
#         if filename.endswith('.json'):
#             filepath = os.path.join(JSON_FOLDER, filename)
#             with open(filepath, 'r', encoding='utf-8') as f:
#                 data = json.load(f)

#             content = data.get('summary', '')
#             metadata = {
#                 'resolution_no': data.get('resolution_no'),
#                 'year': data.get('year'),
#                 'theme': data.get('theme'),
#                 'chapter': data.get('chapter'),
#                 'charter_articles': data.get('charter_articles'),
#                 'entities': data.get('entities'),
#                 'reporting_cycle': data.get('reporting_cycle'),
#                 'operative_authority': data.get('operative_authority'),
#                 'summary': data.get('summary'),
#                 'url': data.get('url')
#             }

#             embedding_response = client.embeddings.create(
#                 input=content,
#                 model=EMBEDDING_MODEL
#             )

#             embedding = embedding_response.data[0].embedding

#             vector = {
#                 'id': f"{data['resolution_no']}_{data['year']}",
#                 'values': embedding,
#                 'metadata': metadata
#             }

#             index.upsert([vector])
#             print(f"Uploaded: {filename}")

# if __name__ == "__main__":
#     process_and_upload()

