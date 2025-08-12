import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    SUPABASE_URL = os.getenv("SUPABASE_URL")
    SUPABASE_KEY = os.getenv("SUPABASE_ANON_KEY")
    OPENAI_API_KEY   = os.getenv("OPENAI_API_KEY")
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    anthropic_api_key = os.getenv("CLAUDE_API_KEY")
    UNSC_INDEX_NAME= os.getenv("UNSC_INDEX_NAME")
    GROQ_API_KEY=os.getenv("GROQ_API_KEY")
    CLAUDE_API_KEY=os.getenv("CLAUDE_API_KEY")
    PINECONE_ENV=os.getenv("PINECONE_ENVIRONMENT")
    MEETING_TRANSCRIPTS_INDEX=os.getenv("MEETING_TRANSCRIPTS_INDEX")