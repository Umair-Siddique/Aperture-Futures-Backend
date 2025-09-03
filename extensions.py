from supabase import create_client
from config import Config
import anthropic
from groq import Groq
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone

def init_supabase(app):
    # Create two clients: one with anon key for public operations, one with service role for admin
    app.supabase = create_client(Config.SUPABASE_URL, Config.SUPABASE_ANON_KEY)
    app.supabase_admin = create_client(Config.SUPABASE_URL, Config.SUPABASE_SERVICE_ROLE_KEY)

def init_groq(app):
    app.groq = Groq(api_key=Config.GROQ_API_KEY)
def init_anthropic(app):
    app.anthropic = anthropic.Anthropic(api_key=Config.CLAUDE_API_KEY)

def init_openai_embeddings(app):
    app.embeddings = OpenAIEmbeddings(
        model="text-embedding-3-large",
        openai_api_key=Config.OPENAI_API_KEY,
    )

def init_pinecone(app):
    pc = Pinecone(api_key=Config.PINECONE_API_KEY)
    app.pinecone_index = pc.Index(name=Config.MEETING_TRANSCRIPTS_INDEX,host=Config.PINECONE_HOST)