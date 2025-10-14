from supabase import create_client
from config import Config
import anthropic
from groq import Groq
from langchain_openai import OpenAIEmbeddings
from langchain_openai import OpenAI
from openai import OpenAI as OpenAIClient
from pinecone import Pinecone

def get_api_key_from_supabase(supabase_client, key_name, fallback_key=None):
    """Get API key from Supabase, with fallback to environment variable"""
    try:
        result = supabase_client.table('api_keys').select('key_value').eq('key_name', key_name).single().execute()
        if result.data and result.data.get('key_value'):
            return result.data['key_value']
    except Exception as e:
        print(f"Warning: Could not fetch {key_name} from Supabase: {e}")
    
    # Fallback to environment variable
    if fallback_key:
        return fallback_key
    return None

def init_supabase(app):
    # Create two clients: one with anon key for public operations, one with service role for admin
    app.supabase = create_client(Config.SUPABASE_URL, Config.SUPABASE_ANON_KEY)
    app.supabase_admin = create_client(Config.SUPABASE_URL, Config.SUPABASE_SERVICE_ROLE_KEY)

def init_groq(app):
    # Try to get GROQ API key from Supabase first, fallback to environment
    groq_key = get_api_key_from_supabase(app.supabase_admin, 'GROQ_API_KEY', Config.GROQ_API_KEY)
    
    if not groq_key:
        raise ValueError("GROQ_API_KEY not found in Supabase or environment variables.")
    
    app.groq = Groq(api_key=groq_key)
    print("NEW DEPLOY v2")
    print("✓ GROQ client initialized successfully Updated one")

def init_anthropic(app):
    if not Config.CLAUDE_API_KEY:
        raise ValueError("CLAUDE_API_KEY not found in environment variables. Please check your .env file.")
    app.anthropic = anthropic.Anthropic(api_key=Config.CLAUDE_API_KEY)

def init_openai_embeddings(app):
    if not Config.OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY not found in environment variables. Please check your .env file.")
    app.embeddings = OpenAIEmbeddings(
        model="text-embedding-3-large",
        openai_api_key=Config.OPENAI_API_KEY,
    )

def init_services(app):
    # Initialize embeddings
    if not Config.OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY not found in environment variables. Please check your .env file.")
    app.embeddings = OpenAIEmbeddings(
        model="text-embedding-3-large",
        openai_api_key=Config.OPENAI_API_KEY,
    )

    # Initialize LLMs and clients
    app.llm = OpenAI(temperature=0, api_key=Config.OPENAI_API_KEY)
    app.openai_client = OpenAIClient(api_key=Config.OPENAI_API_KEY)
    
    if not Config.CLAUDE_API_KEY:
        raise ValueError("CLAUDE_API_KEY not found in environment variables. Please check your .env file.")
    app.anthropic = anthropic.Anthropic(api_key=Config.CLAUDE_API_KEY)
    
    if not Config.PINECONE_API_KEY:
        raise ValueError("PINECONE_API_KEY not found in environment variables. Please check your .env file.")
    app.pc = Pinecone(api_key=Config.PINECONE_API_KEY)

    app.retriever = None  

def init_pinecone(app):
    if not Config.PINECONE_API_KEY:
        raise ValueError("PINECONE_API_KEY not found in environment variables. Please check your .env file.")
    pc = Pinecone(api_key=Config.PINECONE_API_KEY)
    app.pinecone_index = pc.Index(name=Config.MEETING_TRANSCRIPTS_INDEX, host=Config.PINECONE_HOST)