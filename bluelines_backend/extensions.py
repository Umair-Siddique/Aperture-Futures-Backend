from supabase import create_client
from config import Config
from langchain_openai import OpenAIEmbeddings
from langchain_openai import OpenAI
from openai import OpenAI as OpenAIClient
import anthropic
from pinecone import Pinecone
from groq import Groq

def init_supabase(app):
    app.supabase = create_client(Config.SUPABASE_URL, Config.SUPABASE_KEY)

def init_groq(app):
    app.groq = Groq(api_key=Config.GROQ_API_KEY)

def init_anthropic(app):
    app.anthropic = anthropic.Anthropic(api_key=Config.CLAUDE_API_KEY)
    
def init_services(app):
    # Initialize embeddings
    app.embeddings = OpenAIEmbeddings(
        model="text-embedding-3-large",
        openai_api_key=Config.OPENAI_API_KEY,
    )

    # Initialize LLMs and clients
    app.llm = OpenAI(temperature=0, api_key=Config.OPENAI_API_KEY)
    app.openai_client = OpenAIClient(api_key=Config.OPENAI_API_KEY)
    app.anthropic_client = anthropic.Anthropic(api_key=Config.anthropic_api_key)
    app.pc = Pinecone(api_key=Config.PINECONE_API_KEY)

    # Initialize a simple retriever (without the complex SelfQueryRetriever for now)
    # This will be used by the conditional_retrieval function
    app.retriever = None  # We'll handle retrieval differently in the retriever_api.py