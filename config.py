import os
from dotenv import load_dotenv

load_dotenv()

class Config:
	    
		print(f"GROQ_API_KEY loaded: {bool(os.getenv('GROQ_API_KEY'))}")
		
		SUPABASE_URL = os.getenv("SUPABASE_URL")
		SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY")  # For public operations
		SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")  # For admin operations
		OPENAI_API_KEY   = os.getenv("OPENAI_API_KEY")
		PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
		anthropic_api_key = os.getenv("CLAUDE_API_KEY")
		UNSC_INDEX_NAME= os.getenv("UNSC_INDEX_NAME")
		GROQ_API_KEY=os.getenv("GROQ_API_KEY")
		CLAUDE_API_KEY=os.getenv("CLAUDE_API_KEY")
		PINECONE_ENV=os.getenv("PINECONE_ENVIRONMENT")
		MEETING_TRANSCRIPTS_INDEX=os.getenv("MEETING_TRANSCRIPTS_INDEX")
		PINECONE_HOST=os.getenv('PINECONE_HOST')
		GEMINI_API_KEY=os.getenv('GEMINI_API_KEY')

		# SMTP / Email settings
		SMTP_HOST = os.getenv("SMTP_HOST", "smtp.protonmail.ch")
		SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
		SMTP_USER = os.getenv("SMTP_USER")
		SMTP_PASSWORD = os.getenv("SMTP_PASSWORD")
		SMTP_FROM = os.getenv("SMTP_FROM", SMTP_USER)
		SMTP_USE_TLS = os.getenv("SMTP_USE_TLS", "true").lower() == "true"

		# Optional: front-end URL for password reset link (if you want links to point to the frontend)
		PASSWORD_RESET_URL_BASE = os.getenv("PASSWORD_RESET_URL_BASE")  # e.g. https://app.example.com/reset-password
		FRONTEND_RESET_PASSWORD_URL = os.getenv("FRONTEND_RESET_PASSWORD_URL", "http://localhost:5173/reset-password")