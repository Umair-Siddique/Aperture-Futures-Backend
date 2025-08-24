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
		PINECONE_HOST=os.getenv('PINECONE_HOST')

		# SMTP / Email settings
		SMTP_HOST = os.getenv("SMTP_HOST", "smtp.protonmail.ch")
		SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
		SMTP_USER = os.getenv("SMTP_USER")
		SMTP_PASSWORD = os.getenv("SMTP_PASSWORD")
		SMTP_FROM = os.getenv("SMTP_FROM", SMTP_USER)
		SMTP_USE_TLS = os.getenv("SMTP_USE_TLS", "true").lower() == "true"

		# Optional: front-end URL for password reset link (if you want links to point to the frontend)
		PASSWORD_RESET_URL_BASE = os.getenv("PASSWORD_RESET_URL_BASE")  # e.g. https://app.example.com/reset-password