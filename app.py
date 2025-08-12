from flask import Flask
from config import Config
from dotenv import load_dotenv
from extensions import init_supabase,init_groq,init_anthropic,init_openai_embeddings,init_pinecone
from blueprints.auth import auth_bp
from blueprints.transcribe import transcribe_bp
from blueprints.transcribe_video import transcribe_video_bp

from flask_cors import CORS

def create_app():

    app = Flask(__name__)
    app.config.from_object(Config)

    CORS(app, supports_credentials=True, origins=['http://localhost:5173','http://bluelines-rag.s3-website.eu-north-1.amazonaws.com'])

    # Initialize extensions
    init_supabase(app)
    init_groq(app)
    init_anthropic(app)
    init_openai_embeddings(app)
    init_pinecone(app)

    # Register blueprints
    app.register_blueprint(auth_bp, url_prefix="/auth")
    app.register_blueprint(transcribe_bp,url_prefix="/transcribe")
    app.register_blueprint(transcribe_video_bp,url_prefix="/transcribe_video")

    return app