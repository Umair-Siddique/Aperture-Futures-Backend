from flask import Flask
from config import Config
import tempfile
from extensions import init_supabase, init_groq, init_anthropic, init_openai_embeddings, init_pinecone, init_services
from blueprints.auth import auth_bp
from blueprints.transcribe import transcribe_bp
from blueprints.conversations import conversations_bp
from blueprints.forgot_password import forgot_password_bp
from blueprints.retriever import retriever_bp
from blueprints.reports import report_bp

# Import BlueLines blueprints
from bluelines_backend.blueprints.chat import chat_bp as bluelines_chat_bp
from bluelines_backend.blueprints.retriever_api import retriever_bp as bluelines_retriever_bp

from flask_cors import CORS

def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)
    
    # Remove file size limits for production
    # app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100 MB max file size - REMOVED
    app.config['UPLOAD_FOLDER'] = tempfile.gettempdir()

    CORS(app, supports_credentials=True, origins=['http://localhost:5173','https://blue-lines-life-lines-rag.vercel.app', 'https://aperture-futures-frontend-fcis.vercel.app'])

    # Initialize LifeLines extensions
    init_supabase(app)
    init_groq(app)
    init_anthropic(app)
    init_openai_embeddings(app)
    init_pinecone(app)
    
    # Initialize services needed by bluelines_backend
    init_services(app)

    # Register LifeLines blueprints
    app.register_blueprint(auth_bp, url_prefix="/auth")
    app.register_blueprint(transcribe_bp, url_prefix="/transcript")
    app.register_blueprint(conversations_bp, url_prefix="/conversations")
    app.register_blueprint(forgot_password_bp, url_prefix="/forgot_password")
    app.register_blueprint(retriever_bp, url_prefix="/retriever")
    app.register_blueprint(report_bp, url_prefix="/report")

    @app.route("/")
    def index():
        return {"status": "ok", "message": "Welcome to Aperture Futures Backend Updated one🚀"}

    # ✅ Test route
    @app.route("/test")
    def test():
        return {"status": "ok", "message": "Flask backend is running fine Updated one✅"}
    # Register BlueLines blueprints
    app.register_blueprint(bluelines_chat_bp, url_prefix="/bluelines/chat")
    app.register_blueprint(bluelines_retriever_bp, url_prefix="/bluelines/retriever")
    
    return app