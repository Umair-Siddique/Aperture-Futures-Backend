# celery_app.py
import os
import sys
from celery import Celery
from config import Config
import ssl

# Add current directory to Python path for Render deployment
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

def make_celery():
    broker = Config.CELERY_BROKER_URL
    backend = Config.CELERY_RESULT_BACKEND

    # Convert redis:// to rediss:// if SSL is required
    if broker and broker.startswith('redis://') and 'ssl=true' in broker.lower():
        broker = broker.replace('redis://', 'rediss://')
    
    if backend and backend.startswith('redis://') and 'ssl=true' in backend.lower():
        backend = backend.replace('redis://', 'rediss://')

    app = Celery(
        "transcribe",
        broker=broker,
        backend=backend,
        include=["tasks.transcribe_tasks"],
    )

    # Base configuration
    config = {
        'task_track_started': True,
        'task_acks_late': True,
        'worker_prefetch_multiplier': 1,
        'task_soft_time_limit': 60*60,   # 1h soft limit
        'task_time_limit': 60*60*2,      # 2h hard limit
        'broker_connection_retry_on_startup': True,
        'result_expires': 3600,          # Results expire after 1 hour
        
        # Serialization settings
        'task_serializer': 'json',
        'accept_content': ['json'],
        'result_accept_content': ['json'],
        'result_serializer': 'json',
        
        'timezone': 'UTC',
        'enable_utc': True,
        'worker_send_task_events': True,
        'task_send_sent_event': True,
        
        # Better error handling
        'task_reject_on_worker_lost': True,
        
        # Ensure tasks are properly imported
        'imports': ['tasks.transcribe_tasks'],
    }

    # Add SSL configuration only for rediss:// URLs
    if broker and broker.startswith('rediss://'):
        config['broker_use_ssl'] = {
            "ssl_cert_reqs": ssl.CERT_NONE,
        }
    
    if backend and backend.startswith('rediss://'):
        config['redis_backend_use_ssl'] = {
            "ssl_cert_reqs": ssl.CERT_NONE,
        }

    app.conf.update(config)
    
    return app

celery_app = make_celery()
