# celery_app.py
import os
from celery import Celery
from config import Config
import ssl

def make_celery():
    broker = Config.CELERY_BROKER_URL
    backend = Config.CELERY_RESULT_BACKEND

    app = Celery(
        "transcribe",
        broker=broker,
        backend=backend,
        include=["tasks.transcribe_tasks"],
    )

    app.conf.update(
        broker_use_ssl={
            "ssl_cert_reqs": ssl.CERT_NONE,
        },
        redis_backend_use_ssl={
            "ssl_cert_reqs": ssl.CERT_NONE,
        },
        task_track_started=True,
        task_acks_late=True,
        worker_prefetch_multiplier=1,
        task_soft_time_limit=60*60,   # 1h soft limit
        task_time_limit=60*60*2,      # 2h hard limit
        broker_connection_retry_on_startup=True,
        result_expires=3600,          # Results expire after 1 hour
        
        # Serialization settings
        task_serializer='json',
        accept_content=['json'],
        result_accept_content=['json'],
        result_serializer='json',
        
        timezone='UTC',
        enable_utc=True,
        worker_send_task_events=True,
        task_send_sent_event=True,
        
        # Better error handling
        task_reject_on_worker_lost=True,
    )
    return app

celery_app = make_celery()
