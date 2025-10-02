# tasks/__init__.py
from celery_app import celery_app

# Make the celery app available as 'celery' for Render
celery = celery_app
