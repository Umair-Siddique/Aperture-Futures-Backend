#!/usr/bin/env python
"""
Celery worker entry point for Render deployment.
This file ensures tasks are properly loaded before the worker starts.
"""

import os
import sys

# Add current directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Import celery app - this will automatically import tasks at the end of celery_app.py
from celery_app import celery_app

# Verify tasks are registered
print("=" * 60)
print("Celery Worker Starting")
print("=" * 60)
print(f"Registered tasks: {list(celery_app.tasks.keys())}")
print("=" * 60)

if __name__ == '__main__':
    # Start the worker
    celery_app.worker_main()

