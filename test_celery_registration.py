#!/usr/bin/env python
"""
Test script to verify Celery tasks are properly registered.
Run this before deploying to Render to ensure tasks will load correctly.
"""

import os
import sys

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

def test_celery_registration():
    print("=" * 60)
    print("Testing Celery Task Registration")
    print("=" * 60)
    
    try:
        # Import celery app
        from celery_app import celery_app
        
        print("\n✓ Successfully imported celery_app")
        
        # Check registered tasks
        registered_tasks = list(celery_app.tasks.keys())
        print(f"\nRegistered tasks ({len(registered_tasks)}):")
        for task in registered_tasks:
            print(f"  - {task}")
        
        # Check for our specific tasks
        required_tasks = ['transcribe.audio', 'transcribe.video']
        missing_tasks = []
        
        print("\nChecking for required tasks:")
        for task in required_tasks:
            if task in registered_tasks:
                print(f"  ✓ {task} - FOUND")
            else:
                print(f"  ✗ {task} - MISSING")
                missing_tasks.append(task)
        
        if missing_tasks:
            print(f"\n✗ ERROR: Missing tasks: {missing_tasks}")
            print("\nThis means the Celery worker won't be able to execute these tasks.")
            return False
        else:
            print("\n✓ SUCCESS: All required tasks are registered!")
            print("\nYour Celery setup should work correctly on Render.")
            return True
            
    except Exception as e:
        print(f"\n✗ ERROR: Failed to import celery_app or tasks")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("=" * 60)

if __name__ == "__main__":
    success = test_celery_registration()
    sys.exit(0 if success else 1)

