# test_celery.py
from celery_app import celery_app
from tasks.transcribe_tasks import transcribe_audio_task
import time

if __name__ == "__main__":
    print("Testing with existing transcribe.audio task...")
    
    # Test with a dummy file path (this will fail but we can see if the task is picked up)
    result = transcribe_audio_task.delay(
        title="Test Audio",
        description="Test transcription", 
        members_list=["Test User"],
        file_path="dummy_path.mp3"  # This will fail but we can see the task is processed
    )
    
    print(f"Task submitted with ID: {result.id}")
    print(f"Task state: {result.state}")
    
    # Wait a bit and check again
    time.sleep(2)
    print(f"Task state after 2 seconds: {result.state}")
    
    if result.state == 'FAILURE':
        print(f"Task failed as expected: {result.result}")
    elif result.state == 'SUCCESS':
        print(f"Task succeeded: {result.result}")
    else:
        print(f"Task still in state: {result.state}")
