# clear_redis.py
import redis
from config import Config

def clear_redis():
    try:
        r = redis.from_url(Config.CELERY_RESULT_BACKEND)
        r.flushdb()  # Clear current database
        print("✅ Redis cache cleared successfully!")
    except Exception as e:
        print(f"❌ Failed to clear Redis: {e}")

if __name__ == "__main__":
    clear_redis()
