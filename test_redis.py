# test_redis.py
import redis
from config import Config

def test_redis_connection():
    try:
        # Parse Redis URL
        redis_url = Config.CELERY_BROKER_URL
        print(f"Testing Redis connection to: {redis_url}")
        
        # Create Redis client
        r = redis.from_url(redis_url)
        
        # Test connection
        r.ping()
        print("✅ Redis connection successful!")
        
        # Test basic operations
        r.set('test_key', 'test_value')
        value = r.get('test_key')
        print(f"✅ Redis read/write test successful: {value}")
        
        # Clean up
        r.delete('test_key')
        
    except Exception as e:
        print(f"❌ Redis connection failed: {e}")
        print("Make sure your Redis server is running and accessible")

if __name__ == "__main__":
    test_redis_connection()
