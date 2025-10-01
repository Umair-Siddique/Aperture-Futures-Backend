# Render Deployment Guide for Celery + Flask App

## Problem Summary

You were getting a `NotRegistered('transcribe.audio')` error because of a **circular import** between `celery_app.py` and `tasks/transcribe_tasks.py`. This has been fixed.

## What Was Fixed

### 1. **celery_app.py** - Removed circular import
- Removed `include=["tasks.transcribe_tasks"]` from Celery constructor
- Removed `imports` from config
- Moved task import to **AFTER** `celery_app` is created
- This ensures `celery_app` exists before tasks try to import it

### 2. **tasks/__init__.py** - Created package marker
- Added `__init__.py` to make `tasks/` a proper Python package

### 3. **worker.py** - Created worker entry point (optional)
- Alternative way to start the worker with verification

## Render Configuration

### Services You Need

You should have **TWO** services on Render:

#### 1. **Web Service** (Flask API)
- **Build Command**: `./build.sh`
- **Start Command**: `gunicorn run:app`
- **Environment Variables**: All your existing env vars

#### 2. **Background Worker** (Celery)
- **Build Command**: `./build.sh`
- **Start Command**: `celery -A celery_app worker --loglevel=info`
- **Environment Variables**: Same as Web Service (especially `CELERY_BROKER_URL` and `CELERY_RESULT_BACKEND`)

### Important: Redis Configuration

Your Celery worker and Flask API must connect to the **SAME** Redis instance.

#### Environment Variables Required:

```bash
# Both services need these:
CELERY_BROKER_URL=redis://your-redis-host:6379/0
CELERY_RESULT_BACKEND=redis://your-redis-host:6379/0

# If using Redis with SSL (Render Redis or Upstash):
CELERY_BROKER_URL=rediss://your-redis-host:6380
CELERY_RESULT_BACKEND=rediss://your-redis-host:6380
```

**Note**: The code automatically handles `rediss://` (SSL) connections.

### Render Redis Options:

1. **Render Redis** (recommended for simplicity):
   - Create a Redis instance on Render
   - Use the **Internal Connection String** for both services
   - Example: `redis://red-xyz123:6379`

2. **External Redis** (Upstash, Redis Cloud, etc.):
   - Use the connection string provided
   - Often uses `rediss://` for SSL

## Testing Before Deployment

### Test 1: Verify Task Registration (Local)

```bash
python test_celery_registration.py
```

Expected output:
```
✓ Successfully imported celery_app
✓ transcribe.audio - FOUND
✓ transcribe.video - FOUND
✓ SUCCESS: All required tasks are registered!
```

### Test 2: Start Celery Worker (Local)

```bash
# Make sure Redis is running locally or set CELERY_BROKER_URL to remote Redis
celery -A celery_app worker --loglevel=info
```

You should see:
```
[tasks]
  . transcribe.audio
  . transcribe.video
```

### Test 3: Test Task Execution (Local)

```bash
python test_celery.py
```

## Deployment Steps

### Step 1: Commit and Push Changes

```bash
git add .
git commit -m "Fix Celery task registration - remove circular import"
git push origin main
```

### Step 2: Configure Render Services

#### Web Service (if not already created):
1. Go to Render Dashboard
2. New → Web Service
3. Connect your Git repository
4. **Build Command**: `./build.sh`
5. **Start Command**: `gunicorn run:app` or `gunicorn app:app --bind 0.0.0.0:$PORT`
6. Add all environment variables

#### Celery Worker Service:
1. Go to Render Dashboard
2. New → Background Worker
3. Connect the **SAME** Git repository
4. **Build Command**: `./build.sh`
5. **Start Command**: `celery -A celery_app worker --loglevel=info`
6. Add **ALL** environment variables (same as Web Service)

**CRITICAL**: Both services must have:
- Same `CELERY_BROKER_URL`
- Same `CELERY_RESULT_BACKEND`
- All other env vars needed by your tasks (OPENAI_API_KEY, SUPABASE_URL, etc.)

### Step 3: Deploy

1. Push to GitHub
2. Render will automatically deploy both services
3. Check logs for both services

#### Web Service Logs - Should show:
```
Starting gunicorn...
Worker started
```

#### Celery Worker Logs - Should show:
```
Successfully imported tasks.transcribe_tasks
Registered tasks: [..., 'transcribe.audio', 'transcribe.video']
celery@srv-xxx ready.
[tasks]
  . transcribe.audio
  . transcribe.video
```

### Step 4: Test the API

```bash
curl -X POST https://your-app.onrender.com/transcript/audio \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -F "title=Test" \
  -F "description=Test description" \
  -F "members=Member1,Member2" \
  -F "audio=@test.mp3"
```

Expected response:
```json
{
  "task_id": "abc-123-def-456"
}
```

Then check status:
```bash
curl https://your-app.onrender.com/transcript/tasks/abc-123-def-456 \
  -H "Authorization: Bearer YOUR_TOKEN"
```

## Troubleshooting

### Error: NotRegistered('transcribe.audio')

**Causes:**
1. Celery worker doesn't have the same code as Web Service
2. Worker isn't finding the tasks module
3. Circular import (should be fixed now)

**Solutions:**
- Ensure both services are deploying from the same Git commit
- Check worker logs for import errors
- Verify `tasks/__init__.py` exists
- Run `test_celery_registration.py` locally

### Error: Connection Refused / Timeout

**Causes:**
1. Redis connection string is wrong
2. Services can't reach Redis
3. SSL/TLS mismatch

**Solutions:**
- Use Render's **Internal Connection String** for Redis
- Ensure both services are in the same region as Redis
- Check `rediss://` vs `redis://` (code handles both)

### Tasks Stuck in PENDING

**Causes:**
1. Worker isn't running
2. Worker can't connect to Redis
3. Wrong Redis instance

**Solutions:**
- Check Celery worker logs in Render dashboard
- Verify `CELERY_BROKER_URL` is identical in both services
- Restart the worker service

### Worker Keeps Restarting

**Causes:**
1. Import errors in task code
2. Missing environment variables
3. Out of memory

**Solutions:**
- Check worker logs for Python errors
- Ensure ALL env vars from Web Service are in Worker
- Increase worker instance size if needed

## Environment Variables Checklist

Make sure **BOTH** services have:

- [ ] `CELERY_BROKER_URL`
- [ ] `CELERY_RESULT_BACKEND`
- [ ] `OPENAI_API_KEY`
- [ ] `SUPABASE_URL`
- [ ] `SUPABASE_ANON_KEY`
- [ ] `SUPABASE_SERVICE_ROLE_KEY`
- [ ] `PINECONE_API_KEY`
- [ ] `PINECONE_HOST`
- [ ] `MEETING_TRANSCRIPTS_INDEX`
- [ ] `GROQ_API_KEY`
- [ ] `CLAUDE_API_KEY`
- [ ] Any other env vars your tasks use

## Monitoring

### View Worker Status:
1. Go to Render Dashboard
2. Click on your Celery Worker service
3. Click "Logs" tab
4. Should see tasks being registered and processed

### View Task Status:
Use the Flower monitoring tool (optional):
```bash
# Add to requirements.txt:
flower

# Start command for a separate monitoring service:
celery -A celery_app flower --port=5555
```

## Additional Resources

- [Celery on Render Guide](https://render.com/docs/deploy-celery)
- [Render Redis Documentation](https://render.com/docs/redis)
- [Celery Documentation](https://docs.celeryproject.org/)

## Quick Reference

### Local Development:
```bash
# Terminal 1: Redis (if local)
redis-server

# Terminal 2: Celery Worker
celery -A celery_app worker --loglevel=info

# Terminal 3: Flask App
python run.py
```

### Render Services:

| Service Type | Start Command |
|-------------|---------------|
| Web (Flask) | `gunicorn run:app` |
| Worker (Celery) | `celery -A celery_app worker --loglevel=info` |
| Monitor (Flower) | `celery -A celery_app flower --port=5555` |

