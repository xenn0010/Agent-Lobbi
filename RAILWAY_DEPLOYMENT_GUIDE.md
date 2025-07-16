# Agent Lobbi Backend - Railway Deployment Guide

## Overview

This guide will help you deploy the Agent Lobbi backend (with integrated A2A API bridge) to Railway. The deployment includes:

- **Agent Lobby Core**: Main agent orchestration system
- **A2A API Bridge**: Enhanced API bridge with A2A protocol support
- **Metrics Dashboard**: Real-time monitoring and analytics
- **WebSocket Support**: Real-time communication
- **PostgreSQL Database**: Persistent data storage
- **Redis Cache**: Optional caching layer

## Prerequisites

1. **Railway Account**: Sign up at [railway.app](https://railway.app)
2. **Railway CLI**: Install the Railway CLI tool
3. **Git**: Ensure your code is in a Git repository

## Step 1: Install Railway CLI

```bash
# Install Railway CLI
npm install -g @railway/cli

# Or using curl
curl -fsSL https://railway.app/install.sh | sh
```

## Step 2: Login to Railway

```bash
railway login
```

This will open your browser to authenticate with Railway.

## Step 3: Initialize Railway Project

Navigate to your project directory and run:

```bash
# Initialize Railway project
railway init

# Select "Empty Project" when prompted
# Give your project a name like "agent-lobbi-backend"
```

## Step 4: Add PostgreSQL Database

In your Railway dashboard:

1. Click **"Add Service"** → **"Database"** → **"PostgreSQL"**
2. Railway will automatically create a PostgreSQL instance
3. The `DATABASE_URL` environment variable will be automatically set

## Step 5: Add Redis (Optional but Recommended)

In your Railway dashboard:

1. Click **"Add Service"** → **"Database"** → **"Redis"**
2. Railway will automatically create a Redis instance
3. The `REDIS_URL` environment variable will be automatically set

## Step 6: Configure Environment Variables

In your Railway dashboard, go to your service and add these environment variables:

### Required Variables

```bash
# Server Configuration
PORT=8080
HOST=0.0.0.0
LOBBY_HTTP_PORT=8080
LOBBY_WS_PORT=8081

# Environment
ENV=production
LOG_LEVEL=INFO
DEBUG=false

# Security (Generate secure random strings)
SECRET_KEY=your-secure-secret-key-here
JWT_SECRET_KEY=your-secure-jwt-secret-here
JWT_ALGORITHM=HS256
JWT_EXPIRE_HOURS=24

# CORS Configuration
CORS_ORIGINS=*
CORS_CREDENTIALS=true
CORS_METHODS=GET,POST,PUT,DELETE,OPTIONS
CORS_HEADERS=*

# API Configuration
API_VERSION=v1
MAX_WORKERS=1

# Monitoring
ENABLE_METRICS=true
HEALTH_CHECK_ENABLED=true

# Agent Configuration
MAX_AGENTS=100
DEFAULT_TIMEOUT=30
MAX_TASK_DURATION=300

# Feature Flags
ENABLE_A2A_BRIDGE=true
ENABLE_WEBSOCKETS=true
ENABLE_REAL_TIME_METRICS=true
```

### Auto-Generated Variables (Railway provides these)

```bash
DATABASE_URL=${POSTGRES_DATABASE_URL}  # Auto-provided
REDIS_URL=${REDIS_URL}  # Auto-provided if Redis is added
```

## Step 7: Deploy to Railway

```bash
# Deploy your application
railway up

# Or connect to GitHub and enable auto-deploys
railway link
```

## Step 8: Verify Deployment

Once deployed, your application will be available at a Railway-provided URL like:
`https://your-app.up.railway.app`

### Health Check Endpoints

Test these endpoints to verify deployment:

- **Health Check**: `https://your-app.up.railway.app/api/health`
- **API Documentation**: `https://your-app.up.railway.app/docs`
- **Metrics Dashboard**: `https://your-app.up.railway.app/metrics/dashboard`
- **System Status**: `https://your-app.up.railway.app/api/status`

## Step 9: Custom Domain (Optional)

1. In Railway dashboard, go to **Settings** → **Domains**
2. Add your custom domain
3. Follow Railway's DNS configuration instructions

## Configuration Files

The following files are configured for Railway deployment:

### `railway.toml`
```toml
[build]
builder = "nixpacks"
buildCommand = "pip install -r requirements.txt"

[deploy]
startCommand = "python -m src.railway_main"
healthcheckPath = "/api/health"
healthcheckTimeout = 300
restartPolicyType = "on_failure"

[experimental]
incrementalDeploy = true

[[services]]
name = "agent-lobbi-backend"

[services.variables]
PORT = "8080"
LOBBY_HTTP_PORT = "8080"
LOBBY_WS_PORT = "8081"
DATABASE_URL = "${{ Postgres.DATABASE_URL }}"
REDIS_URL = "${{ Redis.REDIS_URL }}"
ENV = "production"
LOG_LEVEL = "INFO"
CORS_ORIGINS = "*"
API_BASE_URL = "https://agent-lobbi-backend.up.railway.app"
MAX_WORKERS = "1"
```

### `Procfile`
```
web: python -m src.railway_main
```

### `requirements.txt`
Contains all necessary Python dependencies optimized for Railway deployment.

## Monitoring and Logs

### View Logs
```bash
# View real-time logs
railway logs

# Follow logs
railway logs --follow
```

### Monitoring Dashboard
Access the built-in metrics dashboard at:
`https://your-app.up.railway.app/metrics/dashboard`

## Scaling

Railway automatically handles scaling based on your plan:

- **Hobby Plan**: 1 vCPU, 512MB RAM
- **Pro Plan**: 2-8 vCPUs, 1-8GB RAM
- **Team Plan**: Custom scaling

## Database Management

### Access Database
```bash
# Get database connection details
railway variables

# Connect to PostgreSQL
railway connect postgres
```

### Run Migrations
```bash
# If you have Alembic migrations
railway run python -m alembic upgrade head
```

## Troubleshooting

### Common Issues

1. **Port Binding Error**
   - Ensure `HOST=0.0.0.0` is set
   - Use Railway's `PORT` environment variable

2. **Database Connection Issues**
   - Verify `DATABASE_URL` is set
   - Check PostgreSQL service is running

3. **Memory Issues**
   - Reduce `MAX_WORKERS` to 1
   - Optimize memory usage in application

4. **WebSocket Issues**
   - Ensure CORS is properly configured
   - Check Railway's WebSocket support

### Debug Commands

```bash
# Check service status
railway status

# View environment variables
railway variables

# Open Railway dashboard
railway open

# SSH into container (if needed)
railway shell
```

## API Usage Examples

Once deployed, you can interact with your API:

### Health Check
```bash
curl https://your-app.up.railway.app/api/health
```

### A2A Agent Discovery
```bash
curl https://your-app.up.railway.app/.well-known/agent.json
```

### Submit Task
```bash
curl -X POST https://your-app.up.railway.app/api/a2a/task \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Test Task",
    "description": "A test task for the agent system",
    "required_capabilities": ["general"]
  }'
```

## Security Considerations

1. **Environment Variables**: Never commit secrets to Git
2. **CORS**: Configure appropriate CORS origins for production
3. **Authentication**: Implement proper authentication for production use
4. **HTTPS**: Railway provides HTTPS by default
5. **Rate Limiting**: Consider implementing rate limiting

## Cost Optimization

1. **Resource Monitoring**: Monitor CPU/memory usage
2. **Database Optimization**: Optimize queries and indexes
3. **Caching**: Use Redis for frequently accessed data
4. **Scaling**: Scale based on actual usage patterns

## Support

- **Railway Support**: [railway.app/help](https://railway.app/help)
- **Agent Lobbi Documentation**: Check the `/docs` endpoint
- **Community**: Railway Discord community

## Next Steps

After successful deployment:

1. **Configure Monitoring**: Set up alerts and monitoring
2. **Add Authentication**: Implement user authentication
3. **Custom Domain**: Configure your custom domain
4. **CI/CD**: Set up continuous deployment
5. **Backup Strategy**: Configure database backups

Your Agent Lobbi backend is now running on Railway! 🎉 