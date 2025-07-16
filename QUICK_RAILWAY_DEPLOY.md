# Quick Railway Deployment Guide

## TL;DR - Deploy in 5 Minutes

### Prerequisites
```bash
# Install Railway CLI
npm install -g @railway/cli

# Login to Railway
railway login
```

### Deploy Agent Lobbi Backend

#### Option 1: Automated Script (Recommended)

**Windows:**
```cmd
deploy-to-railway.bat
```

**Linux/Mac:**
```bash
./deploy-to-railway.sh
```

#### Option 2: Manual Commands

```bash
# 1. Initialize project
railway init

# 2. Add PostgreSQL
railway add --service postgres

# 3. Set essential environment variables
railway variables set PORT=8080
railway variables set HOST=0.0.0.0
railway variables set ENV=production
railway variables set LOG_LEVEL=INFO

# 4. Deploy
railway up
```

### Access Your Deployed API

After deployment, your Agent Lobbi backend will be available at:
- **Health Check**: `https://your-app.up.railway.app/api/health`
- **API Docs**: `https://your-app.up.railway.app/docs`
- **Metrics**: `https://your-app.up.railway.app/metrics/dashboard`

### Key Commands

```bash
# View logs
railway logs --follow

# Check status
railway status

# Open dashboard
railway open

# View variables
railway variables
```

## What Gets Deployed

- ✅ **Agent Lobby Core** - Main orchestration system
- ✅ **A2A API Bridge** - Enhanced API with A2A protocol
- ✅ **Metrics Dashboard** - Real-time monitoring
- ✅ **WebSocket Support** - Real-time communication
- ✅ **PostgreSQL Database** - Persistent storage
- ✅ **Health Monitoring** - Auto-restart on failure

## Features Available

- 🔗 **A2A Protocol Support** - Agent-to-Agent communication
- 📊 **Real-time Metrics** - Performance monitoring
- 🔄 **WebSocket API** - Live updates
- 🗄️ **Database Integration** - PostgreSQL with auto-migrations
- 📚 **Auto-generated API Docs** - Interactive documentation
- 🛡️ **Security Features** - JWT authentication ready
- 🎯 **Health Checks** - Railway auto-monitoring

## Production Ready

This deployment is production-ready with:
- Auto-scaling based on traffic
- Health check monitoring
- Database backup and recovery
- HTTPS enabled by default
- Environment variable management
- Logging and monitoring

Ready to deploy? Run the script and you'll have your Agent Lobbi backend live in minutes! 🚀 