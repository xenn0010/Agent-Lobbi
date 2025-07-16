#!/bin/bash

# Agent Lobbi Backend - Railway Deployment Script
# This script automates the deployment of Agent Lobbi backend to Railway

set -e  # Exit on any error

echo "🚀 Agent Lobbi Backend - Railway Deployment Script"
echo "=================================================="

# Check if Railway CLI is installed
if ! command -v railway &> /dev/null; then
    echo "❌ Railway CLI is not installed. Please install it first:"
    echo "   npm install -g @railway/cli"
    echo "   Or: curl -fsSL https://railway.app/install.sh | sh"
    exit 1
fi

echo "✅ Railway CLI found"

# Check if user is logged in
if ! railway whoami &> /dev/null; then
    echo "🔐 Please login to Railway first..."
    railway login
fi

echo "✅ Railway authentication verified"

# Function to generate secure random string
generate_secret() {
    openssl rand -base64 32 | tr -d "=+/" | cut -c1-25
}

# Check if we're in a Railway project
if ! railway status &> /dev/null; then
    echo "📋 Initializing Railway project..."
    railway init
    echo "✅ Railway project initialized"
else
    echo "✅ Railway project detected"
fi

# Set environment variables
echo "🔧 Setting up environment variables..."

# Generate secure secrets
SECRET_KEY=$(generate_secret)
JWT_SECRET_KEY=$(generate_secret)

# Core configuration
railway variables set PORT=8080
railway variables set HOST=0.0.0.0
railway variables set LOBBY_HTTP_PORT=8080
railway variables set LOBBY_WS_PORT=8081

# Environment settings
railway variables set ENV=production
railway variables set LOG_LEVEL=INFO
railway variables set DEBUG=false

# Security
railway variables set SECRET_KEY="$SECRET_KEY"
railway variables set JWT_SECRET_KEY="$JWT_SECRET_KEY"
railway variables set JWT_ALGORITHM=HS256
railway variables set JWT_EXPIRE_HOURS=24

# CORS Configuration
railway variables set CORS_ORIGINS="*"
railway variables set CORS_CREDENTIALS=true
railway variables set CORS_METHODS="GET,POST,PUT,DELETE,OPTIONS"
railway variables set CORS_HEADERS="*"

# API Configuration
railway variables set API_VERSION=v1
railway variables set MAX_WORKERS=1

# Monitoring
railway variables set ENABLE_METRICS=true
railway variables set HEALTH_CHECK_ENABLED=true

# Agent Configuration
railway variables set MAX_AGENTS=100
railway variables set DEFAULT_TIMEOUT=30
railway variables set MAX_TASK_DURATION=300

# Feature Flags
railway variables set ENABLE_A2A_BRIDGE=true
railway variables set ENABLE_WEBSOCKETS=true
railway variables set ENABLE_REAL_TIME_METRICS=true

echo "✅ Environment variables configured"

# Check for database services
echo "🗄️  Checking for database services..."

# Add PostgreSQL if not exists
if ! railway service list | grep -q "postgres"; then
    echo "📊 Adding PostgreSQL database..."
    railway add --service postgres
    echo "✅ PostgreSQL database added"
else
    echo "✅ PostgreSQL database found"
fi

# Add Redis if not exists (optional)
read -p "🗄️  Do you want to add Redis cache? (y/N): " add_redis
if [[ $add_redis =~ ^[Yy]$ ]]; then
    if ! railway service list | grep -q "redis"; then
        echo "📊 Adding Redis cache..."
        railway add --service redis
        echo "✅ Redis cache added"
    else
        echo "✅ Redis cache found"
    fi
fi

# Deploy the application
echo "🚀 Deploying to Railway..."
railway up --detach

echo "⏳ Waiting for deployment to complete..."
sleep 30

# Get the deployment URL
DEPLOYMENT_URL=$(railway domain 2>/dev/null || echo "")

if [ -z "$DEPLOYMENT_URL" ]; then
    echo "🌐 Getting deployment URL..."
    # Generate a domain if none exists
    railway domain generate
    DEPLOYMENT_URL=$(railway domain)
fi

echo "✅ Deployment completed!"
echo ""
echo "🎉 Your Agent Lobbi backend is now deployed!"
echo "=================================================="
echo "🌐 URL: https://$DEPLOYMENT_URL"
echo ""
echo "📋 Available Endpoints:"
echo "   Health Check:    https://$DEPLOYMENT_URL/api/health"
echo "   API Docs:        https://$DEPLOYMENT_URL/docs"
echo "   Metrics:         https://$DEPLOYMENT_URL/metrics/dashboard"
echo "   A2A Discovery:   https://$DEPLOYMENT_URL/.well-known/agent.json"
echo ""
echo "🔧 Management Commands:"
echo "   View logs:       railway logs"
echo "   Check status:    railway status"
echo "   Open dashboard:  railway open"
echo "   View variables:  railway variables"
echo ""

# Test the deployment
echo "🧪 Testing deployment..."

# Wait a bit more for the service to be fully ready
sleep 10

# Test health endpoint
echo "   Testing health endpoint..."
if curl -s -f "https://$DEPLOYMENT_URL/api/health" > /dev/null; then
    echo "   ✅ Health check passed"
else
    echo "   ⚠️  Health check failed (service may still be starting)"
fi

# Test A2A discovery endpoint
echo "   Testing A2A discovery endpoint..."
if curl -s -f "https://$DEPLOYMENT_URL/.well-known/agent.json" > /dev/null; then
    echo "   ✅ A2A discovery endpoint working"
else
    echo "   ⚠️  A2A discovery endpoint not ready yet"
fi

echo ""
echo "🎊 Deployment script completed!"
echo ""
echo "📖 Next Steps:"
echo "   1. Visit https://$DEPLOYMENT_URL/docs to explore the API"
echo "   2. Check https://$DEPLOYMENT_URL/metrics/dashboard for monitoring"
echo "   3. Configure custom domain in Railway dashboard (optional)"
echo "   4. Set up monitoring alerts"
echo "   5. Implement authentication for production use"
echo ""
echo "💡 Tip: Run 'railway logs --follow' to monitor your application"

# Save deployment info
cat > deployment-info.txt << EOF
Agent Lobbi Backend - Railway Deployment Information
====================================================

Deployment URL: https://$DEPLOYMENT_URL
Deployment Date: $(date)

Key Endpoints:
- Health Check: https://$DEPLOYMENT_URL/api/health
- API Documentation: https://$DEPLOYMENT_URL/docs
- Metrics Dashboard: https://$DEPLOYMENT_URL/metrics/dashboard
- A2A Discovery: https://$DEPLOYMENT_URL/.well-known/agent.json

Management Commands:
- View logs: railway logs
- Check status: railway status
- Open dashboard: railway open
- SSH access: railway shell

Environment Variables Set:
- All core configuration variables
- Security keys (auto-generated)
- Database and cache configuration
- API and monitoring settings

Services:
- PostgreSQL database
- Redis cache (if selected)
- Web service

Generated Secrets:
- SECRET_KEY: $SECRET_KEY
- JWT_SECRET_KEY: $JWT_SECRET_KEY

Note: Keep this file secure and do not commit to version control.
EOF

echo "📄 Deployment information saved to deployment-info.txt" 