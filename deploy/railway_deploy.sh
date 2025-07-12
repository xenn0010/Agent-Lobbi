#!/bin/bash

# Agent Lobbi - Railway Deployment Script
# This script helps you deploy to Railway.app for $25/month

echo "🚀 Agent Lobbi Railway Deployment"
echo "=================================="
echo ""

# Check if Railway CLI is installed
if ! command -v railway &> /dev/null; then
    echo "📦 Installing Railway CLI..."
    npm install -g @railway/cli
fi

echo "🔐 Logging into Railway..."
railway login

echo "📁 Creating new Railway project..."
railway init

echo "🔧 Setting up environment variables..."
railway variables set PYTHONPATH=/app/src
railway variables set LOBBY_HTTP_PORT=8080
railway variables set LOBBY_WS_PORT=8081
railway variables set ENVIRONMENT=production

echo "🗄️ Adding PostgreSQL database..."
railway add

echo "🚀 Deploying to Railway..."
railway up

echo ""
echo "✅ Deployment complete!"
echo ""
echo "🌐 Your app is now live at:"
railway domain
echo ""
echo "📊 Monitor your deployment:"
echo "railway logs"
echo ""
echo "💰 Cost: $25/month (includes database)"
echo ""
echo "🔗 Next steps:"
echo "1. Update your Vercel environment variables"
echo "2. Test the connection"
echo "3. Set up custom domain (optional)" 