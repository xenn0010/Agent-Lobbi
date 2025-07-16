@echo off
setlocal enabledelayedexpansion

REM Agent Lobbi Backend - Railway Deployment Script (Windows)
REM This script automates the deployment of Agent Lobbi backend to Railway

echo 🚀 Agent Lobbi Backend - Railway Deployment Script (Windows)
echo ===========================================================

REM Check if Railway CLI is installed
railway --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Railway CLI is not installed. Please install it first:
    echo    npm install -g @railway/cli
    exit /b 1
)

echo ✅ Railway CLI found

REM Check if user is logged in
railway whoami >nul 2>&1
if %errorlevel% neq 0 (
    echo 🔐 Please login to Railway first...
    railway login
)

echo ✅ Railway authentication verified

REM Check if we're in a Railway project
railway status >nul 2>&1
if %errorlevel% neq 0 (
    echo 📋 Initializing Railway project...
    railway init
    echo ✅ Railway project initialized
) else (
    echo ✅ Railway project detected
)

REM Set environment variables
echo 🔧 Setting up environment variables...

REM Generate secure secrets (simplified for Windows)
set SECRET_KEY=%RANDOM%%RANDOM%%RANDOM%
set JWT_SECRET_KEY=%RANDOM%%RANDOM%%RANDOM%

REM Core configuration
railway variables set PORT=8080
railway variables set HOST=0.0.0.0
railway variables set LOBBY_HTTP_PORT=8080
railway variables set LOBBY_WS_PORT=8081

REM Environment settings
railway variables set ENV=production
railway variables set LOG_LEVEL=INFO
railway variables set DEBUG=false

REM Security
railway variables set SECRET_KEY="%SECRET_KEY%"
railway variables set JWT_SECRET_KEY="%JWT_SECRET_KEY%"
railway variables set JWT_ALGORITHM=HS256
railway variables set JWT_EXPIRE_HOURS=24

REM CORS Configuration
railway variables set CORS_ORIGINS="*"
railway variables set CORS_CREDENTIALS=true
railway variables set CORS_METHODS="GET,POST,PUT,DELETE,OPTIONS"
railway variables set CORS_HEADERS="*"

REM API Configuration
railway variables set API_VERSION=v1
railway variables set MAX_WORKERS=1

REM Monitoring
railway variables set ENABLE_METRICS=true
railway variables set HEALTH_CHECK_ENABLED=true

REM Agent Configuration
railway variables set MAX_AGENTS=100
railway variables set DEFAULT_TIMEOUT=30
railway variables set MAX_TASK_DURATION=300

REM Feature Flags
railway variables set ENABLE_A2A_BRIDGE=true
railway variables set ENABLE_WEBSOCKETS=true
railway variables set ENABLE_REAL_TIME_METRICS=true

echo ✅ Environment variables configured

REM Check for database services
echo 🗄️ Checking for database services...

REM Add PostgreSQL
echo 📊 Adding PostgreSQL database...
railway add --service postgres

REM Ask about Redis
set /p add_redis="🗄️ Do you want to add Redis cache? (y/N): "
if /i "%add_redis%"=="y" (
    echo 📊 Adding Redis cache...
    railway add --service redis
    echo ✅ Redis cache added
)

REM Deploy the application
echo 🚀 Deploying to Railway...
railway up --detach

echo ⏳ Waiting for deployment to complete...
timeout /t 30 /nobreak >nul

REM Get deployment URL
for /f "tokens=*" %%i in ('railway domain 2^>nul') do set DEPLOYMENT_URL=%%i

if "!DEPLOYMENT_URL!"=="" (
    echo 🌐 Generating deployment URL...
    railway domain generate
    for /f "tokens=*" %%i in ('railway domain') do set DEPLOYMENT_URL=%%i
)

echo ✅ Deployment completed!
echo.
echo 🎉 Your Agent Lobbi backend is now deployed!
echo ==================================================
echo 🌐 URL: https://!DEPLOYMENT_URL!
echo.
echo 📋 Available Endpoints:
echo    Health Check:    https://!DEPLOYMENT_URL!/api/health
echo    API Docs:        https://!DEPLOYMENT_URL!/docs
echo    Metrics:         https://!DEPLOYMENT_URL!/metrics/dashboard
echo    A2A Discovery:   https://!DEPLOYMENT_URL!/.well-known/agent.json
echo.
echo 🔧 Management Commands:
echo    View logs:       railway logs
echo    Check status:    railway status
echo    Open dashboard:  railway open
echo    View variables:  railway variables
echo.

REM Test the deployment
echo 🧪 Testing deployment...
timeout /t 10 /nobreak >nul

echo    Testing health endpoint...
curl -s -f "https://!DEPLOYMENT_URL!/api/health" >nul 2>&1
if %errorlevel% equ 0 (
    echo    ✅ Health check passed
) else (
    echo    ⚠️ Health check failed (service may still be starting)
)

echo.
echo 🎊 Deployment script completed!
echo.
echo 📖 Next Steps:
echo    1. Visit https://!DEPLOYMENT_URL!/docs to explore the API
echo    2. Check https://!DEPLOYMENT_URL!/metrics/dashboard for monitoring
echo    3. Configure custom domain in Railway dashboard (optional)
echo    4. Set up monitoring alerts
echo    5. Implement authentication for production use
echo.
echo 💡 Tip: Run 'railway logs --follow' to monitor your application

REM Save deployment info
echo Agent Lobbi Backend - Railway Deployment Information > deployment-info.txt
echo ==================================================== >> deployment-info.txt
echo. >> deployment-info.txt
echo Deployment URL: https://!DEPLOYMENT_URL! >> deployment-info.txt
echo Deployment Date: %date% %time% >> deployment-info.txt
echo. >> deployment-info.txt
echo Key Endpoints: >> deployment-info.txt
echo - Health Check: https://!DEPLOYMENT_URL!/api/health >> deployment-info.txt
echo - API Documentation: https://!DEPLOYMENT_URL!/docs >> deployment-info.txt
echo - Metrics Dashboard: https://!DEPLOYMENT_URL!/metrics/dashboard >> deployment-info.txt
echo - A2A Discovery: https://!DEPLOYMENT_URL!/.well-known/agent.json >> deployment-info.txt
echo. >> deployment-info.txt
echo Generated Secrets: >> deployment-info.txt
echo - SECRET_KEY: !SECRET_KEY! >> deployment-info.txt
echo - JWT_SECRET_KEY: !JWT_SECRET_KEY! >> deployment-info.txt
echo. >> deployment-info.txt
echo Note: Keep this file secure and do not commit to version control. >> deployment-info.txt

echo 📄 Deployment information saved to deployment-info.txt

pause 