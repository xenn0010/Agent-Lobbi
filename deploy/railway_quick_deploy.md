# 🚀 Railway.app Quick Deployment - $25/month

## 🎯 **Deploy Agent Lobbi to Railway in 5 Minutes**

### **Step 1: Prepare Your Repository**
Your code is already ready! Railway will automatically detect:
- ✅ `railway.json` configuration
- ✅ `requirements.txt` with all dependencies
- ✅ `src/main.py` as the entry point

### **Step 2: Deploy to Railway**

1. **Go to [railway.app](https://railway.app)**
2. **Sign up/Login** with your GitHub account
3. **Click "New Project"**
4. **Select "Deploy from GitHub repo"**
5. **Choose your Agent_lobby repository**
6. **Click "Deploy"**

### **Step 3: Add PostgreSQL Database**

1. **In your Railway project dashboard**
2. **Click "New" → "Database" → "PostgreSQL"**
3. **Wait for database to provision**
4. **Copy the database URL** (Railway will provide this)

### **Step 4: Configure Environment Variables**

In your Railway project dashboard, add these variables:

```env
PYTHONPATH=/app/src
LOBBY_HTTP_PORT=8080
LOBBY_WS_PORT=8081
ENVIRONMENT=production
DATABASE_URL=postgresql://...  # Railway will provide this
```

### **Step 5: Update Vercel Configuration**

In your Vercel dashboard, update environment variables:

```env
LOBBY_API_BRIDGE_URL=https://your-railway-app.railway.app
LOBBY_HTTP_URL=https://your-railway-app.railway.app
LOBBY_WS_URL=wss://your-railway-app.railway.app/ws
```

### **Step 6: Test Your Deployment**

Your Agent Lobbi will be live at:
- **HTTP API**: `https://your-app.railway.app`
- **Health Check**: `https://your-app.railway.app/health`
- **WebSocket**: `wss://your-app.railway.app/ws`

---

## 💰 **Cost Breakdown**

| Component | Cost | What's Included |
|-----------|------|-----------------|
| **App Hosting** | $20/month | Unlimited usage, 24/7 uptime |
| **PostgreSQL** | $5/month | Managed database with backups |
| **Total** | **$25/month** | Complete production setup |

---

## 🎯 **What You Get**

✅ **24/7 Uptime** - No sleeping like Render/Heroku  
✅ **Automatic Deployments** - Push to GitHub = auto-deploy  
✅ **SSL Certificates** - HTTPS included  
✅ **Custom Domains** - Point agentlobbi.com to it  
✅ **Database Backups** - Automatic PostgreSQL backups  
✅ **Monitoring** - Built-in performance monitoring  
✅ **Scaling** - Auto-scale based on traffic  

---

## 🚀 **Ready to Deploy?**

**Click here to start:** [railway.app](https://railway.app)

Your Agent Lobbi will be live in under 5 minutes for just $25/month! 