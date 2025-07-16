# Railway Environment Variables Configuration

## Required Environment Variables for Agent Lobbi Backend

When deploying to Railway, you'll need to set these environment variables in your Railway project dashboard:

### Server Configuration
```
PORT=8080
LOBBY_HTTP_PORT=8080
LOBBY_WS_PORT=8081
HOST=0.0.0.0
```

### Environment
```
ENV=production
LOG_LEVEL=INFO
DEBUG=false
```

### Database Configuration
Railway will automatically provide `DATABASE_URL` when you add a PostgreSQL service.
```
DATABASE_URL=${POSTGRES_DATABASE_URL}  # Auto-provided by Railway
```

### Redis Configuration (Optional)
Railway will automatically provide `REDIS_URL` when you add a Redis service.
```
REDIS_URL=${REDIS_URL}  # Auto-provided by Railway
```

### Security
```
SECRET_KEY=your-secret-key-here
JWT_SECRET_KEY=your-jwt-secret-here
JWT_ALGORITHM=HS256
JWT_EXPIRE_HOURS=24
```

### CORS Configuration
```
CORS_ORIGINS=*
CORS_CREDENTIALS=true
CORS_METHODS=GET,POST,PUT,DELETE,OPTIONS
CORS_HEADERS=*
```

### API Configuration
```
API_BASE_URL=https://your-app.up.railway.app
API_VERSION=v1
MAX_WORKERS=4
```

### Monitoring
```
ENABLE_METRICS=true
METRICS_PORT=9090
HEALTH_CHECK_ENABLED=true
```

### Agent Configuration
```
MAX_AGENTS=100
DEFAULT_TIMEOUT=30
MAX_TASK_DURATION=300
```

### Feature Flags
```
ENABLE_A2A_BRIDGE=true
ENABLE_WEBSOCKETS=true
ENABLE_REAL_TIME_METRICS=true
```

## Railway Deployment Steps

1. **Create Railway Project**
   ```bash
   npm install -g @railway/cli
   railway login
   railway init
   ```

2. **Add PostgreSQL Service**
   - Go to your Railway dashboard
   - Click "Add Service" → "Database" → "PostgreSQL"
   - Railway will automatically set `DATABASE_URL`

3. **Add Redis Service (Optional)**
   - Click "Add Service" → "Database" → "Redis"
   - Railway will automatically set `REDIS_URL`

4. **Set Environment Variables**
   - Go to your service settings in Railway dashboard
   - Add all the environment variables listed above
   - Generate secure values for SECRET_KEY and JWT_SECRET_KEY

5. **Deploy**
   ```bash
   railway up
   ```

The application will be available at your Railway-provided URL with all services running. 