# Multi-stage build for production Agent Lobby
FROM python:3.11-slim as builder

# Set build arguments
ARG BUILD_DATE
ARG VCS_REF
ARG VERSION

# Add metadata
LABEL maintainer="Agent Lobby Team" \
      org.opencontainers.image.title="Agent Lobby" \
      org.opencontainers.image.description="Production-ready multi-agent collaboration platform" \
      org.opencontainers.image.version=$VERSION \
      org.opencontainers.image.created=$BUILD_DATE \
      org.opencontainers.image.revision=$VCS_REF \
      org.opencontainers.image.source="https://github.com/agent-lobby/agent-lobby"

# Install system dependencies for building
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Production stage
FROM python:3.11-slim as production

# Install runtime dependencies only
RUN apt-get update && apt-get install -y \
    libpq5 \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user for security
RUN groupadd -r agentlobby && useradd -r -g agentlobby agentlobby

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Set working directory
WORKDIR /app

# Copy application code
COPY src/ ./src/
COPY tests/ ./tests/
COPY *.py ./
COPY *.md ./

# Create necessary directories
RUN mkdir -p /app/logs /app/data && \
    chown -R agentlobby:agentlobby /app

# Switch to non-root user
USER agentlobby

# Environment variables
ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8080 \
    WS_PORT=8081 \
    LOG_LEVEL=INFO \
    ENABLE_SYSTEM_METRICS=true \
    HEALTH_CHECK_INTERVAL=30 \
    METRICS_INTERVAL=60

# Expose ports
EXPOSE 8080 8081

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

# Default command
CMD ["python", "-m", "src.main"]

# Development stage
FROM production as development

# Switch back to root for development tools
USER root

# Install development dependencies
RUN pip install --no-cache-dir \
    pytest \
    pytest-asyncio \
    pytest-cov \
    black \
    mypy \
    bandit \
    safety \
    locust

# Install debugging tools
RUN apt-get update && apt-get install -y \
    vim \
    htop \
    strace \
    && rm -rf /var/lib/apt/lists/*

# Switch back to agentlobby user
USER agentlobby

# Override command for development
CMD ["python", "-m", "src.main", "--debug"] 