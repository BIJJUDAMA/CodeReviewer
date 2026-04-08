FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# Create a non-root user
RUN useradd -m -u 1000 user
USER user
WORKDIR /app

# Install dependencies
COPY --chown=user requirements.txt .
RUN pip install --user -r requirements.txt

# Copy application code
COPY --chown=user . .

# HF Spaces expects port 7860
EXPOSE 7860

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
