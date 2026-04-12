FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# Create a non-root user
RUN useradd -m -u 1000 user
WORKDIR /app

# Install dependencies from server folder
COPY --chown=user server/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the server directory contents to /app
COPY --chown=user server/ .

# Ensure the user has permissions
RUN chown -R user:user /app

USER user

# HF Spaces expects port 7860
EXPOSE 7860

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
