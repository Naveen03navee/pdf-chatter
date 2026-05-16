# 1. Start with Python
FROM python:3.11-slim

# 2. Set the working directory
WORKDIR /app

# 3. Copy the requirements file
COPY requirements.txt .

# 4. Use Docker BuildKit to cache the downloads like a save-point!
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --default-timeout=1000 --retries=10 -r requirements.txt

# 5. Copy the rest of your code
COPY . .

# 6. Expose the port
EXPOSE 10000

# 7. Start the server
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "10000"]