FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    anthropic \
    streamlit \
    requests \
    beautifulsoup4 \
    pandas \
    numpy \
    scikit-learn \
    lightgbm \
    matplotlib \
    seaborn \
    python-dotenv \
    plotly \
    openai

# Copy application code
COPY . .

# Expose Streamlit port
EXPOSE 8501

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Run Streamlit app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]