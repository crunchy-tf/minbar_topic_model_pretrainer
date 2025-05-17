# Use the same base Python image as your nlp_analyzer_service for best compatibility
FROM python:3.10-slim-bookworm

# Set environment variables to make Python print out stuff immediately
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Set the working directory in the container
WORKDIR /pretrainer

# Install system dependencies needed for building some Python packages
# (e.g., wget for downloader, gcc/python3-dev for hdbscan which bertopic uses)
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    gcc \
    python3-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create cache directories and set permissions BEFORE changing user
RUN mkdir -p /tmp/huggingface_cache /tmp/numba_cache && \
    chown -R nobody:nogroup /tmp/huggingface_cache /tmp/numba_cache && \
    chmod -R 777 /tmp/huggingface_cache /tmp/numba_cache

# Set environment variables for cache locations
ENV TRANSFORMERS_CACHE=/tmp/huggingface_cache
ENV HF_HOME=/tmp/huggingface_cache
ENV NUMBA_CACHE_DIR=/tmp/numba_cache

# Copy the requirements file first to leverage Docker layer caching
COPY ./requirements_pretrainer.txt /pretrainer/requirements_pretrainer.txt

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r /pretrainer/requirements_pretrainer.txt

# Copy the rest of your application code into the container
COPY ./filter_keywords /pretrainer/filter_keywords
COPY ./bertopic_trainer.py /pretrainer/bertopic_trainer.py
COPY ./config_pretrainer.py /pretrainer/config_pretrainer.py
COPY ./main_pretrainer.py /pretrainer/main_pretrainer.py
COPY ./wikipedia_downloader.py /pretrainer/wikipedia_downloader.py
COPY ./wikipedia_filterer.py /pretrainer/wikipedia_filterer.py
COPY ./wikipedia_parser.py /pretrainer/wikipedia_parser.py

# The WORKING_DIR inside config_pretrainer.py points to "pretrainer_workspace"
# relative to the script's location. We need to ensure this directory exists
# and that the container has permissions to write to it.
# The pretrainer_workspace will be created inside the WORKDIR (/pretrainer)
# So, it will be /pretrainer/pretrainer_workspace
RUN mkdir -p /pretrainer/pretrainer_workspace && \
    chown -R nobody:nogroup /pretrainer/pretrainer_workspace && \
    chmod -R 777 /pretrainer/pretrainer_workspace

# Switch to non-root user
USER nobody:nogroup

# When the container runs, it will execute the main pretrainer pipeline
ENTRYPOINT ["python", "main_pretrainer.py"]