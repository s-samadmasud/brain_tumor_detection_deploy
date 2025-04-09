FROM python:3.8-slim-buster

EXPOSE 8501

RUN apt-get update && apt-get install -y \
    build-essential \
    software-properties-common \
    git \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /main

# Copy the requirements file FIRST
COPY requirements.txt .

# Install dependencies BEFORE copying the rest of the project
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy the rest of the project files
COPY . /main/

ENTRYPOINT ["streamlit", "run", "main.py", "--server.port=8501", "--server.address=0.0.0.0"]