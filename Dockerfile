FROM python:3.10-slim

EXPOSE 8501

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    software-properties-common \
    git \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /main

COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

COPY . /main/

# Explicitly set the LD_LIBRARY_PATH (though usually not required in standard setups)
# ENV LD_LIBRARY_PATH /usr/lib/x86_64-linux-gnu:/usr/local/lib

ENTRYPOINT ["streamlit", "run", "main.py", "--server.port=8501", "--server.address=0.0.0.0"]