FROM nvcr.io/nvidia/tensorrt:25.01-py3

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1

WORKDIR /workspace

RUN apt-get update -y \
    && DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt-get install -y \
       libgl1 libglib2.0-0 libtiff-dev libjpeg-dev libopenjp2-7-dev \
       zlib1g-dev libfreetype6-dev liblcms2-dev libwebp-dev \
       libharfbuzz-dev libfribidi-dev libxcb1-dev git \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade pip

COPY requirements.txt requirements.txt
COPY app.py app.py 
COPY config.yaml config.yaml

RUN python -m pip install -r requirements.txt

CMD ["python","app.py"]
