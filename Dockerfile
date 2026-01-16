FROM pytorch/pytorch:2.2.1-cuda12.1-cudnn8-runtime

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1

RUN apt-get update  -y 
RUN DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt-get install \
    libgl1-mesa-glx libglib2.0-0 libtiff5-dev libjpeg8-dev \
    libopenjp2-7-dev zlib1g-dev libfreetype6-dev liblcms2-dev \
    libwebp-dev tcl8.6-dev tk8.6-dev python3-tk libharfbuzz-dev \
    libfribidi-dev libxcb1-dev git -y && \
    rm -rf /var/lib/apt/lists/*
RUN python -m pip install --upgrade pip

ENV LD_LIBRARY_PATH=/opt/conda/lib/python3.10/site-packages/tensorrt_libs:${LD_LIBRARY_PATH}

COPY requirements.txt requirements.txt
COPY src/app.py app.py 
COPY config.yaml config.yaml

RUN python -m pip install --no-cache-dir -r requirements.txt
RUN python -m pip install --no-cache-dir Pillow-SIMD
# RUN apt-get remove -y python3-opencv 
# RUN pip uninstall -y opencv opencv-python opencv-python-headless opencv-contrib-python
# RUN pip install --no-cache-dir opencv-python-headless==4.8.0.76

RUN python -m pip list

EXPOSE 8040
CMD ["python","app.py"]
