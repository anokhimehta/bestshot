FROM rocm/pytorch:rocm6.2_ubuntu22.04_py3.10_pytorch_release_2.3.0

USER root

# GPU monitoring tools
RUN apt-get update && apt-get install -y nvtop

RUN apt-get update && apt-get install -y curl ca-certificates && \
    curl -fsSL -o /usr/local/bin/kubectl https://dl.k8s.io/release/v1.30.0/bin/linux/amd64/kubectl && \
    chmod +x /usr/local/bin/kubectl
    
# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip wheel setuptools

COPY training/docker/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt \
    && rm /tmp/requirements.txt

COPY training/ /app/training/
COPY data/batch_pipeline/ /app/data/batch_pipeline/
WORKDIR /app
