FROM rocm/pytorch:rocm6.2_ubuntu22.04_py3.10_pytorch_release_2.3.0

USER root

# GPU monitoring tools
RUN apt-get update && apt-get install -y nvtop rocm-smi

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip wheel setuptools

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt \
    && rm /tmp/requirements.txt