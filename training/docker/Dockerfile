FROM quay.io/jupyter/pytorch-notebook:cuda12-python-3.12

USER root
# GPU monitoring
RUN apt-get update && apt-get install -y nvtop

USER ${NB_UID}
RUN python -m pip install --no-cache-dir --upgrade pip wheel setuptools

COPY requirements.txt /tmp/requirements.txt
RUN python -m pip install --no-cache-dir -r /tmp/requirements.txt \ 
    && rm /tmp/requirements.txt \
    && fix-permissions "/home/${NB_USER}" \
    && fix-permissions "/${CONDA_DIR}" 