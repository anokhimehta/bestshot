import os
from configs.gpu_batch import CONFIG as GPU_BATCH
from configs.cpu_batch import CONFIG as CPU_BATCH
from configs.cpu_sequential import CONFIG as CPU_SEQ
from configs.gpu_onnx import CONFIG as GPU_ONNX 

CONFIG_NAME = os.getenv("CONFIG_NAME", "cpu_sequential")

if CONFIG_NAME == "gpu_batch":
    CONFIG = GPU_BATCH
elif CONFIG_NAME == "cpu_batch":
    CONFIG = CPU_BATCH
elif CONFIG_NAME == "cpu_sequential":
    CONFIG = CPU_SEQ
elif CONFIG_NAME == "gpu_onnx":
    CONFIG = GPU_ONNX

else:
    raise ValueError(f"Unknown config: {CONFIG_NAME}")