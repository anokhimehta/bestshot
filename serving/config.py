import os

# Load the default config based on environment variable
CONFIG_NAME = os.getenv("CONFIG_NAME", "cpu_sequential")

if CONFIG_NAME == "cpu_sequential": 
    from configs.cpu_sequential import CONFIG
elif CONFIG_NAME == "cpu_batch": 
    from configs.cpu_batch import CONFIG
elif CONFIG_NAME == "gpu_batch":
    from configs.gpu_batch import CONFIG
else:
    raise ValueError(f"Unknown config: {CONFIG_NAME}")