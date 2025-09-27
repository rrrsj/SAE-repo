import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from datasets import load_dataset
from huggingface_hub import snapshot_download


dataset = snapshot_download(
    "Qwen/Qwen2-0.5B-Instruct",
    local_dir="./model_checkpoint")
