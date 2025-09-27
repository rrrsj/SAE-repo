import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from datasets import load_dataset
from huggingface_hub import snapshot_download


dataset = snapshot_download(
    "Skylion007/openwebtext",
    revision="refs/convert/parquet",
    repo_type="dataset",
    local_dir="./dataset")
