import torch
import os
import argparse
import yaml

def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str)
    args = parser.parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)
    return config

def set_dir(path):
    if os.path.exists(path):
        return
    else:
        os.makedirs(path)
        return

def get_device(config):
    requested = str(config["global"].get("device", "auto")).lower()

    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Config requested CUDA, but no CUDA GPU is available.")
        return torch.device("cuda")

    if requested == "cpu":
        return torch.device("cpu")

    raise ValueError(f"Unknown device option: {requested}")
