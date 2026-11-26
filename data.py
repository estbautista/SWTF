import os
import glob
import torch
from torch.utils.data import Dataset

class WindowDataset(Dataset):
    def __init__(self, folder):
        self.files = sorted(glob.glob(os.path.join(folder, "window_*.pt")))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        sample = torch.load(self.files[idx], map_location="cpu")
        return sample

def collate_single(batch):
    return batch[0]


def move_sample_to_device(sample, device):
    out = {}
    for k, v in sample.items():
        if torch.is_tensor(v):
            out[k] = v.to(device)
        else:
            out[k] = v
    return out
