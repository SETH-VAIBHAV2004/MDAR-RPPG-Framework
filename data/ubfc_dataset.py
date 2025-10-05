import os
import os.path as osp
import glob
from typing import List, Tuple, Dict

import numpy as np
import torch
from torch.utils.data import Dataset


def list_subjects(processed_root: str) -> List[str]:
    subs = []
    for p in sorted(glob.glob(osp.join(processed_root, '*'))):
        if osp.isdir(p):
            subs.append(osp.basename(p))
    return subs


def split_subjects(subjects: List[str], train_ratio: float = 0.8, seed: int = 42) -> Tuple[List[str], List[str]]:
    rng = np.random.default_rng(seed)
    idxs = np.arange(len(subjects))
    rng.shuffle(idxs)
    k = int(len(subjects) * train_ratio)
    train = [subjects[i] for i in idxs[:k]]
    val = [subjects[i] for i in idxs[k:]]
    return train, val


class UBFCChunks(Dataset):
    def __init__(self, processed_root: str, subjects: List[str]):
        self.samples: List[Tuple[str, str]] = []
        for s in subjects:
            feat_dir = osp.join(processed_root, s, 'features')
            if not osp.isdir(feat_dir):
                continue
            for npz in sorted(glob.glob(osp.join(feat_dir, 'chunk_*.npz'))):
                self.samples.append((s, npz))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        subject, npz_path = self.samples[idx]
        data = np.load(npz_path, allow_pickle=True)
        feats = data['features'].astype(np.float32)  # (T, C)
        # label waveform: prefer 'gt' if present; otherwise zeros
        gt = data['gt'] if 'gt' in data.files and data['gt'] is not None else None
        if gt is None:
            gt = np.zeros((feats.shape[0],), dtype=np.float32)
        else:
            gt = gt.astype(np.float32)
        return {
            'x': torch.from_numpy(feats),    # (T, C)
            'y': torch.from_numpy(gt),       # (T,)
            'subject': subject,
            'path': npz_path,
        }


