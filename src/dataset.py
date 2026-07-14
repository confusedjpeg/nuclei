from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split


def list_pairs(root: Path) -> List[Tuple[Path, Path]]:
    img_dir = root / "images"
    msk_dir = root / "masks"
    pairs = []
    for img_path in sorted(img_dir.glob("*.tif")):
        m = msk_dir / img_path.name
        if m.exists():
            pairs.append((img_path, m))
    return pairs


class NucleiDataset(Dataset):
    def __init__(self, root: str | Path, size: int = 256, binarize: bool = True):
        self.root = Path(root)
        self.size = size
        self.binarize = binarize
        self.pairs = list_pairs(self.root)
        if not self.pairs:
            raise FileNotFoundError(f"No image/mask .tif pairs found in {self.root}/images + masks")

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path, msk_path = self.pairs[idx]

        img = Image.open(img_path).convert("RGB")
        msk = Image.open(msk_path)

        if img.size != (self.size, self.size):
            img = img.resize((self.size, self.size), Image.BILINEAR)
        if msk.size != (self.size, self.size):
            msk = msk.resize((self.size, self.size), Image.NEAREST)

        img = np.array(img, dtype=np.float32) / 255.0
        msk = np.array(msk, dtype=np.float32)

        if self.binarize:
            msk = (msk > 0.5).astype(np.float32)

        img = np.transpose(img, (2, 0, 1))
        msk = np.expand_dims(msk, 0)

        return torch.from_numpy(img), torch.from_numpy(msk)


def make_loaders(
    root: str | Path,
    batch: int = 4,
    workers: int = 0,
    val_split: float = 0.15,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader]:
    ds = NucleiDataset(root)
    n_val = max(1, int(len(ds) * val_split))
    n_train = len(ds) - n_val
    generator = torch.Generator().manual_seed(seed)
    train, val = random_split(ds, [n_train, n_val], generator=generator)
    return (
        DataLoader(train, batch_size=batch, shuffle=True, num_workers=workers),
        DataLoader(val, batch_size=batch, shuffle=False, num_workers=workers),
    )


if __name__ == "__main__":
    dl_train, dl_val = make_loaders(Path(__file__).resolve().parents[1] / "dsb2018" / "train")
    x, y = next(iter(dl_train))
    print(f"Train batches: {len(dl_train)}, Val batches: {len(dl_val)}")
    print(f"Image batch: {x.shape} {x.dtype}  min={x.min().item():.3f} max={x.max().item():.3f}")
    print(f"Mask  batch: {y.shape} {y.dtype}  unique={torch.unique(y).tolist()}")
