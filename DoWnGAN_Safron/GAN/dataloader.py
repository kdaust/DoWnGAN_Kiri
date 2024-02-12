import torch
from torch.utils.data import Dataset
import xarray as xr
import numpy as np

class train_dataloader(Dataset):
    """Data loader from torch.Tensors"""
    def __init__(
        self,
        coarse_list: list,
        invariant_list: list,
        coarse_full: torch.Tensor,
        invariant_full: torch.Tensor,
        fine_full: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Initializes the dataset.
        Returns:
            torch.Tensor: The dataset batches.
        """
        self.coarse_list = coarse_list
        self.invariant_list = invariant_list
        self.coarse_full = coarse_full
        self.invariant_full = invariant_full
        self.fine_full = fine_full

    def __len__(self):
        return self.fine_full.size(0)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        coarse_res = [x[idx,...] for x in self.coarse_list]
        fine_ = self.fine_full[idx,...]
        coarse_ = self.coarse_full[idx,...]
        invariant_ = self.invariant_full
        
        return coarse_res, self.invariant_list, coarse_, fine_, invariant_
