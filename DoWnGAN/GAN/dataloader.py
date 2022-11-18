import torch
from torch.utils.data import Dataset
import xarray as xr
import numpy as np

class NetCDFSR(Dataset):
    """Data loader from torch.Tensors"""
    def __init__(
        self,
        coarse: torch.Tensor,
        fine: torch.Tensor,
        invarient: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Initializes the dataset.
        Returns:
            torch.Tensor: The dataset batches.
        """
        self.fine = fine
        self.coarse = coarse
        self.invarient = invarient

    def __len__(self):
        return self.fine.size(0)

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        fine_ = torch.cat([self.fine[idx, ...],self.invarient],3)
        coarse_ = torch.cat([self.coarse[idx, ...],self.invarient],3)

        return coarse_, fine_
