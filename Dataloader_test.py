
import torch
from xarray.core import dataset
from xarray.core.dataset import Dataset
import xarray as xr
import numpy as np
import pandas as pd

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

cudadevice = torch.device("cuda:0")
def load_preprocessed():
    coarse_train = xr.open_dataset(config.PROC_DATA+f"/coarse_train_{config.region}.nc", engine="netcdf4")
    fine_train = xr.open_dataset(config.PROC_DATA+f"/fine_train_{config.region}.nc", engine="netcdf4")
    invariant_train = xr.open_dataset(config.PROC_DATA+f"/fine_train_{config.region}.nc", engine="netcdf4")
    return coarse_train, fine_train, invariant_train

coarse_train, fine_train, invariant_train = load_preprocessed()
print("Loading region into memory...")
coarse_train = torch.from_numpy(coarse_train.to_array().to_numpy()).transpose(0, 1).to(config.device).float()
fine_train = torch.from_numpy(fine_train.to_array().to_numpy()).transpose(0, 1).to(config.device).float()
invariant_train = torch.from_numpy(invariant_train.to_array().to_numpy()).transpose(0, 1).to(config.device).float()
print("Finished loading to memory...")

dataset = NetCDFSR(coarse_train, fine_train, invariant_train, device=cudadevice)
dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=32, shuffle=True)