
import torch
import xarray as xr
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

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
        #print(self.fine.size())
        #print(self.coarse.size())
        #print(self.invarient.size())
        #print(self.fine[idx,...].size())

        fine_ = torch.cat([self.fine[idx, ...],self.invarient],0)
        coarse_ = torch.cat([self.coarse[idx, ...],self.invarient],0)
        # print(self.fine[idx,...])
        # print(self.invarient)
        # print(fine_)
        return coarse_, fine_

cudadevice = torch.device("cuda:0")
def load_preprocessed():
    coarse_train = xr.open_dataset("~/Masters/Data/Test_Upsample/coarse_small.nc", engine="netcdf4")
    fine_train = xr.open_dataset("~/Masters/Data/Test_Upsample/fine_small.nc", engine="netcdf4")
    invariant_train = xr.open_dataset("~/Masters/Data/Test_Upsample/DEM_Final_out.nc", engine="netcdf4")
    return coarse_train, fine_train, invariant_train

coarse_train, fine_train, invariant_train = load_preprocessed()
print("Loading region into memory...")
coarse_train = torch.from_numpy(coarse_train.to_array().to_numpy()).transpose(0, 1).to(cudadevice).float()
print(coarse_train)
fine_train = torch.from_numpy(fine_train.to_array().to_numpy()).transpose(0, 1).to(cudadevice).float()
invariant_train = torch.from_numpy(invariant_train.to_array().to_numpy().squeeze(0)).to(cudadevice).float()
print("Finished loading to memory...")

dataset = NetCDFSR(coarse_train, fine_train, invariant_train, device=cudadevice)
print(dataset[5])
dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=32, shuffle=True)
