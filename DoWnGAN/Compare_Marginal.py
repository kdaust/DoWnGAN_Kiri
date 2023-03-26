
import mlflow
import mlflow.pytorch
from xarray.core.dataset import Dataset
import xarray as xr
import netCDF4
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
device = torch.device("cuda:0")


mod_noise = "/media/data/mlflow_exp/4/9625d9c4e7584218827d4ec1740eb7f0/artifacts/Generator/Generator_500"
G = mlflow.pytorch.load_model(mod_noise)

data_folder = "/home/kiridaust/Masters/Data/processed_data/ds_wind/"

cond_fields = xr.open_dataset(data_folder + "coarse_validation.nc", engine="netcdf4")
coarse = torch.from_numpy(cond_fields.to_array().to_numpy()).transpose(0, 1).to(device).float()
invariant = xr.open_dataset(data_folder + "DEM_Crop.nc", engine = "netcdf4")
invariant = torch.from_numpy(invariant.to_array().to_numpy()).to(device).float()

batchsize = 50
invariant = invariant.repeat(batchsize,1,1,1)
# noise_f = torch.normal(0,1,size = [batchsize,1,128,128], device=device)
# invariant = torch.cat([invariant, noise_f], 1)
# print(invariant.size())

sample = 2
coarse_in = coarse[sample,...]
print(coarse_in.size())
coarse_in = coarse_in.unsqueeze(0).repeat(batchsize,1,1,1)

models = ['28374f6f7190495cbf92d698584a2da2','65e9cd4ba68045bdb79526d0196b654e']
modNm = ['CovLR_PFS','Inject_PFS']
#modDat = [coarse_noise, coarse_in]
xp = 64
yp = 64

res = dict()
for i in range(len(modNm)):
    print("Analysing model",modNm[i])
    mod_noise = "/media/data/mlflow_exp/4/" + models[i] +"/artifacts/Generator/Generator_500"
    G = mlflow.pytorch.load_model(mod_noise)
    if(i == 0):
        noise_c = torch.normal(0,1,size = [batchsize, 1, coarse_in.shape[2],coarse_in.shape[3]], device=device)
        curr_coarse = torch.cat([coarse_in, noise_c], 1)
    else:
        curr_coarse = coarse_in
    gen_out = G(curr_coarse,invariant).cpu().detach()
    for i in range(20):
        fine_gen = G(curr_coarse,invariant)
        gen_out = torch.cat([gen_out,fine_gen.cpu().detach()],0)
        del fine_gen
    
    dat1 = gen_out[:,0,xp,yp].flatten()
    res[modNm[i]] = dat1
    del gen_out
    
for nm in modNm:
    sns.kdeplot(res[nm], label = nm)

plt.legend()
plt.savefig("Marginal_Compare.png", dpi = 600)
