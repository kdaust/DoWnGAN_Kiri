
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


#mod_noise = "/media/data/mlflow_exp/4/9625d9c4e7584218827d4ec1740eb7f0/artifacts/Generator/Generator_500"
#G = mlflow.pytorch.load_model(mod_noise)

data_folder = "/home/kiridaust/Masters/Data/ToyDataSet/"

# cond_fields = xr.open_dataset(data_folder + "coarse_validation.nc", engine="netcdf4")
# coarse = torch.from_numpy(cond_fields.to_array().to_numpy()).transpose(0, 1).to(device).float()
# invariant = xr.open_dataset(data_folder + "DEM_Crop.nc", engine = "netcdf4")
# invariant = torch.from_numpy(invariant.to_array().to_numpy()).to(device).float()
coarse = np.load(data_folder+"coarse_val_toydat.npy")
coarse = np.swapaxes(coarse, 0, 2)
coarse = torch.from_numpy(coarse)[:,None,...].to(device).float()
fine = np.load(data_folder+"fine_val_toydat.npy")
fine = np.swapaxes(fine, 0, 2)
fine = torch.from_numpy(fine)[:,None,...]

batchsize = 32
#invariant = invariant.repeat(batchsize,1,1,1)
# noise_f = torch.normal(0,1,size = [batchsize,1,128,128], device=device)
# invariant = torch.cat([invariant, noise_f], 1)
# print(invariant.size())

sample = 0
coarse_in = coarse[sample,...]
print(coarse_in.size())
coarse_in = coarse_in.unsqueeze(0).repeat(batchsize,1,1,1)

models = ['844ed50e4d0e4fa68554c4b4cd15b224','ccfb1a914fcd43c58aae2ef2c27a54ef']
modNm = ['CovBoth_PFS','Inject_PFS']
#modDat = [coarse_noise, coarse_in]
xp = 54
yp = 120

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
    gen_out = G(curr_coarse).cpu().detach()
    for j in range(40):
        fine_gen = G(curr_coarse)
        gen_out = torch.cat([gen_out,fine_gen.cpu().detach()],0)
        del fine_gen
    
    dat1 = gen_out[:,0,xp,yp].flatten()
    res[modNm[i]] = dat1
    del gen_out
    del G
    del curr_coarse
    
for nm in modNm:
    sns.kdeplot(res[nm], label = nm)
sns.kdeplot(fine[:,0,xp,yp].flatten(),label = "Real")
plt.legend()
#plt.show()
plt.xlabel("Value")
plt.savefig("Marginal_Compare_Toydat_CovBoth.svg", dpi = 600)
print("done!")
