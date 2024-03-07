# Begin - load the data and initiate training
# Defines the hyperparameter and constants configurationsimport gc
from DoWnGAN.networks.full_noise_generator_temperature import Generator
from DoWnGAN.networks.critic_covariates import Critic
from DoWnGAN.GAN.dataloader import NetCDFSR
import DoWnGAN.mlflow_tools.mlflow_utils as mlf 
import DoWnGAN.config.hyperparams as hp
from DoWnGAN.config import config
#from DoWnGAN.GAN.BourgainEmbed import BourgainSampler
#from DoWnGAN.helpers.gen_experiment_datasets import generate_train_test_coarse_fine, load_preprocessed
from xarray.core import dataset
from xarray.core.dataset import Dataset
import xarray as xr
import numpy as np
import torch
from mlflow.tracking import MlflowClient

highres_in = True
toydata = False
rotation = False
data_folder = "/home/kdaust/data/ds_all_vars/"

def load_preprocessed():
    if(not toydata):
        coarse_train = xr.open_dataset(data_folder + "coarse_train.nc", engine="netcdf4")
        fine_train = xr.open_dataset(data_folder + "fine_train.nc", engine="netcdf4")
        coarse_test = xr.open_dataset(data_folder + "coarse_test.nc", engine="netcdf4")
        fine_test = xr.open_dataset(data_folder + "fine_test.nc", engine="netcdf4")
        invarient = torch.load(data_folder + "hr_topo.pt").unsqueeze(0)
        return coarse_train, fine_train, coarse_test, fine_test, invarient
    else:
       coarse_train = np.load(data_folder+"coarse_train.npy")
       coarse_train = np.swapaxes(coarse_train, 0, 2)
       fine_train = np.load(data_folder+"fine_train.npy")
       fine_train = np.swapaxes(fine_train, 0, 2)
       coarse_test = np.load(data_folder+"coarse_test.npy")
       coarse_test = np.swapaxes(coarse_test, 0, 2)
       fine_test = np.load(data_folder+"fine_test.npy")
       fine_test = np.swapaxes(fine_test, 0, 2)
       invar = np.load(data_folder+"dem_crop.npy")
       return coarse_train, fine_train, coarse_test, fine_test, invar


assert torch.cuda.is_available(), "CUDA not available"
torch.cuda.empty_cache()
# Load dataset
coarse_train, fine_train, coarse_test, fine_test, invariant = load_preprocessed()


# Convert to tensors
print("Loading region into memory...")
if(not toydata):
    coarse_train = torch.from_numpy(coarse_train.to_array().to_numpy()).transpose(0, 1)
    fine_train = torch.from_numpy(fine_train.to_array().to_numpy()).transpose(0, 1)[:,(0,1,3,4),...]
    coarse_test = torch.from_numpy(coarse_test.to_array().to_numpy()).transpose(0, 1)
    fine_test = torch.from_numpy(fine_test.to_array().to_numpy()).transpose(0, 1)[:,(0,1,3,4),...]
    invariant = invariant.to(config.device)

    ##for precip:
    # which_zero = torch.sum(fine_train, dim = (1,2,3))
    # fine_train = fine_train[which_zero > 0.1,...]
    # coarse_train = coarse_train[which_zero > 0.1,...]

    # which_zero = torch.sum(fine_test, dim = (1,2,3))
    # fine_test = fine_test[which_zero > 0.1,...]
    # coarse_test = coarse_test[which_zero > 0.1,...]
    #invarient = torch.from_numpy(invarient.to_array().to_numpy().squeeze(0)).to(config.device).float()
    
#else:
    # coarse_train = torch.from_numpy(coarse_train)[:,None,...].to(config.device).float()
    # coarse_test = torch.from_numpy(coarse_test)[:,None,...].to(config.device).float()
    # fine_train = torch.from_numpy(fine_train)[:,None,...].to(config.device).float()
    # fine_test = torch.from_numpy(fine_test)[:,None,...].to(config.device).float()
print("Yep this works...")

class StageData:
    def __init__(self, ):

        #print("Min Value: ", torch.min(fine_train))
        print("Coarse data shape: ", coarse_train.shape)
        print("Fine data shape: ", fine_train.shape)
        if(highres_in):
            #print("Invarient shape: ", invarient.shape)
            print("Invarient shape: ", invariant.shape)

        self.fine_dim_n = fine_train.shape[-1]
        self.n_predictands = fine_train.shape[1] ##adding invariant
        self.coarse_dim_n = coarse_train.shape[-1]
        self.n_covariates = coarse_train.shape[1]##adding invarient
        
        if(highres_in):
            self.n_invariant = 1
            print("Network dimensions: ")
            print("Fine: ", self.fine_dim_n, "x", self.n_predictands)
            print("Coarse: ", self.coarse_dim_n, "x", self.n_covariates)
            #print("Invariant: ", invar_train.shape[1], "x", self.n_invariant)
            self.critic = Critic(self.coarse_dim_n, self.fine_dim_n,self.n_covariates, self.n_predictands).to(config.device)
            self.generator = Generator(self.coarse_dim_n, self.fine_dim_n, self.n_covariates, self.n_invariant, self.n_predictands).to(config.device)
        else:
            self.n_predictands = 1
            print("Network dimensions: ")
            print("Fine: ", self.fine_dim_n, "x", self.n_predictands)
            print("Coarse: ", self.coarse_dim_n, "x", self.n_covariates)
            #print("Generator params: ",self.coarse_dim_n,self.fine_dim_n,self.n_covariates,self.n_predictands)
            #self.critic = Critic(self.coarse_dim_n, self.fine_dim_n, self.n_predictands).to(config.device)
            self.critic = Critic(self.coarse_dim_n, self.fine_dim_n,self.n_covariates, self.n_predictands).to(config.device)
            self.generator = Generator(self.coarse_dim_n, self.fine_dim_n, self.n_covariates, self.n_predictands).to(config.device)

        # Define optimizers
        self.G_optimizer = torch.optim.Adam(self.generator.parameters(), hp.lr, betas=(0.9, 0.99))
        self.C_optimizer = torch.optim.Adam(self.critic.parameters(), hp.lr, betas=(0.9, 0.99))

        # Set up the run
        # Define the mlflow experiment drectories
        self.mlclient = MlflowClient(tracking_uri=config.EXPERIMENT_PATH)
        self.exp_id = mlf.define_experiment(self.mlclient)
        self.tag = mlf.write_tags()

        # Definte the dataset objects
        if(highres_in):
            self.dataset = NetCDFSR(coarse_train, fine_train, invariant, device=config.device)
            self.testdataset = NetCDFSR(coarse_test, fine_test, invariant, device = config.device)
        else:
            self.dataset = NetCDFSR(coarse_train, fine_train, invarient=None, device=config.device)
            self.testdataset = NetCDFSR(coarse_test, fine_test, invarient=None, device = config.device)
        
        self.dataloader = torch.utils.data.DataLoader(
            dataset=self.dataset, batch_size=hp.batch_size, shuffle=True
        )
        self.testdataloader = torch.utils.data.DataLoader(
            dataset=self.testdataset, batch_size=hp.batch_size, shuffle=True
        )
