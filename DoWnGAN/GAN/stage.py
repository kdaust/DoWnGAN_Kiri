# Begin - load the data and initiate training
# Defines the hyperparameter and constants configurationsimport gc
from DoWnGAN.networks.generator import Generator
from DoWnGAN.networks.critic import Critic
from DoWnGAN.GAN.dataloader import NetCDFSR
import DoWnGAN.mlflow_tools.mlflow_utils as mlf 
import DoWnGAN.config.hyperparams as hp
from DoWnGAN.config import config
#from DoWnGAN.helpers.gen_experiment_datasets import generate_train_test_coarse_fine, load_preprocessed
from xarray.core import dataset
from xarray.core.dataset import Dataset
import xarray as xr

import torch

from mlflow.tracking import MlflowClient

def load_preprocessed():
    coarse_train = xr.open_dataset("~/Masters/Data/Test_Upsample/coarse_input/coarse_train_tpe2.nc", engine="netcdf4")
    fine_train = xr.open_dataset("~/Masters/Data/Test_Upsample/coarse_input/fine_train_small.nc", engine="netcdf4")
    coarse_test = xr.open_dataset("~/Masters/Data/Test_Upsample/coarse_input/coarse_test_tpe2.nc", engine="netcdf4")
    fine_test = xr.open_dataset("~/Masters/Data/Test_Upsample/coarse_input/fine_test_small.nc", engine="netcdf4")
    invarient = xr.open_dataset("~/Masters/Data/Test_Upsample/coarse_input/invarient_tpe2.nc", engine="netcdf4")
    return coarse_train, fine_train, coarse_test, fine_test, invarient


assert torch.cuda.is_available(), "CUDA not available"
torch.cuda.empty_cache()
# Load dataset
coarse_train, fine_train, coarse_test, fine_test, invarient = load_preprocessed()


# Convert to tensors
print("Loading region into memory...")
coarse_train = torch.from_numpy(coarse_train.to_array().to_numpy()).transpose(0, 1).to(config.device).float()
fine_train = torch.from_numpy(fine_train.to_array().to_numpy()).transpose(0, 1).to(config.device).float()
coarse_test = torch.from_numpy(coarse_test.to_array().to_numpy()).transpose(0, 1).to(config.device).float()
fine_test = torch.from_numpy(fine_test.to_array().to_numpy()).transpose(0, 1).to(config.device).float()
invarient = torch.from_numpy(invarient.to_array().to_numpy().squeeze(0)).to(config.device).float()
print("Yep this works...")
class StageData:
    def __init__(self, ):


        # Uncomment to add stochasticity
        #noise_train = torch.normal(mean=torch.zeros_like(coarse_train[:, :1, ...]), std=torch.ones_like(coarse_train[:, :1, ...]))
        #noise_test = torch.normal(mean=torch.zeros_like(coarse_test[:, :1, ...]), std=torch.ones_like(coarse_test[:, :1, ...]))

        # coarse_train = torch.cat([coarse_train, noise_train], 1)
        # coarse_test = torch.cat([coarse_test, noise_test], 1)


        print("Coarse data shape: ", coarse_train.shape)
        print("Fine data shape: ", fine_train.shape)
        print("Invarient shape: ", invarient.shape)


        # Get shapes for networks
        self.fine_dim_n = fine_train.shape[-1]
        self.n_predictands = fine_train.shape[1]
        self.coarse_dim_n = coarse_train.shape[-1]
        self.n_covariates = coarse_train.shape[1] + 1##adding invarient

        print("Network dimensions: ")
        print("Fine: ", self.fine_dim_n, "x", self.n_predictands)
        print("Coarse: ", self.coarse_dim_n, "x", self.n_covariates)

        self.critic = Critic(self.coarse_dim_n, self.fine_dim_n, self.n_predictands).to(config.device)
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
        self.dataset = NetCDFSR(coarse_train, fine_train, invarient, device=config.device)
        self.dataloader = torch.utils.data.DataLoader(
            dataset=self.dataset, batch_size=hp.batch_size, shuffle=True
        )

        self.testdataset = NetCDFSR(coarse_test, fine_test, invarient, device = config.device)
        self.testdataloader = torch.utils.data.DataLoader(
            dataset=self.testdataset, batch_size=hp.batch_size, shuffle=True
        )
