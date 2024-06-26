# Begin - load the data and initiate training
# Defines the hyperparameter and constants configurationsimport gc
from DoWnGAN.networks.generator_stochastic_tilegan import Generator
from DoWnGAN.networks.critic_covariates import Critic
from DoWnGAN.GAN.dataloader import train_dataloader
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
data_folder = "/home/kdaust/Masters/tiled_t9/"

assert torch.cuda.is_available(), "CUDA not available"
torch.cuda.empty_cache()
# Load dataset
#coarse_train, fine_train, coarse_test, fine_test, invarient = load_preprocessed()


# Convert to tensors
print("Loading region into memory...")
invar_paths = ["inv1.pt","inv2.pt","inv3.pt","inv4.pt","inv5.pt","inv6.pt","inv7.pt","inv8.pt","inv9.pt"]
coarse_paths = ["coarse1.pt","coarse2.pt","coarse3.pt","coarse4.pt","coarse5.pt","coarse6.pt","coarse7.pt","coarse8.pt","coarse9.pt"]

coarse_tiles = [torch.load(data_folder + x).to(config.device).float() for x in coarse_paths]
invar_tiles = [torch.load(data_folder + x).unsqueeze(0).to(config.device).float() for x in invar_paths]
coarse_full = torch.load(data_folder + "coarse_full.pt").to(config.device).float()
invar_full = torch.load(data_folder + "inv_full.pt").unsqueeze(0).to(config.device).float()
fine_full = torch.load(data_folder + "fine_full.pt").to(config.device).float()

print("Data successfully loaded...")

class StageData:
    def __init__(self, ):

        print("Coarse data shape: ", coarse_full.shape)
        print("Fine data shape: ", fine_full.shape)
        print("Invarient shape: ", invar_full.shape)
        print("Coarse tile shape: ", coarse_tiles[0].shape)

        self.n_predictands = fine_full.shape[1] ##adding invariant
        self.coarse_tile_dim = coarse_tiles[0].shape[-1]
        self.fine_tile_dim = invar_tiles[0].shape[-1]
        self.n_covariates = coarse_full.shape[1]##adding invarient
        self.n_invariant = 1
        
        print("Network dimensions: ")
        print("Fine: ", self.fine_tile_dim, "x", self.n_predictands)
        print("Coarse: ", self.coarse_tile_dim, "x", self.n_covariates)
        self.critic = Critic(coarse_full.shape[-1], fine_full.shape[-1], self.n_covariates, self.n_predictands).to(config.device)
        self.generator = Generator(self.coarse_tile_dim, self.fine_tile_dim, self.n_covariates, self.n_invariant, self.n_predictands).to(config.device)
    

        # Define optimizers
        self.G_optimizer = torch.optim.Adam(self.generator.parameters(), hp.lr, betas=(0.9, 0.99))
        self.C_optimizer = torch.optim.Adam(self.critic.parameters(), hp.lr, betas=(0.9, 0.99))

        # Set up the run
        # Define the mlflow experiment drectories
        self.mlclient = MlflowClient(tracking_uri=config.EXPERIMENT_PATH)
        self.exp_id = mlf.define_experiment(self.mlclient)
        self.tag = mlf.write_tags()

        # Definte the dataset objects
        self.ds_train = train_dataloader(coarse_tiles, invar_tiles, coarse_full, invar_full, fine_full, device = config.device)

        self.dataloader_train = torch.utils.data.DataLoader(
            dataset=self.ds_train, batch_size=hp.batch_size, shuffle=True
        )