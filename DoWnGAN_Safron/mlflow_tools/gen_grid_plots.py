# Plots matplotlib grids and saves to file
import torch
import matplotlib.pyplot as plt
import torchvision
import DoWnGAN.config.hyperparams as hp
from DoWnGAN.config import config
import mlflow

def _comb_lr(a,b):
    a_overlap = a[:,:,:,-16:]
    b_overlap = b[:,:,:,0:16]
    avg_overlap = (a_overlap + b_overlap)/2
    comb = torch.cat([a[:,:,:,:-16],avg_overlap,b[:,:,:,16:]], dim = 3)
    return comb
    
def _comb_tb(top, bottom):
    t_overlap = top[:,:,-16:,:]
    b_overlap = bottom[:,:,0:16,:]
    tb_avg = (t_overlap + b_overlap)/2
    res = torch.cat([top[:,:,:-16,:], tb_avg, bottom[:,:,16:,:]], dim = 2)
    return(res)

def _combine_tile(g_list):
    top = _comb_lr(g_list[0],g_list[1])
    bottom = _comb_lr(g_list[2],g_list[3])
    res = _comb_tb(top,bottom)
    return(res)

def gen_grid_images(G, c_list, i_list, coarse, fine, epoch, train_test):
    """
    Plots a grid of images and saves them to file
    Args:
        coarse (torch.Tensor): The coarse input.
        fake (torch.Tensor): The fake input.
        real (torch.Tensor): The real input.
    """
    torch.manual_seed(3.14)
    random = torch.randint(0, hp.batch_size, (5, ))
    fake_out = [G(c.to(config.device),i.to(config.device)).detach() for (c, i) in zip(c_list,i_list)]    
    fake = _combine_tile(fake_out) ##stich together
    
    coarse = torchvision.utils.make_grid(
        coarse[random, ...],
        nrow=5
    )[0, ...]

    fake = torchvision.utils.make_grid(
        fake,
        nrow=5
    )[0, ...]

    real = torchvision.utils.make_grid(
        real[random, ...],
        nrow=5
    )[0, ...]


    fig = plt.figure(figsize=(20, 20))
    fig.suptitle("Training Samples")

    # Plot the coarse and fake samples
    subfigs = fig.subfigures(nrows=3, ncols=1)
    
    # Coarse Samples
    subfigs[0].suptitle("Coarse ERA5")
    ax = subfigs[0].subplots(1, 1)
    ax.imshow(coarse.cpu().detach(), origin="lower")

    # Generated fake
    subfigs[1].suptitle("Generated Temperature")
    ax = subfigs[1].subplots(1, 1)
    ax.imshow(fake.cpu().detach(), origin="lower")

    # Ground Truth
    subfigs[2].suptitle("WRF")
    ax = subfigs[2].subplots(1, 1)
    ax.imshow(fine.cpu().detach(), origin="lower")

    if epoch % 10 == 0:
        plt.savefig(f"{mlflow.get_artifact_uri()}/{train_test}_{epoch}.png")
    plt.savefig(f"{mlflow.get_artifact_uri()}/{train_test}.png")
    plt.close(fig)
