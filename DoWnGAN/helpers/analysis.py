import numpy as np
import torch
import mlflow
import xarray as xr
import scipy
from matplotlib import pyplot as plt
from matplotlib import colorbar, colors, gridspec
from scipy import ndimage

device = torch.device("cuda:0")

def mutual_information_2d(x, y, sigma=1, normalized=False):
    """
    Computes (normalized) mutual information between two 1D variate from a
    joint histogram.
    Parameters
    ----------
    x : 1D array
        first variable
    y : 1D array
        second variable
    sigma: float
        sigma for Gaussian smoothing of the joint histogram
    Returns
    -------
    nmi: float
        the computed similariy measure
    """
    bins = (256, 256)
    EPS = np.finfo(float).eps
    jh = np.histogram2d(x, y, bins=bins)[0]

    # smooth the jh with a gaussian filter of given sigma
    ndimage.gaussian_filter(jh, sigma=sigma, mode='constant',
                                 output=jh)

    # compute marginal histograms
    jh = jh + EPS
    sh = np.sum(jh)
    jh = jh / sh
    s1 = np.sum(jh, axis=0).reshape((-1, jh.shape[0]))
    s2 = np.sum(jh, axis=1).reshape((jh.shape[1], -1))

    # Normalised Mutual Information of:
    # Studholme,  jhill & jhawkes (1998).
    # "A normalized entropy measure of 3-D medical image alignment".
    # in Proc. Medical Imaging 1998, vol. 3338, San Diego, CA, pp. 132-143.
    if normalized:
        mi = ((np.sum(s1 * np.log(s1)) + np.sum(s2 * np.log(s2)))
                / np.sum(jh * np.log(jh))) - 1
    else:
        mi = ( np.sum(jh * np.log(jh)) - np.sum(s1 * np.log(s1))
               - np.sum(s2 * np.log(s2)))

    return mi



def plot_img(img, value_range=(-2,2), cmap = "viridis", extent=None):
    im = plt.imshow(img, cmap = cmap, interpolation='nearest',
        vmin = value_range[0], vmax = value_range[1], extent=extent)
    plt.gca().tick_params(left=False, bottom=False,
        labelleft=False, labelbottom=False)


def calc_ralsd(G,dataloader,pred_num,is_inv = True):
    torch.cuda.empty_cache()
    RALSD = []
    for i, data in enumerate(dataloader):
        if(i > 400):
            break
        #print("running batch ", i)
        if is_inv:
          out = G(data[0].to("cuda:0").float(),data[2].to(device).float())
        else:
          out = G(data[0].to("cuda:0").float())

        real = data[1].cpu()
        #print(real.shape)
        if(real.shape[1] == 1):
          real = real[:,0,...]
        else:
          real = real[:,pred_num,...]
        zonal = out[:,pred_num,...].cpu().detach()
        
        distMetric = ralsd(zonal.numpy(),real.numpy())
        #t1 = np.mean(distMetric,axis = 0)
        RALSD.append(distMetric)
        del data
        del out
        del real

    return(RALSD)


def ralsd(img,real):
    # Input data
    ynew = img # Generated data
    npix = ynew.shape[-1] # Shape of image in one dimension

    # Define the wavenumbers basically
    kfreq = np.fft.fftfreq(npix) * npix 
    kfreq2D = np.meshgrid(kfreq, kfreq) 
    knrm = np.sqrt(kfreq2D[0]**2 + kfreq2D[1]**2) # Magnitude of wavenumber/vector
    knrm = knrm.flatten() 

    # Computes the fourier transform and returns the amplitudes
    def calculate_2dft(image):
        fourier_image = np.fft.fftn(image)
        fourier_amplitudes = np.abs(fourier_image)**2
        return fourier_amplitudes.flatten()

    powers = []
    for i in range(ynew.shape[0]):
        wind_2d = calculate_2dft(ynew[i, ...])
        wind_real = calculate_2dft(real[i,...])
        kbins = np.arange(0.5, npix//2+1, 1.) # Bins to average the spectra
        # kvals = 0.5 * (kbins[1:] + kbins[:-1]) # "Interpolate" at the bin center
        # This ends up computing the radial average (kinda weirdly because knrm and wind_2d are flat, but
        # unique knrm bins correspond to different radii (calculated above)
        Abins, _, _ = scipy.stats.binned_statistic(knrm, wind_2d, statistic = "mean", bins = kbins) 
        Abins *= np.pi * (kbins[1:]**2 - kbins[:-1]**2)
        
        # now for ground truth
        Abins_R, _, _ = scipy.stats.binned_statistic(knrm, wind_real, statistic = "mean", bins = kbins) 
        Abins_R *= np.pi * (kbins[1:]**2 - kbins[:-1]**2)
        Abins_stand = Abins/Abins_R
        # Add to a list -- each element is a RASPD
        powers.append(Abins_stand)
    return(powers)
  
def rankhist_preds(G, coarse, fine, invariant, random, batchsize = 4, is_invar = True):
  if is_invar:
    inv_in = invariant.repeat(int(batchsize),1,1,1).to(device).float()
  allrank = []
  mp = torch.nn.MaxPool2d(8)
  for sample in random:
      print("Processing",sample)
      coarse_in = coarse[sample,...]
      coarse_in = coarse_in.unsqueeze(0).repeat(int(batchsize),1,1,1).to(device).float()
      if is_invar:
        gen_out = G(coarse_in, inv_in).cpu().detach()
      else:
        gen_out = G(coarse_in).cpu().detach()
      for i in range(24):
        if is_invar:
          fine_gen = G(coarse_in, inv_in)
        else:
          fine_gen = G(coarse_in)
        gen_out = torch.cat([gen_out,fine_gen.cpu().detach()],0)
        del fine_gen
      
      # real = torch.squeeze(mp(fine[sample,0,...].unsqueeze(0)))
      # fake = torch.squeeze(mp(gen_out[:,0,...].unsqueeze(1)))
      real = torch.squeeze((fine[sample,0,...]))
      fake = torch.squeeze((gen_out[:,0,...]))

      rankvals = []
      for i in range(128):
          for j in range(128):
              obs = real[i,j].numpy()
              ensemble = fake[:,i,j].flatten().numpy()
              allvals = np.append(ensemble,obs)
              rankvals.append(sorted(allvals).index(obs))
  
      allrank.append(rankvals)
          
  l2 = np.array([item for sub in allrank for item in sub])
  return(l2)

