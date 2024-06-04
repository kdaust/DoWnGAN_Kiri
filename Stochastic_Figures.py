import numpy as np
import torch
import mlflow
from matplotlib import pyplot as plt
from matplotlib import colorbar, colors, gridspec

device = torch.device("cuda:0")
gen_location = "../Generators/final_gens/wind_S_crps_lower_noise/"
G = mlflow.pytorch.load_model(gen_location)

import xarray as xr
batchsize = 4
data_folder = "../Data/ds_wind_full/"
cond_fields = xr.open_dataset(data_folder + "coarse_test.nc", engine="netcdf4")
fine_fields = xr.open_dataset(data_folder + "fine_train.nc", engine="netcdf4")


coarse = torch.from_numpy(cond_fields.to_array().to_numpy()).transpose(0, 1)
fine = torch.from_numpy(fine_fields.to_array().to_numpy()).transpose(0, 1)
#invariant = torch.load(data_folder + "hr_topo.pt")
data_folder = "../Data/wind_loc2/"
invariant = xr.open_dataset(data_folder + "hr_topo_loc2.nc", engine = "netcdf4")
invariant = torch.from_numpy(invariant.to_array().to_numpy())
invariant = invariant[0,...]
inv2 = torch.flipud(invariant)
torch.save(inv2, data_folder + "hr_topo_loc2.pt")

inv_in = invariant.repeat(int(batchsize),1,1,1).to(device).float()
coarse_in = coarse[1042,...]
coarse_in = coarse_in.unsqueeze(0).repeat(int(batchsize),1,1,1).to(device).float()
gen_out = G(coarse_in, inv_in).cpu().detach()
plt.close()
plt.imshow(fine[12,1,...])
plt.show()

plt.close()
plt.imshow(torch.flipud(invariant))
plt.show()

##rank histogram
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
#########################################################################

def rankhist_extreme(G, coarse, fine, invariant, random, batchsize = 4, is_invar = True):
  if is_invar:
    inv_in = invariant.repeat(int(batchsize),1,1,1).to(device).float()
  allrank9 = []
  allrank1 = []
  for sample in random:
      print("Processing",sample)
      coarse_in = coarse[sample,...]
      coarse_in = coarse_in.unsqueeze(0).repeat(int(batchsize),1,1,1).to(device).float()
      if is_invar:
        gen_out = G(coarse_in, inv_in).cpu().detach()
      else:
        gen_out = G(coarse_in).cpu().detach()
      for i in range(8):
        if is_invar:
          fine_gen = G(coarse_in, inv_in)
        else:
          fine_gen = G(coarse_in)
        gen_out = torch.cat([gen_out,fine_gen.cpu().detach()],0)
        del fine_gen
      
      # real = torch.squeeze(mp(fine[sample,0,...].unsqueeze(0)))
      # fake = torch.squeeze(mp(gen_out[:,0,...].unsqueeze(1)))
      real = torch.squeeze((fine[sample,1,...]))
      fake = torch.squeeze((gen_out[:,1,...]))
      r_q9 = torch.quantile(real, 0.999).numpy()
      f_q9 = torch.quantile(torch.reshape(fake,[fake.size(0),128*128]), 0.999, 1).numpy()
      r_q1 = torch.quantile(real, 0.001).numpy()
      f_q1 = torch.quantile(torch.reshape(fake,[fake.size(0),128*128]), 0.001, 1).numpy()
      allvals9 = np.append(f_q9,r_q9)
      allrank9.append(sorted(allvals9).index(r_q9))
      allvals1 = np.append(f_q1,r_q1)
      allrank1.append(sorted(allvals1).index(r_q1))
  return(allrank1, allrank9)


###########Synthetic Rank Hists##########
data_folder = "../Data/synthetic/no_small/"
coarse_train = np.load(data_folder+"coarse_val_reg.npy")
coarse_train = np.swapaxes(coarse_train, 0, 2)
fine_train = np.load(data_folder+"fine_val_reg.npy")
fine_train = np.swapaxes(fine_train, 0, 2)
coarse = torch.from_numpy(coarse_train)[:,None,...]
fine = torch.from_numpy(fine_train)[:,None,...]

mod = "../Generators/stochastic_paper/Synthetic/S_CRPS_full_230/"
mods = ["../Generators/stochastic_paper/Synthetic/F_MAE_nc/","../Generators/stochastic_paper/Synthetic/F_MAE_full/","../Generators/stochastic_paper/Synthetic/S_MAE_full/","../Generators/stochastic_paper/Synthetic/S_CRPS_full_230/"]
modnm = ["f_mae_nc","f_mae_full","s_mae_full","s_crps_full"]
random_samps = torch.randint(0, 1000, (50, ))

results = dict()
for x,mod in enumerate(mods):
  print("-"*20)
  G = mlflow.pytorch.load_model(mod)
  res = rankhist_preds(G, coarse, fine, None, random_samps, is_invar = False)
  results[modnm[x]] = res



res = rankhist_preds(G, coarse, fine, None, random_samps, is_invar = False)
res_nc = rankhist_preds(G, coarse, fine, None, random_samps, is_invar = False)
plt.close()
plt.hist(res)
plt.show()

##run for all models

inv_in = invariant.repeat(int(batchsize),1,1,1).to(device).float()
"../Generators/final_gens/wind_stochastic_crps/"
mods = ["../Generators/stochastic_paper/wind_noisecov_all","../Generators/final_gens/wind_freqsep/", "../Generators/final_gens/wind_stochastic_mae/","../Generators/final_gens/wind_stochastic_crps/"]
mods = ["../Generators/stochastic_paper/wind_noisecov_all/", "../Generators/final_gens/wind_freqsep/", "../Generators/final_gens/wind_stochastic_mae/","../Generators/final_gens/wind_S_crps_lower_noise/"]
modnm = ["Basic","FreqSep", "Stochastic","Stochastic_CRPS"]
random_samps = torch.randint(0, 8760, (25, ))


mod = "../Generators/Temperature/humid_7covars/humid_temp/"
mods = ["../Generators/Temperature/humid_7covars/justhumid/","../Generators/Temperature/humid_7covars/humid_temp/"]
modnm = ["humid", "humidWithTemp"]

results = dict()
for x,mod in enumerate(mods):
  print("-"*20)
  G = mlflow.pytorch.load_model(mod)
  res = rankhist_preds(G, coarse, fine, invariant, random_samps, batchsize = 6)
  results[modnm[x]] = res

plt.close()
plt.hist(results['humid_v2'])
plt.show()


mods = ["../Generators/stochastic_paper/wind_noisecov_all/", "../Generators/final_gens/wind_freqsep/", "../Generators/final_gens/wind_stochastic_mae/","../Generators/final_gens/wind_S_crps_lower_noise/"]
modnm = ["Basic","FreqSep", "Stochastic","Stochastic_CRPS"]
random_samps = torch.randint(0, 17544, (400, ))
results9 = dict()
results1 = dict()
for x,mod in enumerate(mods):
  print("-"*20)
  G = mlflow.pytorch.load_model(mod)
  res1, res9 = rankhist_extreme(G, coarse, fine, invariant, random_samps, batchsize = 6)
  results1[modnm[x]] = res1
  results9[modnm[x]] = res9
  
plt.close()
plt.hist(results9['Stochastic'])
plt.show()

with open('gen_rankhist_extreme_quantile_001.pickle', 'wb') as handle:
    pickle.dump(results1, handle, protocol=pickle.HIGHEST_PROTOCOL)

coarse2 = torch.add(coarse,15)
fine2 = torch.add(fine,15)
inv2 = torch.add(invariant, 15)

G = mlflow.pytorch.load_model("../Generators/final_gens/wind_S_crps_critic_cov_gp10/artifacts/Generator/Generator_230/")
res = rankhist_preds(G, coarse, fine, invariant, random_samps)
plt.close()
plt.hist(res)
plt.show()

plt.close()
plt.hist(results["Stochastic"])
plt.show()

plt.close()
plt.hist(results['Stochastic_CRPS'])
plt.show()


##quantile
mods = ["../Generators/stochastic_paper/wind_noisecov_all","../Generators/stochastic_paper/wind_freqsep_all/", "../Generators/stochastic_paper/wind_stochastic_nocrps_all","../Generators/critic_covariates/wind_stochastic_crps/"]
modnm = ["Basic","FreqSep", "Stochastic","CRPS"]

results = dict()
for x,mod in enumerate(mods):
  G = mlflow.pytorch.load_model(mod)
  coarse_in = coarse[0,...]
  coarse_in = coarse_in.unsqueeze(0).repeat(int(batchsize),1,1,1).to(device).float()
  gen_out = G(coarse_in, inv_in).cpu().detach()
  for i in range(1,5000):
    print(i)
    coarse_in = coarse[i,...]
    coarse_in = coarse_in.unsqueeze(0).repeat(int(batchsize),1,1,1).to(device).float()
    fine_gen = G(coarse_in, inv_in)
    gen_out = torch.cat([gen_out,fine_gen.cpu().detach()],0)
    del fine_gen
  results[modnm[x]] = gen_out
  
import pickle
with open('gen_extreme_map_v3.pickle', 'wb') as handle:
    pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)

plt.close()
plt.imshow(torch.quantile(results['Stochastic_CRPS'][:,1,...], 0.999, dim = 0))
plt.show()

with open('gen_extreme_map_v3.pickle', 'rb') as handle:
    results = pickle.load(handle)

q_basic = torch.quantile(results['Basic'], 0.999, dim = 0)
plt.close()
plt.imshow(q_basic[1,...])
plt.show()
q_fs = torch.quantile(results['FreqSep'], 0.999, dim = 0)
plt.close()
plt.imshow(q_fs[1,...])
plt.show()
q_mae = torch.quantile(results['Stochastic'], 0.999, dim = 0)
plt.close()
plt.imshow(q_mae[1,...])
plt.show()
q_crps = torch.quantile(results['CRPS'], 0.999, dim = 0)
plt.close()
plt.imshow(q_crps[1,...])
plt.show()
q_real = torch.quantile(fine[0:5000,...], 0.999, dim = 0)

test = q_real[1,...] - q_basic[1,...]




import pickle
with open('gen_extreme_map.pickle', 'wb') as handle:
    pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)

plt.close()
plt.hist(results['Stochastic_CRPS'][5000:25000])
plt.show()

###rank hist for synthetic+DEM
data_folder = "../Data/synthetic/spat_complexity/W10/"
coarse_train = np.load(data_folder+"coarse_stochastic.npy")
coarse_train = np.swapaxes(coarse_train, 0, 2)
fine_train = np.load(data_folder+"fine_stochastic.npy")
fine_train = np.swapaxes(fine_train, 0, 2)
coarse = torch.from_numpy(coarse_train)[:,None,...]
fine = torch.from_numpy(fine_train)[:,None,...]
plt.close()
plt.imshow(coarse[42,0,...])
plt.show()

coarse_mn = torch.mean(coarse, dim = 0)
batchsize = 8
G = mlflow.pytorch.load_model("../Generators/stochastic_paper/Synthetic/spat_complexity/W10/")
coarse_in = coarse_mn.repeat(batchsize, 1,1,1).to(device).float()

gen_out = G(coarse_in).cpu().detach()
for i in range(64):
    fine_gen = G(coarse_in)
    gen_out = torch.cat([gen_out,fine_gen.cpu().detach()],0)
    del fine_gen
    
import scipy    
ks_res_ncov = np.empty([128,128])
for i in range(128):
  for j in range(128):
    gen = gen_out[0:500,0,i,j]
    target = fine[0:500,0,i,j]
    test = scipy.stats.kstest(gen,target)
    ks_res_ncov[i,j] = test.statistic

ks_w10 = ks_res_ncov.flatten()
ks_w1 = ks_res_ncov.flatten()
ks_w01 = ks_res_ncov.flatten()
plt.close()
plt.imshow(ks_res_ncov)
plt.colorbar()
plt.savefig("spat_comp_w10_ks.png", dpi = 400, bbox_inches='tight')
plt.show()

plt.close()
plt.imshow(fine[42,0,...])
plt.colorbar()
plt.show()

"../Generators/stochastic_paper/Synthetic/spat_complexity/W1/gen210/"
mods = ["../Generators/stochastic_paper/Synthetic/spat_complexity/W01/", "../Generators/stochastic_paper/Synthetic/spat_complexity/W1/Generator_240/", "../Generators/stochastic_paper/Synthetic/spat_complexity/W10/"]
dats = ["../Data/synthetic/spat_complexity/W01/","../Data/synthetic/spat_complexity/W1/","../Data/synthetic/spat_complexity/W10/"]
modnm = ["w1", "w2","w4"]
random_samps = torch.randint(0, 1000, (25, ))
  

results = dict()
for x,mod in enumerate(mods):
  coarse_train = np.load(dats[x]+"coarse_val_reg.npy")
  coarse_train = np.swapaxes(coarse_train, 0, 2)
  fine_train = np.load(dats[x]+"fine_val_reg.npy")
  fine_train = np.swapaxes(fine_train, 0, 2)
  coarse = torch.from_numpy(coarse_train)[:,None,...]
  fine = torch.from_numpy(fine_train)[:,None,...]
  
  G = mlflow.pytorch.load_model(mod)
  results[modnm[x]] = rankhist_preds(G, coarse, fine, None, random_samps, is_invar = False)

plt.close()
plt.hist(results["w2"])
plt.show()

plt.close()
plt.hist(results["w4"])
plt.show()

G = mlflow.pytorch.load_model("../Generators/stochastic_paper/Synthetic/spat_complexity/W1/Generator_170/")
data_path = "../Data/synthetic/spat_complexity/W10/"
coarse_train = np.load(data_path+"coarse_val_reg.npy")
coarse_train = np.swapaxes(coarse_train, 0, 2)
fine_train = np.load(dats[x]+"fine_val_reg.npy")
fine_train = np.swapaxes(fine_train, 0, 2)
coarse = torch.from_numpy(coarse_train)[:,None,...]
fine = torch.from_numpy(fine_train)[:,None,...]
res = rankhist_preds(G, coarse, fine, None, random_samps, is_invar = False)  
plt.close()
plt.hist(res)
plt.show()

##check extremes
random = torch.randint(0, 17500, (100, ))
random = range(350)
random = np.linspace(0,8760,100).astype(int)
inv_in = invariant.repeat(int(batchsize),1,1,1).to(device).float()
mods = ["../Generators/stochastic_paper/wind_noisecov_all", "../Generators/stochastic_paper/wind_freqsep_all", "../Generators/critic_covariates/wind_stochastic_crps"]
mods = ["../Generators/stochastic_paper/wind_noisecov_all/", "../Generators/final_gens/wind_freqsep/", "../Generators/final_gens/wind_stochastic_mae/","../Generators/final_gens/wind_stochastic_crps/"]
modnm = ["Basic", "Freqsep","Stochastic", "Stochastic_CRPS"]

results = dict()
for x,mod in enumerate(mods):
  G = mlflow.pytorch.load_model(mod)
  all_gens = []
  for j in random:
    print(j)
    coarse_in = coarse[j,...]
    coarse_in = coarse_in.unsqueeze(0).repeat(int(batchsize),1,1,1).to(device).float()
    
    gen_out = G(coarse_in, inv_in).cpu().detach()
    for i in range(100):
        fine_gen = G(coarse_in,inv_in)
        gen_out = torch.cat([gen_out,fine_gen.cpu().detach()],0)
        del fine_gen
    all_gens.append(gen_out)
    del gen_out
  
  results[modnm[x]] = torch.cat(all_gens,0)

gen_results = results
import pickle
with open('gen_extreme_critic_cov.pickle', 'wb') as handle:
    pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('gen_extreme_critic_cov.pickle', 'rb') as handle:
    gen_results = pickle.load(handle)
    
    
def plot_img(img, value_range=(-2,2), extent=None):
    plt.imshow(img, cmap = "RdBu", interpolation='nearest',
        vmin = value_range[0], vmax = value_range[1], extent=extent)
    plt.gca().tick_params(left=False, bottom=False,
        labelleft=False, labelbottom=False)
        
med_res = dict()
quart_res = dict()
for key in gen_results.keys():
  print(".")
  med_res[key] = torch.median(gen_results[key], dim = 0)
  temp = gen_results[key].numpy()
  quart_res[key] = np.subtract(*np.quantile(temp,[0.75,0.25], axis = 0))

real_med = torch.median(fine, dim = 0)
reql_iqr = np.subtract(*np.quantile(fine.numpy(),[0.75,0.25], axis = 0))

pred = 1
plt.close()
#plt.rcParams['text.usetex'] = False
figsize = (5*4, 2*4)
vrng = np.quantile(real_med[0][pred,...],[0.05,0.95])
plt.figure(figsize=figsize)
gs = gridspec.GridSpec(2, 5, wspace=0.05, hspace=0.1)
plt.subplot(gs[0,0])
plot_img(real_med[0][pred,...], value_range = vrng)
plt.ylabel("Median", fontsize = 14)
plt.subplot(gs[0,1])
t1 = np.quantile(real_med[0][pred,...] - med_res['Basic'][0][pred,...],0.95)
vrng = (-t1,t1)
plot_img(real_med[0][pred,...] - med_res['Basic'][0][pred,...], value_range = vrng)
plt.subplot(gs[0,2])
plot_img(real_med[0][pred,...] - med_res['Freqsep'][0][pred,...], value_range = vrng)
plt.subplot(gs[0,3])
plot_img(real_med[0][pred,...] - med_res['Stochastic'][0][pred,...], value_range = vrng)
plt.subplot(gs[0,4])
plot_img(real_med[0][pred,...] - med_res['Stochastic_CRPS'][0][pred,...], value_range = vrng)
#plt.colorbar()
###########################################
vrng = np.quantile(reql_iqr[pred,...],[0.05,0.95])
plt.subplot(gs[1,0])
plot_img(reql_iqr[pred,...],value_range = vrng)
plt.ylabel("IQR")
plt.xlabel("Ground Truth", fontsize = 14)
plt.subplot(gs[1,1])
t1 = np.quantile(reql_iqr[pred,...] - quart_res['Basic'][pred,...],0.95)
vrng = (-t1,t1)
plot_img(reql_iqr[pred,...] - quart_res['Basic'][pred,...],value_range = vrng)
plt.xlabel("$F_{nc}^{MAE}$", fontsize=14)
plt.subplot(gs[1,2])
plot_img(reql_iqr[pred,...] - quart_res['Freqsep'][pred,...],value_range = vrng)
plt.xlabel("$F_{full}^{MAE}$", fontsize=14)
plt.subplot(gs[1,3])
plot_img(reql_iqr[pred,...] - quart_res['Stochastic'][pred,...],value_range = vrng)
plt.xlabel("$S_{full}^{MAE}$", fontsize=14)
plt.subplot(gs[1,4])
plot_img(reql_iqr[pred,...] - quart_res['Stochastic_CRPS'][pred,...],value_range = vrng)
#plt.colorbar()
plt.xlabel("$S_{full}^{CRPS}$", fontsize=14)
plt.savefig("Median_IQR_Maps_Merid.png", bbox_inches='tight', dpi = 600)

gen_results = results    
q9_res = dict()
q1_res = dict()
for key in gen_results.keys():
  print(key)
  q9_res[key] = torch.quantile(gen_results[key], 0.9999, dim = 0)
  q1_res[key] = torch.quantile(gen_results[key], 0.0001, dim = 0)
  
q_real = torch.quantile(fine, 0.9999, dim = 0)
q_real1 = torch.quantile(fine, 0.0001, dim = 0)
with open('gen_extreme_quantiles.pickle', 'wb') as handle:
    pickle.dump((q_real,q_real1), handle, protocol=pickle.HIGHEST_PROTOCOL)
    

qdif9 = dict()
qdif1 = dict()
for key in gen_results.keys():
  qdif9[key] = q_real[1,...] - q9_res[key][1,...] 
  qdif1[key] = q_real1[1,...] - q1_res[key][1,...] 

def plot_img(img, value_range=(-2,2), ecol = "black", extent=None):
    plt.imshow(img, cmap = "RdBu", interpolation='nearest',
        vmin = value_range[0], vmax = value_range[1], extent=extent, edgecolor = ecol)
    plt.gca().tick_params(left=False, bottom=False,
        labelleft=False, labelbottom=False)

plt.close()
plt.imshow(qdif1['Basic'])
plt.set_edgecolor("green")
plt.show()

plt.close()
#mpl.rcParams['text.usetex'] = True
figsize = (2*4, 2*4)
plt.figure(figsize=figsize)
gs = gridspec.GridSpec(2, 2, wspace=0.05, hspace=0.1)
plt.subplot(gs[0,0])
plot_img(qdif1['Basic'])
plt.contour(qdif1['Basic'],(0,), colors='w')
plt.xlabel("$F_{NC}^{MAE}$", fontsize=14)
#plt.xlabel("Baseline", fontsize=14)
plt.subplot(gs[0,1])
plot_img(qdif1['FreqSep'])
plt.contour(qdif1['FreqSep'],(0,), colors='w')
plt.xlabel("$F_{full}^{MAE}$", fontsize=14)
#plt.xlabel("+ Noise Inject", fontsize=14)
plt.subplot(gs[1,0])
plot_img(qdif1['Stochastic'])
plt.contour(qdif1['Stochastic'],(0,), colors='w')
plt.xlabel("$S_{full}^{MAE}$", fontsize=14)
#plt.xlabel("+ Stoch Training", fontsize=14)
plt.subplot(gs[1,1])
plot_img(qdif1['CRPS'])
#plt.colorbar()
plt.contour(qdif1['CRPS'],(0,), colors='w')
plt.xlabel("$S_{full}^{CRPS}$", fontsize=14)
#plt.xlabel("+ CRPS", fontsize=14)
#plt.show()
plt.savefig("QuantileMaps_Zonal_001.png", bbox_inches='tight', dpi = 600)


qbasic = q_real[pred,...] - q_basic[pred,...]
qfs = q_real[pred,...] - q_fs[pred,...]
qmae = q_real[pred,...] - q_mae[pred,...]
qcrps = q_real[pred,...] - q_crps[pred,...]

q_basic = torch.quantile(gen_results['Basic'], 0.9999, dim = 0)
# plt.close()
# plt.imshow(q_basic[1,...])
# plt.show()
q_fs = torch.quantile(gen_results['Freqsep'], 0.9999, dim = 0)
# plt.close()
# plt.imshow(q_fs[1,...])
# plt.show()
q_mae = torch.quantile(gen_results['Stochastic'], 0.9999, dim = 0)
# plt.close()
# plt.imshow(q_mae[1,...])
# plt.show()
q_crps = torch.quantile(gen_results['Stochastic_CRPS'], 0.9999, dim = 0)
# plt.close()
# plt.imshow(q_crps[1,...])
# plt.show()
q_real = torch.quantile(fine, 0.9999, dim = 0)

pred = 1
qbasic = q_real[pred,...] - q_basic[pred,...]
qfs = q_real[pred,...] - q_fs[pred,...]
qmae = q_real[pred,...] - q_mae[pred,...]
qcrps = q_real[pred,...] - q_crps[pred,...]


qbasic9 = q_real[pred,...] - q_basic[pred,...]
qfs9 = q_real[pred,...] - q_fs[pred,...]
qmae9 = q_real[pred,...] - q_mae[pred,...]
qcrps9 = q_real[pred,...] - q_crps[pred,...]

qcrps_test = qcrps9.numpy()

qbasic9 = qbasic9.flatten().numpy()
qfs9 = qfs9.flatten().numpy()
qmae9 = qmae9.flatten().numpy()
qcrps9 = qcrps9.flatten().numpy()

qbasic = qbasic.flatten().numpy()
qfs = qfs.flatten().numpy()
qmae = qmae.flatten().numpy()
qcrps = qcrps.flatten().numpy()
np.quantile(qbasic, 0.99)
np.quantile(qfs, 0.99)
np.quantile(qmae, 0.99)
np.quantile(qcrps, 0.99)

torch.mean(qbasic)
torch.mean(qfs)
torch.mean(qmae)
torch.mean(qcrps)

import matplotlib as mpl

# samp_dif = []
# for i in range(128):
#   for j in range(128):
#     zonal_real = fine[:,1,i,j].numpy()
#     zonal_sample = fine[temp,1,i,j].numpy()
#     real = np.quantile(zonal_real, 0.999)
#     real_sample = np.quantile(zonal_sample, 0.999)
#     samp_dif.append(real - real_sample)
# 
# dif = np.array(samp_dif)
# 
# import seaborn as sns
# plt.close()
# sns.violinplot(dif)
# plt.show()

c_dif = []
noc_dif = []
nc_dif = []
samp_dif = []
wcomp = 0
quant = 0.9999
results = gen_results
import random
for i in range(128):
  for j in range(128):
    zonal_gen_nocrps = results["Stochastic"][:,wcomp,i,j].numpy()
    zonal_gen = results["Stochastic_CRPS"][:,wcomp,i,j].numpy()
    zonal_nc = results["Basic"][:,wcomp,i,j].numpy()   
    zonal_real = fine[:,wcomp,i,j].numpy()
    #zonal_sample = fine[random.repeat(3500),wcomp,i,j].numpy()
    crps = np.quantile(zonal_gen, quant)
    nocrps = np.quantile(zonal_gen_nocrps, quant)
    noisecov = np.quantile(zonal_nc, quant)
    real = np.quantile(zonal_real, quant)
    #real_sample = np.quantile(zonal_sample, quant)
    c_dif.append(real - crps)
    noc_dif.append(real - nocrps)
    nc_dif.append(real - noisecov)
    #samp_dif.append(real - real_sample)

cdif = np.array(c_dif)
nocdif = np.array(noc_dif)
regdif = np.array(nc_dif)
sampdif = np.array(samp_dif)
import seaborn as sns
plt.close()
sns.violinplot([cdif,nocdif,regdif])
plt.show()


##make nice grid figure - stolen from Leinonen

gen_location = "../Generators/final_gens/wind_S_crps_lower_noise/"
G = mlflow.pytorch.load_model(gen_location)

def plot_img(img, value_range=(-1,3), extent=None):
    plt.imshow(img, interpolation='nearest',
        norm=colors.Normalize(*value_range), extent=extent)
    plt.gca().tick_params(left=False, bottom=False,
        labelleft=False, labelbottom=False)

num_samples=4; num_instances=4; out_fn=None; plot_stride=1; num_frames = 1

num_rows = num_samples*num_frames
num_cols = 2+num_instances

plt.close()
figsize = (num_cols*4, num_rows*4)
plt.figure(figsize=figsize)

gs = gridspec.GridSpec(num_rows, num_cols, wspace=0.05, hspace=0.05)

#value_range = batch_gen.decoder.value_range

batchsize = num_instances
inv_in = invariant.repeat(int(batchsize),1,1,1).to(device).float()
import random
random.seed(3.1415926535)
pred = [1,1,0,0]

for s in range(num_samples):
    sample = random.randint(0,8000)
    for t in range(num_frames):
        i = s*num_frames+t
        plt.subplot(gs[i,0])
        plot_img(fine[sample,pred[s],...])
        if(s == 3):
            plt.xlabel("WRF", fontsize=18)
        plt.subplot(gs[i,1])
        plot_img(coarse[sample,pred[s],...])
        if(s == 3):
            plt.xlabel("ERA5", fontsize=18)
        for k in range(num_instances):
            coarse_in = coarse[sample,...]
            coarse_in = coarse_in.unsqueeze(0).repeat(int(batchsize),1,1,1).to(device).float()

            gen_out = G(coarse_in, inv_in).cpu().detach()
            j = 2+k
            if k == (num_instances - 1):
              for z in range(24):
                fine_gen = G(coarse_in,inv_in)
                gen_out = torch.cat([gen_out,fine_gen.cpu().detach()],0)
                del fine_gen

              plt.subplot(gs[i,j])
              plot_img(torch.std(gen_out, 0)[pred[s],...], value_range=(0,1)) 
              if(s == 3):
                plt.xlabel("Cond. StDev", fontsize=18)
            else:
              plt.subplot(gs[i,j])
              plot_img(gen_out[k,pred[s],...])
              if(s == 3):
                plt.xlabel("Gen {}".format(k+1), fontsize=18)
            

plt.savefig("Example_Realisations_CRPS_Both.png", bbox_inches='tight')


# def rankhist_preds(G, coarse, fine, invariant):
#   batchsize = 4
#   random = torch.randint(0, 2000, (100, ))
#   inv_in = invariant.repeat(int(batchsize),1,1,1).to(device).float()
#   ens = []
#   truth = []
#   for sample in random:
#       print("Processing",sample)
#       coarse_in = coarse[sample,...]
#       coarse_in = coarse_in.unsqueeze(0).repeat(int(batchsize),1,1,1).to(device).float()
# 
#       gen_out = G(coarse_in, inv_in).cpu().detach()
#       for i in range(24):
#           fine_gen = G(coarse_in,inv_in)
#           gen_out = torch.cat([gen_out,fine_gen.cpu().detach()],0)
#           del fine_gen
#       
#       # real = torch.squeeze((fine[sample,0,120,120])).numpy()
#       # fake = torch.squeeze((gen_out[:,0,120,120]))
#       # ens.append(fake.unsqueeze(1))
#       # truth.append(real)
#       
#       real = torch.squeeze((fine[sample,0,...]))
#       fake = torch.squeeze((gen_out[:,0,...]))
#       ens.append(torch.reshape(fake, [100,128*128]))
#       truth.append(torch.reshape(real, [128*128]))
#   return(torch.cat(ens, 1).numpy(), np.array(truth))

##noise increase
###specify generator
gen_location = "../Generators/Noise_Increase/level1"
G = mlflow.pytorch.load_model(gen_location)

###specify data
data_folder = "../Data/synthetic/no_small/"
coarse_train = np.load(data_folder+"coarse_stochastic.npy")
coarse_train = np.swapaxes(coarse_train, 0, 2)
fine_train = np.load(data_folder+"fine_stochastic.npy")
fine_train = np.swapaxes(fine_train, 0, 2)
coarse = torch.from_numpy(coarse_train)[:,None,...]
fine = torch.from_numpy(fine_train)[:,None,...]

coarse_mn = torch.mean(coarse, dim = 0)
plt.imshow(coarse_mn[0,...])
plt.show()
batchsize = 8
coarse_in = coarse_mn.repeat(batchsize, 1,1,1).to(device).float()

gen_out = G(coarse_in).cpu().detach()
for i in range(64):
    fine_gen = G(coarse_in)
    gen_out = torch.cat([gen_out,fine_gen.cpu().detach()],0)
    del fine_gen
    
import scipy.stats
ks_res_ninj = np.empty([128,128])
for i in range(128):
  for j in range(128):
    gen = gen_out[0:500,0,i,j]
    target = fine[:,0,i,j]
    test = scipy.stats.kstest(gen,target)
    ks_res_ninj[i,j] = test.statistic
    
wass_l1 = ks_res_ninj.flatten()

plt.close()
plt.imshow(ks_res_ninj)
plt.show()

t1 = ks_ninj
t2 = t1[t1 > 0.05]
print(t2.size/t1.size)

import seaborn as sns
xp = 5
yp = 120
gen_ninj1 = gen_out[:,0,xp,yp].flatten()
gen_ninj1 = gen_ninj1.numpy()
samp2 = fine[:,0,xp,yp].flatten()
plt.close()
sns.kdeplot(gen_ninj,label = "Gen")
sns.kdeplot(samp2,label = "Real")
plt.legend()
plt.show()

## level 2
###specify generator
gen_location = "../Generators/Noise_Increase/level2"
G = mlflow.pytorch.load_model(gen_location)

gen_out = G(coarse_in).cpu().detach()
for i in range(64):
    fine_gen = G(coarse_in)
    gen_out = torch.cat([gen_out,fine_gen.cpu().detach()],0)
    del fine_gen
    
    
ks_res_ninj = np.empty([128,128])
for i in range(128):
  for j in range(128):
    gen = gen_out[0:500,0,i,j]
    target = fine[:,0,i,j]
    test = scipy.stats.kstest(gen,target)
    ks_res_ninj[i,j] = test.statistic
    
wass_l2 = ks_res_ninj.flatten()

xp = 5
yp = 120
gen_ninj2 = gen_out[:,0,xp,yp].flatten()
gen_ninj2 = gen_ninj2.numpy()
samp2 = fine[:,0,xp,yp].flatten()
plt.close()
sns.kdeplot(gen_ninj,label = "Gen")
sns.kdeplot(samp2,label = "Real")
plt.legend()
plt.show()

##level 3
gen_location = "../Generators/Noise_Increase/level3/"
G = mlflow.pytorch.load_model(gen_location)

gen_out = G(coarse_in).cpu().detach()
for i in range(64):
    fine_gen = G(coarse_in)
    gen_out = torch.cat([gen_out,fine_gen.cpu().detach()],0)
    del fine_gen
    
ks_res_ninj = np.empty([128,128])
for i in range(128):
  for j in range(128):
    gen = gen_out[0:500,0,i,j]
    target = fine[:,0,i,j]
    test = scipy.stats.kstest(gen,target)
    ks_res_ninj[i,j] = test.statistic
    
wass_l3 = ks_res_ninj.flatten()
plt.close()
sns.violinplot(y = [wass_l1,wass_l2,wass_l3],x = ["Low Noise","Med Noise", "Full Noise"], cut = 0)
plt.show()
xp = 5
yp = 120
gen_ninj3 = gen_out[:,0,xp,yp].flatten()
gen_ninj3 = gen_ninj3.numpy()
real = fine[:,0,xp,yp].flatten().numpy()
plt.close()
sns.kdeplot(gen_ninj,label = "Gen")
sns.kdeplot(samp2,label = "Real")
plt.legend()
plt.show()


##noise covariate
data_folder = "../Data/synthetic/no_small/"
coarse_train = np.load(data_folder+"coarse_stochastic.npy")
coarse_train = np.swapaxes(coarse_train, 0, 2)
fine_train = np.load(data_folder+"fine_stochastic.npy")
fine_train = np.swapaxes(fine_train, 0, 2)
coarse = torch.from_numpy(coarse_train)[:,None,...]
fine = torch.from_numpy(fine_train)[:,None,...]
coarse_mn = torch.mean(coarse, dim = 0)

batchsize = 8
gen_location = "../Generators/stochastic_paper/Synthetic/noise_cov_nosmall/"
G = mlflow.pytorch.load_model(gen_location)
coarse_in = coarse_mn.repeat(batchsize, 1,1,1).to(device).float()

gen_out = G(coarse_in).cpu().detach()
for i in range(64):
    fine_gen = G(coarse_in)
    gen_out = torch.cat([gen_out,fine_gen.cpu().detach()],0)
    del fine_gen

import seaborn as sns
xp = 10
yp = 121
gen_ncov = gen_out[:,0,xp,yp].flatten().numpy()
real_marg = fine[:,0,xp,yp].flatten().numpy()
plt.close()
sns.kdeplot(gen_ncov,label = "Gen")
sns.kdeplot(real_marg,label = "Real")
plt.legend()
plt.show()

import scipy
ks_res_ncov = np.empty([128,128])
for i in range(128):
  for j in range(128):
    gen = gen_out[0:500,0,i,j]
    target = fine[:,0,i,j]
    test = scipy.stats.kstest(gen,target)
    ks_res_ncov[i,j] = test.statistic

ks_ncov = ks_res_ncov.flatten()
plt.close()
plt.imshow(ks_res_ncov)
plt.show()

#######################noise inject###########
gen_location = "../Generators/stochastic_paper/Synthetic/noise_inj_nosmall/"
G = mlflow.pytorch.load_model(gen_location)
random_samps = torch.randint(0, 1000, (100, ))
rh_res = rankhist_preds(G, coarse, fine, None, random_samps, is_invar = False)
plt.close()
plt.hist(rh_res)
plt.show()

coarse_orig = coarse_mn.repeat(batchsize, 1,1,1)
coarse_in = coarse_orig.to(device).float()

gen_out = G(coarse_in).cpu().detach()
for i in range(64):
    fine_gen = G(coarse_in)
    gen_out = torch.cat([gen_out,fine_gen.cpu().detach()],0)
    del fine_gen

ks_res_ninj = np.empty([128,128])
for i in range(128):
  for j in range(128):
    gen = gen_out[0:500,0,i,j]
    target = fine[:,0,i,j]
    test = scipy.stats.kstest(gen,target)
    ks_res_ninj[i,j] = test.statistic

plt.close()
plt.imshow(ks_res_ninj)
plt.show()

fine_mn = torch.squeeze(torch.mean(fine, dim = 0))
plt.close()
plt.imshow(fine_mn)
plt.show()

ks_ninj = ks_res_ninj.flatten()
fine_mn_val = fine_mn.flatten().numpy()



import seaborn as sns
plt.close()
sns.violinplot([ks_ninj,ks_ncov])
plt.show()

xp = 10
yp = 121
gen_ninj = gen_out[:,0,xp,yp].flatten().numpy()
real_marg = fine[:,0,xp,yp].flatten().numpy()
plt.close()
sns.kdeplot(gen_ninj,label = "Gen")
sns.kdeplot(real_marg,label = "Real")
plt.legend()
plt.show()

#######################Bimodal Marginal###########
gen_location = "../Generators/stochastic_paper/Bimodal/"
G = mlflow.pytorch.load_model(gen_location)

data_folder = "../Data/synthetic/Bimodal_Synth/"
coarse_train = np.load(data_folder+"coarse_val.npy")
coarse_train = np.swapaxes(coarse_train, 0, 2)
fine_train = np.load(data_folder+"fine_val.npy")
fine_train = np.swapaxes(fine_train, 0, 2)
coarse = torch.from_numpy(coarse_train)[:,None,...]
fine = torch.from_numpy(fine_train)[:,None,...]
coarse_mn = torch.mean(coarse, dim = 0)

coarse_orig = coarse_mn.repeat(batchsize, 1,1,1)
coarse_in = coarse_orig.to(device).float()

gen_out = G(coarse_in).cpu().detach()
for i in range(64):
    fine_gen = G(coarse_in)
    gen_out = torch.cat([gen_out,fine_gen.cpu().detach()],0)
    del fine_gen

ks_res_ninj = np.empty([128,128])
for i in range(128):
  for j in range(128):
    gen = gen_out[0:500,0,i,j]
    target = fine[:,0,i,j]
    test = scipy.stats.kstest(gen,target)
    ks_res_ninj[i,j] = test.statistic

ks_ninj = ks_res_ninj.flatten()

import seaborn as sns
plt.close()
sns.violinplot([ks_ninj,ks_ncov])
plt.show()

xp = 120
yp = 5
gen_bimodal = gen_out[:,0,xp,yp].flatten().numpy()
real_marg = fine[:,0,xp,yp].flatten().numpy()
plt.close()
sns.kdeplot(gen_ninj,label = "Gen")
sns.kdeplot(real_marg,label = "Real")
plt.legend()
plt.show()

quant = 0.5
gen_quantile = torch.quantile(gen_out, quant, dim = 0)
gen_quantile = torch.squeeze(gen_quantile)
fine_quantile = torch.quantile(fine[0:gen_out.shape[0],...],quant, dim = 0)
fine_quantile = torch.squeeze(fine_quantile)
minval = min(torch.min(gen_quantile), torch.min(fine_quantile))
maxval = max(torch.max(gen_quantile), torch.max(fine_quantile))

plt.imshow(gen_quantile, vmin=minval, vmax=maxval)
plt.imshow(fine_quantile, vmin=minval, vmax=maxval)


################noise covariates
###specify generator
gen_location = "Generators/Synthetic/NoiseCov_LR/Generator_210/"
G = mlflow.pytorch.load_model(gen_location)

###specify data
data_folder = "../Data/SynthReg/"
coarse_train = np.load(data_folder+"coarse_stochastic.npy")
coarse_train = np.swapaxes(coarse_train, 0, 2)
fine_train = np.load(data_folder+"fine_stochastic.npy")
fine_train = np.swapaxes(fine_train, 0, 2)
coarse = torch.from_numpy(coarse_train)[:,None,...]
fine = torch.from_numpy(fine_train)[:,None,...]


coarse_mn = torch.mean(coarse, dim = 0)
plt.imshow(coarse_mn[0,...])
plt.show()
batchsize = 8
noise = torch.normal(0,1,size = [batchsize, 1, coarse.shape[2], coarse.shape[3]])
coarse_orig = coarse_mn.repeat(batchsize, 1,1,1)
coarse_in = torch.cat([coarse_orig,noise], dim = 1).to(device).float()

gen_out = G(coarse_in).cpu().detach()
for i in range(32):
    noise = torch.normal(0,1,size = [batchsize, 1, coarse.shape[2], coarse.shape[3]])
    coarse_in = torch.cat([coarse_orig,noise], dim = 1).to(device).float()
    fine_gen = G(coarse_in)
    gen_out = torch.cat([gen_out,fine_gen.cpu().detach()],0)
    del fine_gen
    
quant = 0.5
gen_quantile = torch.quantile(gen_out, quant, dim = 0)
gen_quantile = torch.squeeze(gen_quantile)
fine_quantile = torch.quantile(fine[0:gen_out.shape[0],...],quant, dim = 0)
fine_quantile = torch.squeeze(fine_quantile)
minval = min(torch.min(gen_quantile), torch.min(fine_quantile))
maxval = max(torch.max(gen_quantile), torch.max(fine_quantile))

plt.imshow(gen_quantile, vmin=minval, vmax=maxval)
plt.imshow(fine_quantile, vmin=minval, vmax=maxval)


plt.close()

plt.show()

torch.max(fine_quantile)

plt.close()
plt.imshow(fine_quantile - gen_quantile)
plt.show()
##example KDEs

import seaborn as sns
xp = 5
yp = 5
samp1 = gen_out[:,0,xp,yp].flatten()
samp2 = fine[0:264,0,xp,yp].flatten()
plt.close()
sns.kdeplot(samp1,label = "Gen")
sns.kdeplot(samp2,label = "Real")
plt.legend()
plt.show()

##examples images
import xarray as xr
batchsize = 4
data_folder = "./ds_wind/"
cond_fields = xr.open_dataset(data_folder + "coarse_test.nc", engine="netcdf4")
fine_fields = xr.open_dataset(data_folder + "fine_test.nc", engine="netcdf4")
coarse = torch.from_numpy(cond_fields.to_array().to_numpy()).transpose(0, 1)
fine = torch.from_numpy(fine_fields.to_array().to_numpy()).transpose(0, 1)
invariant = xr.open_dataset(data_folder + "DEM_Crop.nc", engine = "netcdf4")
invariant = torch.from_numpy(invariant.to_array().to_numpy())

invariant_in = invariant.repeat(int(batchsize),1,1,1).to(device).float()

sample = 1000
coarse_in = coarse[sample,...]
coarse_in = coarse_in.unsqueeze(0).repeat(int(batchsize),1,1,1).to(device).float()

G = mlflow.pytorch.load_model("./Generators/wind/just_crps/")
# gen_out = G(coarse_in, invariant_in).cpu().detach()
# #gen_out = G(coarse_in).cpu().detach()

# plt.imshow(fine[sample,0,...])
# plt.show()

# figure, axis = plt.subplots(2, 2)
  
# # For Sine Function
# axis[0, 0].imshow(gen_out[0,0,...])
# axis[0, 1].imshow(gen_out[1,0,...])
# axis[1, 0].imshow(gen_out[2,0,...])
# axis[1, 1].imshow(gen_out[3,0,...])
  
# Combine all the operations and display

sample = 1042
coarse_in = coarse[sample,...]
coarse_in = coarse_in.unsqueeze(0).repeat(int(batchsize),1,1,1).to(device).float()
invariant_in = invariant.repeat(int(batchsize),1,1,1).to(device).float()

gen_out = G(coarse_in, invariant_in).cpu().detach()
for i in range(24):
    fine_gen = G(coarse_in, invariant_in)
    gen_out = torch.cat([gen_out,fine_gen.cpu().detach()],0)
    del fine_gen
    
gen_var = torch.std(gen_out, dim = 0)
plt.close()
plt.imshow(gen_var[0,...])
plt.savefig("Example_var_nocrps.png", dpi=300)
for i in range(30):
    plt.imshow(gen_out[i,0,...])
    plt.savefig("Realisation_"+str(i)+".png",dpi = 300)
    plt.close()
    
plt.imshow(fine[1042,0,...])
plt.show()


zonal = zonal.to(device)
torch.quantile(zonal, torch.tensor([0.001,0.5,0.99]).to(device))


##############
###RALSD
import scipy
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

class NetCDFSR:
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
        noise_inv = False
        if torch.is_tensor(idx):
            idx = idx.tolist()

        #fine_ = self.fine[idx, ...]
        #fine_ = torch.cat([self.fine[idx,...],self.invarient],0)
        #coarse_ = torch.cat([self.coarse[idx, ...],self.invarient],0)
        fine_ = self.fine[idx,...]
        coarse_ = self.coarse[idx,...]
        if(self.invarient is None):
            return coarse_, fine_, -1
        else:
            if(noise_inv):
                noise_inv = torch.normal(0,1, size = [1,self.invarient.shape[1],self.invarient.shape[2]],device=self.device)
                invarient_ = torch.cat([self.invarient,noise_inv],0)
            else:
                invarient_ = self.invarient
            return coarse_, fine_, invarient_



def calc_ralsd(G,dataloader,pred_num,is_inv = True):
    torch.cuda.empty_cache()
    RALSD = []
    for i, data in enumerate(dataloader):
        if(i > 200):
            break
        print("running batch ", i)
        #torch.cuda.empty_cache()
        if is_inv:
          out = G(data[0].to("cuda:0").float(),data[2].to(device).float())
        else:
          out = G(data[0].to("cuda:0").float())
        # if(dsnum == 1):
        #   out = torch.subtract(out, 15)
        #out = G(data[0].to("cuda:0").float())
        #print(data[1][:,0,...].size())
        real = data[1][:,pred_num,...].cpu().detach()
        zonal = out[:,pred_num,...].cpu().detach()
        # gen_img.append(zonal)
        # real_img.append(real)
        
        distMetric = ralsd(zonal.numpy(),real.numpy())
        t1 = np.mean(distMetric,axis = 0)
        RALSD.append(t1)
        #print("RALSD: ",log_dist)
        del data
        del out
        del real
    #gen = torch.stack(gen_img,0)
    #real = torch.stack(real_img,0)
    return(RALSD)


##synthetic
"../Generators/Temperature/nonoise_end/"
models = ["../Generators/stochastic_paper/Synthetic/F_MAE_nc/","../Generators/stochastic_paper/Synthetic/F_MAE_full/","../Generators/stochastic_paper/Synthetic/S_MAE_full/","../Generators/stochastic_paper/Synthetic/S_CRPS_full_230/"]
modNm = ["f_mae_nc","f_mae_full","s_mae_full","s_crps_full"]

data_folder = "../Data/synthetic/no_small/"
coarse_train = np.load(data_folder+"coarse_val_reg.npy")
coarse_train = np.swapaxes(coarse_train, 0, 2)
fine_train = np.load(data_folder+"fine_val_reg.npy")
fine_train = np.swapaxes(fine_train, 0, 2)
coarse = torch.from_numpy(coarse_train)[:,None,...]
fine = torch.from_numpy(fine_train)[:,None,...]

ds_synth = NetCDFSR(coarse, fine, None, device=device)
datasets = [ds_synth,ds_synth,ds_synth,ds_synth]

res = dict()
for i in range(len(models)):
    print("Analysing model",modNm[i])
    G = mlflow.pytorch.load_model(models[i])
    dataloader = torch.utils.data.DataLoader(
        dataset=datasets[i], batch_size=8, shuffle=True
    )
    
    RALSD = calc_ralsd(G, dataloader,i, is_inv = False)
    ral = np.mean(RALSD,axis = 0)
    sdral = np.std(RALSD,axis = 0)
    res[modNm[i]] = np.column_stack((ral,sdral))


##real data
"../Generators/Temperature/humid_7covars/justhumid/"
models = ["../Generators/stochastic_paper/wind_noisecov_all/", "../Generators/final_gens/wind_freqsep/", "../Generators/final_gens/wind_stochastic_mae/","../Generators/final_gens/wind_S_crps_lower_noise/"]
models = ["../Generators/stochastic_paper/wind_noisecov_all/", "../Generators/stochastic_paper/wind_freqsep_all/", "../Generators/stochastic_paper/wind_stochastic_nocrps_all/","../Generators/stochastic_paper/wind_stochastic_crps_all/"]
models = ["../Generators/final_gens/wind_stochastic_crps/","../Generators/final_gens/wind_S_crps_critic_cov_gp10/artifacts/Generator/Generator_230/"]
modNm = ["Basic","Freqsep","Stochastic","Stochastic_CRPS"]
modNm = ["gp100","gp10"]

models = [ "../Generators/Temperature/humid_7covars/justhumid/", "../Generators/Temperature/humid_7covars/humid_temp/" ]
modNm = ["Humid","Humid(T)"]

data_folder = "../Data/ds_wind_full/"
cond_fields = xr.open_dataset(data_folder + "coarse_test.nc", engine="netcdf4")
fine_fields = xr.open_dataset(data_folder + "fine_test.nc", engine="netcdf4")
coarse = torch.from_numpy(cond_fields.to_array().to_numpy()).transpose(0, 1)
fine = torch.from_numpy(fine_fields.to_array().to_numpy()).transpose(0, 1)
#invariant = torch.load(data_folder + "hr_topo.pt")
invariant = xr.open_dataset(data_folder + "DEM_Crop.nc", engine = "netcdf4")
invariant = torch.from_numpy(invariant.to_array().to_numpy())


# data_folder = "../Data/ds_temphumid/"
# cond_fields = xr.open_dataset(data_folder + "coarse_test.nc", engine="netcdf4")
# fine_fields = xr.open_dataset(data_folder + "fine_test.nc", engine="netcdf4")
# coarse2 = torch.from_numpy(cond_fields.to_array().to_numpy()).transpose(0, 1)
# fine2 = torch.from_numpy(fine_fields.to_array().to_numpy()).transpose(0, 1)
# invariant = torch.load(data_folder + "hr_topo.pt")


ds_1 = NetCDFSR(coarse, fine, torch.squeeze(invariant).unsqueeze(0), device=device)
ds_2 = NetCDFSR(coarse, fine_h, torch.squeeze(invariant).unsqueeze(0), device=device)
datasets = [ds_1,ds_1,ds_1,ds_1] ##datasets for each model
pred_num = [1,1,1,1]
# dataloader = torch.utils.data.DataLoader(
#         dataset=ds_wind, batch_size=4, shuffle=True
#     )
# test = next(iter(dataloader))

res = dict()
for i in range(len(models)):
    print("Analysing model",modNm[i])
    G = mlflow.pytorch.load_model(models[i])
    dataloader = torch.utils.data.DataLoader(
        dataset=datasets[i], batch_size=6, shuffle=True
    )
    
    RALSD = calc_ralsd(G, dataloader,pred_num[i])
    ral = np.mean(RALSD,axis = 0)
    sdral = np.std(RALSD,axis = 0)
    res[modNm[i]] = np.column_stack((ral,sdral))

res_zonal = res
res_merid = res
plt.close()
cols = ['orange','blue','red','green']
for i,nm in enumerate(modNm):
    plt.plot(res[nm][:,0], label = nm, color = cols[i])
    plt.fill_between(range(64),res[nm][:,0]+res[nm][:,1],res[nm][:,0]-res[nm][:,1], alpha = 0.1, color = cols[i])
plt.hlines(y = 1, xmin=0, xmax=64, color = "black")
plt.xlabel("Frequency Band")
plt.ylabel("Standardised Amplitude")
plt.legend()
plt.show()
plt.savefig('RALSD_Humid_Temp.png',dpi = 600)

##############################################
###########DEM weights
def calc_rankhist(coarse, fine, invariant, G):
    batchsize = 4
    random = torch.randint(0, 1000, (30, ))
    invariant_in = invariant.repeat(int(batchsize),1,1,1).to(device).float()

    allrank = []
    #mp = torch.nn.MaxPool2d(8)
    for sample in random:
        print("Processing",sample)
        coarse_in = coarse[sample,...]
        coarse_in = coarse_in.unsqueeze(0).repeat(int(batchsize),1,1,1).to(device).float()

        gen_out = G(coarse_in, invariant_in).cpu().detach()
        for i in range(24):
            fine_gen = G(coarse_in, invariant_in)
            gen_out = torch.cat([gen_out,fine_gen.cpu().detach()],0)
            del fine_gen
        
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

### CRPS temperature
import xarray as xr
batchsize = 4
data_folder = "../Data/ds_temp/"
cond_fields = xr.open_dataset(data_folder + "coarse_test.nc", engine="netcdf4")
fine_fields = xr.open_dataset(data_folder + "fine_test.nc", engine="netcdf4")
coarse = torch.from_numpy(cond_fields.to_array().to_numpy()).transpose(0, 1)
fine = torch.from_numpy(fine_fields.to_array().to_numpy()).transpose(0, 1)
invariant = xr.open_dataset(data_folder + "DEM_Crop.nc", engine = "netcdf4")
invariant = torch.from_numpy(invariant.to_array().to_numpy())

G = mlflow.pytorch.load_model("Generators\crps\increaseweight\Generator_500")

res_crps = calc_rankhist(coarse, fine, invariant, G)
plt.hist(res_crps, bins=10, density=True)
plt.ylim((0,0.055))
plt.xlabel("Rank")
plt.ylabel("Density")
plt.savefig('RankHistCRPS.png',dpi = 600)

##W1
invar = np.load("../Data/SynthDEM/dem_crop.npy")
invariant = torch.from_numpy(invar)[None,...]
data_folder = "../Data/SynthDEM/W1/"
coarse_train = np.load(data_folder+"coarse_val_reg.npy")
coarse_train = np.swapaxes(coarse_train, 0, 2)
fine_train = np.load(data_folder+"fine_val_reg.npy")
fine_train = np.swapaxes(fine_train, 0, 2)
coarse = torch.from_numpy(coarse_train)[:,None,...]
fine = torch.from_numpy(fine_train)[:,None,...]
G = mlflow.pytorch.load_model("Generators\Synthetic\SynthDEM\W1")

plt.imshow(fine[42,0,...])
plt.savefig('ExampleW1.png',dpi = 600)

res_w1 = calc_rankhist(coarse, fine, invariant, G)

plt.hist(res_w1, bins=10, density=True)
plt.ylim((0,0.025))
plt.xlabel("Rank")
plt.ylabel("Density")
plt.savefig('RankHistW1.png',dpi = 600)

data_folder = "../Data/SynthDEM/W5/"
coarse_train = np.load(data_folder+"coarse_val_reg.npy")
coarse_train = np.swapaxes(coarse_train, 0, 2)
fine_train = np.load(data_folder+"fine_val_reg.npy")
fine_train = np.swapaxes(fine_train, 0, 2)
coarse = torch.from_numpy(coarse_train)[:,None,...]
fine = torch.from_numpy(fine_train)[:,None,...]
G = mlflow.pytorch.load_model("Generators\Synthetic\SynthDEM\W5")

plt.imshow(fine[42,0,...])
plt.savefig('ExampleW5.png',dpi = 600)

res_w5 = calc_rankhist(coarse, fine, invariant, G)
plt.hist(res_w5, bins=10, density=True)
plt.xlabel("Rank")
plt.ylabel("Density")
plt.savefig('RankHistW5.png',dpi = 600)


data_folder = "../Data/SynthDEM/W10/"
coarse_train = np.load(data_folder+"coarse_val_reg.npy")
coarse_train = np.swapaxes(coarse_train, 0, 2)
fine_train = np.load(data_folder+"fine_val_reg.npy")
fine_train = np.swapaxes(fine_train, 0, 2)
coarse = torch.from_numpy(coarse_train)[:,None,...]
fine = torch.from_numpy(fine_train)[:,None,...]
G = mlflow.pytorch.load_model("Generators\Synthetic\SynthDEM\W10")

batchsize = 4
coarse_mn = torch.mean(coarse, dim = 0)
coarse_in = coarse_mn.repeat(batchsize, 1,1,1).to(device).float()
invariant_in = invariant.repeat(batchsize, 1,1,1).to(device).float()

gen_out = G(coarse_in, invariant_in).cpu().detach()
for i in range(32):
    fine_gen = G(coarse_in, invariant_in)
    gen_out = torch.cat([gen_out,fine_gen.cpu().detach()],0)
    del fine_gen

def crps_empirical2(pred, truth):
    """
    Computes negative Continuous Ranked Probability Score CRPS* [1] between a
    set of samples ``pred`` and true data ``truth``. This uses an ``n log(n)``
    time algorithm to compute a quantity equal that would naively have
    complexity quadratic in the number of samples ``n``::

        CRPS* = E|pred - truth| - 1/2 E|pred - pred'|
              = (pred - truth).abs().mean(0)
              - (pred - pred.unsqueeze(1)).abs().mean([0, 1]) / 2

    Note that for a single sample this reduces to absolute error.

    **References**

    [1] Tilmann Gneiting, Adrian E. Raftery (2007)
        `Strictly Proper Scoring Rules, Prediction, and Estimation`
        https://www.stat.washington.edu/raftery/Research/PDF/Gneiting2007jasa.pdf

    :param torch.Tensor pred: A set of sample predictions batched on rightmost dim.
        This should have shape ``(num_samples,) + truth.shape``.
    :param torch.Tensor truth: A tensor of true observations.
    :return: A tensor of shape ``truth.shape``.
    :rtype: torch.Tensor
    """
    if pred.shape[1:] != (1,) * (pred.dim() - truth.dim() - 1) + truth.shape:
        raise ValueError(
            "Expected pred to have one extra sample dim on left. "
            "Actual shapes: {} versus {}".format(pred.shape, truth.shape)
        )
    opts = dict(device=pred.device, dtype=pred.dtype)
    num_samples = pred.size(0)
    if num_samples == 1:
        return (pred[0] - truth).abs()

    pred = pred.sort(dim=0).values
    diff = pred[1:] - pred[:-1]
    weight = torch.arange(1, num_samples, **opts) * torch.arange(
        num_samples - 1, 0, -1, **opts
    )
    weight = weight.reshape(weight.shape + (1,) * (diff.dim() - 1))

    return (pred - truth).abs().mean(0) - (diff * weight).sum(0) / num_samples**2

crps_res = crps_empirical2(gen_out[:,0,...], fine[42,0,...])
plt.imshow(crps_res)
import seaborn as sns
xp = 5
yp = 125
samp1 = gen_out[:,0,xp,yp].flatten()
samp2 = fine[0:132,0,xp,yp].flatten()
plt.close()
sns.kdeplot(samp1,label = "Gen")
sns.kdeplot(samp2,label = "Real")
plt.legend()

fine_out = fine[0:132,0,...]
fine_mn = torch.mean(fine_out, dim = 0)
fine_mn.shape

gen_mn = torch.mean(gen_out[:,0,...], dim = 0)
gen_mn.shape

diff = torch.abs(fine_mn - gen_mn)
plt.imshow(diff)
diff2 = diff/fine_mn
plt.imshow(diff2)
plt.hist(diff2.flatten())

plt.imshow(fine[42,0,...])
plt.savefig('ExampleW10.png',dpi = 600)



res_w10 = calc_rankhist(coarse, fine, invariant, G)
plt.hist(res_w10, bins=10, density=True)
plt.xlabel("Rank")
plt.ylabel("Density")
plt.savefig('RankHistW10.png',dpi = 600)

###rankhist synthetic
def calc_rankhist_synth(coarse, fine, G):
    batchsize = 4
    random = torch.randint(0, 1000, (30, ))

    allrank = []
    #mp = torch.nn.MaxPool2d(8)
    for sample in random:
        print("Processing",sample)
        coarse_in = coarse[sample,...]
        coarse_in = coarse_in.unsqueeze(0).repeat(int(batchsize),1,1,1).to(device).float()

        gen_out = G(coarse_in).cpu().detach()
        for i in range(24):
            fine_gen = G(coarse_in)
            gen_out = torch.cat([gen_out,fine_gen.cpu().detach()],0)
            del fine_gen
        
        real = torch.squeeze((fine[sample,...]))
        fake = torch.squeeze((gen_out[:,...]))

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

gen_location = "Generators/Synthetic/New/Generator_250/"
G = mlflow.pytorch.load_model(gen_location)

###specify data
data_folder = "../Data/SynthReg/"
coarse_train = np.load(data_folder+"coarse_val_reg.npy")
coarse_train = np.swapaxes(coarse_train, 0, 2)
fine_train = np.load(data_folder+"fine_val_reg.npy")
fine_train = np.swapaxes(fine_train, 0, 2)
coarse = torch.from_numpy(coarse_train)[:,None,...]
fine = torch.from_numpy(fine_train)[:,None,...]

res = calc_rankhist_synth(coarse, fine, G)

###rankhist for wind data
import xarray as xr
batchsize = 4
data_folder = "../Data/ds_wind/"
cond_fields = xr.open_dataset(data_folder + "coarse_test.nc", engine="netcdf4")
fine_fields = xr.open_dataset(data_folder + "fine_test.nc", engine="netcdf4")
coarse = torch.from_numpy(cond_fields.to_array().to_numpy()).transpose(0, 1)
noise = torch.normal(0,1,size = [coarse.shape[0], 1, coarse.shape[2], coarse.shape[3]])
coarse_noise = torch.cat([coarse,noise], dim = 1)
fine = torch.from_numpy(fine_fields.to_array().to_numpy()).transpose(0, 1)
invariant = xr.open_dataset(data_folder + "DEM_Crop.nc", engine = "netcdf4")
invariant = torch.from_numpy(invariant.to_array().to_numpy())

invariant_in = invariant.repeat(int(batchsize),1,1,1).to(device).float()

G = mlflow.pytorch.load_model("Generators\\wind\\noise_covar_LR")

test = G(coarse_noise[0:4,...].to(device), invariant_in.to(device))
plt.imshow(test[0,0,...].cpu().detach())

res = calc_rankhist(coarse_noise, fine, invariant, G)
plt.hist(res, bins=10, density=True)
plt.ylim((0,0.055))
plt.xlabel("Rank")
plt.ylabel("Density")
plt.savefig('RankHistWind_Covar.png',dpi = 600)

G = mlflow.pytorch.load_model("Generators\\wind\\noise_inject")

test = G(coarse[1000:1004,...].to(device), invariant_in.to(device))
plt.imshow(test[0,1,...].cpu().detach())
plt.savefig('ExampleWind2.png',dpi = 600)

res_inject = calc_rankhist(coarse, fine, invariant, G)
plt.hist(res_inject, bins=10, density=True)
plt.ylim((0,0.055))
plt.xlabel("Rank")
plt.ylabel("Density")
plt.savefig('RankHistWind_Inject.png',dpi = 600)

###test plot median IQR
pred = 1
fig, axes = plt.subplots(ncols=7,nrows = 2, figsize=(5.16*4, 2*4), 
                  gridspec_kw={"width_ratios":[1,0.08,1,1,1,1,0.08]})
fig.subplots_adjust(wspace=0.05, hspace=0.1)

vrng = np.quantile(real_med[0][pred,...],[0.01,0.99])
im0  = axes[0,0].imshow(real_med[0][pred,...], cmap = "viridis", vmin=vrng[0], vmax=vrng[1])
t1 = np.quantile(real_med[0][pred,...] - med_res['Basic'][0][pred,...],0.99)
vrng = (-t1,t1)
im1  = axes[0,2].imshow(real_med[0][pred,...] - med_res['Basic'][0][pred,...], vmin=vrng[0], vmax=vrng[1], cmap = "RdBu")
im2  = axes[0,3].imshow(real_med[0][pred,...] - med_res['Freqsep'][0][pred,...], vmin=vrng[0], vmax=vrng[1], cmap = "RdBu")
im3  = axes[0,4].imshow(real_med[0][pred,...] - med_res['Stochastic'][0][pred,...], vmin=vrng[0], vmax=vrng[1], cmap = "RdBu")
im3  = axes[0,5].imshow(real_med[0][pred,...] - med_res['Stochastic_CRPS'][0][pred,...], vmin=vrng[0], vmax=vrng[1], cmap = "RdBu")

vrng = np.quantile(real_quant[pred,...],[0.01,0.99])
im5  = axes[1,0].imshow(real_quant[pred,...], vmin=vrng[0], vmax=vrng[1], cmap="viridis")
t1 = np.quantile(real_quant[pred,...] - quart_res['Basic'][pred,...],0.99)
vrng = (-t1,t1)
im6  = axes[1,2].imshow(real_quant[pred,...] - quart_res['Basic'][pred,...], vmin=vrng[0], vmax=vrng[1], cmap="RdBu")
im7  = axes[1,3].imshow(real_quant[pred,...] - quart_res['Freqsep'][pred,...], vmin=vrng[0], vmax=vrng[1], cmap="RdBu")
im8  = axes[1,4].imshow(real_quant[pred,...] - quart_res['Stochastic'][pred,...], vmin=vrng[0], vmax=vrng[1], cmap="RdBu")
im9  = axes[1,5].imshow(real_quant[pred,...] - quart_res['Stochastic_CRPS'][pred,...], vmin=vrng[0], vmax=vrng[1], cmap="RdBu")

for i in range(2):
  for j in range(7):
    axes[i,j].get_yaxis().set_ticks([])
    axes[i,j].get_xaxis().set_ticks([])

axes[0,0].set_ylabel("y label")

fig.colorbar(im0,fraction=0.046, cax=axes[0,1])
fig.colorbar(im2,fraction=0.046, cax=axes[0,6])
fig.colorbar(im5,fraction=0.046, cax=axes[1,1])
fig.colorbar(im6,fraction=0.046, cax=axes[1,6])

plt.show()
plt.savefig("Median_IQR_Maps_Merid.png", bbox_inches='tight', dpi = 600)
