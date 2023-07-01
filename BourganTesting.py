#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 23 12:14:57 2023

@author: kiridaust
"""

import torch
import numpy as np
import scipy
from scipy.spatial.distance import pdist, squareform

def bourgain_embedding(data, p, m, distmat):
    """
    bourgain embedding main function.
    Args:
        data (ndarray): Input data for embedding. Shape must be nxm, 
                        where n is the number of data points, m is the data dimension.
        p, m (float): bourgain embedding hyperparameters.
        distmat (ndarray): Distance matrix for data, shape must be nxn.
    
    Returns:
        ans (ndarray): results for bourgain embedding, shape must be nxk, where k is the
            latent space dimension.

    """
    assert(p>0 and p<1)
    assert(isinstance(m, int))
    n = data.shape[0]
    K = np.ceil(np.log(n)/np.log(1/p))
    S={}
    for j in range(int(K)):
        for i in range(m):
            S[str(i)+str('_')+str(j)]=[]
            prob = np.power(p, j+1)
            rand_num = np.random.rand(n)
            good = rand_num<prob
            good = np.argwhere(good==True).reshape((-1))
            S[str(i)+str('_')+str(j)].append(good)


    ans = np.zeros((n, int(K)*m))

    for (c, x) in enumerate(data):
        fx = np.zeros((m, int(K)))
        for i in range(fx.shape[0]):
            for j in range(fx.shape[1]):
                fx[i, j] = mindist(c, S[str(i)+str('_')+str(j)], distmat)

        fx = fx.reshape(-1)
        ans[c, :] = fx

    ans = ans - np.mean(ans, axis=0)
    dists = np.linalg.norm(ans, ord='fro')/np.sqrt(ans.shape[0])
    ans = ans/dists * np.sqrt(ans.shape[1])
    return ans

def mindist(x_id, idxset, distmat):
    """
    helper function to find the minimal distance in a given point set
    Args:
        x_id (int): id for reference point
        idxset (list): ids for the point set to test
        distmat (ndarray): distance matrix for all points
    
    Returns:
        mindist (float): minimal distance
    """
    d = distmat[x_id, idxset[0]]
    if d.shape[0] == 0:
        return 0
    else:
        return torch.min(d)
    
###bourgan dists
def pairwise_dist(data, dist, p=None):
    if not isinstance(dist, str):
        raise ValueError('dist must be str')
    
    if dist == 'l2':
        return squareform(pdist(data, metric='euclidean'))
    elif dist == 'lp':
        return squareform(pdist(data, metric='minkowski', p=p))


def pairwise_dist_generic(data, distfunc):
    n = data.shape[0]
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if j >= i:
                continue    

            dist[i, j] = distfunc(data[i, :], data[j, :])
            dist[j, i] = dist[i, j]
    
    return

    
def dist_l2(a, b):
    a = a.view((a.shape[0], -1))
    b = b.view((b.shape[0], -1))
    return torch.norm(a-b, p=2, dim=1)

def dist_l1(a, b):
    a = a.view((a.shape[0], -1))
    b = b.view((b.shape[0], -1))
    return torch.norm(a-b, p=1, dim=1)


def loadDist(dist_config):
    dist_name = dist_config['name']
    if dist_name == 'l2':
        return dist_l2
    elif dist_name == 'l1':
        return dist_l1
    else:
        raise ValueError("no dist called "+dist_name)


####Main sampler
class BourgainSampler(object):
    def __init__(self, data, path=None, dist='l2'):
        if path is not None:
            self.load(path)
            return

        self.name = "bourgain"
        p = 0.7
        m = 20
        self.eps = 0.01
        self.origin_data = data
        batch_size = data.size(0)
        reshaped_tensor = data.view(batch_size, 1, data.size(1), data.size(2))
        expanded_tensor = reshaped_tensor.expand(batch_size, batch_size, data.size(1), data.size(2))
        distmat = torch.pow(expanded_tensor - expanded_tensor.transpose(0, 1), 2).mean(dim=(2, 3))
        #distmat = torch.cdist(data,data)
        self.embedded_data = bourgain_embedding(data, p=p, m=m, distmat=distmat)
        #the data stored in Sampler are numpy array
        self.scale = float(self.get_scale(self.embedded_data, distmat))
        print('scale factor:', self.scale)

        
    def sampling(self, n):   #bourgain sampling
        num_data = self.embedded_data.shape[0]
        sampled_idx = np.random.choice(num_data, n)
        sampled_data = self.embedded_data[sampled_idx, :]
        noise = np.random.normal(scale=self.eps, size=sampled_data.shape)
        sampled_data = sampled_data + noise
        sampled_data = torch.from_numpy(sampled_data).float()
        #return torch.Tensor
        return sampled_data


    def get_scale(self, embedded_data, distmat):
        l2 = pairwise_dist(embedded_data, 'l2')
        for i in range(l2.shape[0]):
            l2[i, i] = 1
        l2 = torch.from_numpy(l2)
        div1 = torch.sqrt(torch.divide(distmat, l2))
        return torch.max(div1)

    def save(self, path):
        np.savez(path, eps=self.eps, embed=self.embedded_data, scale=self.scale, origin_data=self.origin_data)

    def load(self, path):
        ff = np.load(path)
        self.embedded_data = ff['embed']
        self.eps = ff['eps']
        self.scale = ff['scale']
        self.origin_data = ff['origin_data']
        self.name = "bourgain"


###test it
from xarray.core import dataset
from xarray.core.dataset import Dataset
import xarray as xr
device = torch.device("cuda:0")
data_folder = "/home/kiridaust/Masters/Data/processed_data/ds_temp/"
fine_train = xr.open_dataset(data_folder + "fine_train.nc", engine="netcdf4")
fine_train = torch.from_numpy(fine_train.to_array().to_numpy()).transpose(0, 1).to(device).float()
fine_train.shape
test_data = fine_train[torch.randint(0,8000,(1,400)),0,...].cpu()
test_data = torch.squeeze(test_data)

z_sampler = BourgainSampler(test_data)
sample = z_sampler.sampling(100)
sample.shape
stest = sample[:,0:256]
stest2 = torch.reshape(stest, (100,16,16))
import matplotlib.pyplot as plt
plt.imshow(stest2[5,...])
plt.hist(torch.flatten(sample))
import seaborn as sns
sns.kdeplot(torch.flatten(sample))
sns.kdeplot(sample[3,...])
test2 = torch.mean(sample, dim = 1)
sns.kdeplot(test2)


input_tensor = test_data
# Assuming your tensor is called 'input_tensor' with dimensions (100, 128, 128)
batch_size = data.size(0)
reshaped_tensor = data.view(batch_size, 1, data.size(1), data.size(2))
expanded_tensor = reshaped_tensor.expand(batch_size, batch_size, data.size(1), data.size(2))
distmat = torch.pow(expanded_tensor - expanded_tensor.transpose(0, 1), 2).mean(dim=(2, 3))


# Print the shape of the distance matrix
print(distance_matrix.shape)

