import torch
from DoWnGAN.config import config

def _comb1(a,b):
  a_overlap = a[:,:,:,-16:]
  b_overlap = b[:,:,:,0:16]
  avg_overlap = (a_overlap * config.back_mask_row + b_overlap * config.front_mask_row)
  comb = torch.cat([a[:,:,:,:-16],avg_overlap,b[:,:,:,16:]], dim = 3)
  return comb

def _comb2(top, bottom):
  t_overlap = top[:,:,-16:,:]
  b_overlap = bottom[:,:,0:16,:]
  tb_avg = (t_overlap * config.back_mask_col + b_overlap * config.front_mask_col)
  res = torch.cat([top[:,:,:-16,:], tb_avg, bottom[:,:,16:,:]], dim = 2)
  return(res)

def _comb_row(rlist):
  ab = _comb1(rlist[0],rlist[1])
  abc = _comb1(ab, rlist[2])
  return(abc)

def _comb_col(clist):
  ab = _comb2(clist[0],clist[1])
  abc = _comb2(ab, clist[2])
  return(abc)

def _combine_tile(g_list):
  row_ls = []
  for i in range(3):
    row_ls.append(_comb_row(g_list[i*3:(i+1)*3]))
  res = _comb_col(row_ls)
  return(res)
