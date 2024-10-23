#from google.colab import drive
drive.mount('/content/drive') ## ref of reprod Drive
import os
import gc

import wandb
import torch

TORCH = '2.3.0'
CUDA = 'cpu'




"""## Dependencies"""

import argparse
from argparse import Namespace
from argparse import ArgumentError
import pickle

import sys

import math
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch_geometric.utils import dense_to_sparse, to_undirected, to_dense_adj, remove_self_loops
from torch_geometric.data import Data
from torch.optim import Adam
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GATConv
from torch_geometric.nn.conv import GraphConv
from torch.autograd import Variable
from torch.nn.parameter import Parameter

from types import SimpleNamespace as SN
from copy import deepcopy

def set_seed(seed):
  os.environ['PYTHONHASHSEED']=str(seed)
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.use_deterministic_algorithms(True)


"""# Utils
ref: https://github.com/iHeartGraph/Euler/blob/main/benchmarks/loaders/load_utils.py"""

'''
Splits edges into 85:5:10 train val test partition
(Following route of VGRNN paper)
'''
def edge_tvt_split(ei):
  ne = ei.size(1)
  val = int(ne*0.85)
  te = int(ne*0.90)

  masks = torch.zeros(3, ne).bool()
  rnd = torch.randperm(ne)
  masks[0, rnd[:val]] = True
  masks[1, rnd[val:te]] = True
  masks[2, rnd[te:]] = True

  return masks[0], masks[1], masks[2]


"""### load_vrgnn.py
ref: https://github.com/iHeartGraph/Euler/blob/main/benchmarks/loaders/load_vgrnn.py"""
class TData(Data):
  TR = 0
  VA = 1
  TE = 2
  ALL = 3

  def __init__(self, **kwargs):
    super(TData, self).__init__(**kwargs)

    # Getter methods so I don't have to write this every time
    self.tr = lambda t : self.eis[t][:, self.masks[t][0]]
    self.va = lambda t : self.eis[t][:, self.masks[t][1]]
    self.te = lambda t : self.eis[t][:, self.masks[t][2]]
    self.all = lambda t : self.eis[t]

    # To match Euler models
    self.xs = self.x
    self.x_dim = self.x.size(1)

  def get_masked_edges(self, t, mask):
    if mask == self.TR:
        return self.tr(t)
    elif mask == self.VA:
        return self.va(t)
    elif mask == self.TE:
        return self.te(t)
    elif mask == self.ALL:
        return self.all(t)
    else:
        raise ArgumentError("Mask must be TData.TR, TData.VA, TData.TE, or TData.ALL")

  def ei_masked(self, mask, t):
    '''
    So method sig matches Euler models
    '''
    return self.get_masked_edges(t, mask)

  def ew_masked(self, *args):
    '''
    VGRNN datasets don't have weighted edges
    '''
    return None



"""### generators.py"""

'''
Uses Kipf-Welling pull #25 to quickly find negative edges
(For some reason, this works a touch better than the builtin
torch geo method)
'''
def fast_negative_sampling(edge_list, batch_size, num_nodes, oversample=1.25):
  # For faster membership checking
  el_hash = lambda x : x[0,:] + x[1,:]*num_nodes

  el1d = el_hash(edge_list).numpy()
  neg = np.array([[],[]])

  while(neg.shape[1] < batch_size):
    maybe_neg = np.random.randint(0,num_nodes, (2, int(batch_size*oversample)))
    maybe_neg = maybe_neg[:, maybe_neg[0] != maybe_neg[1]] # remove self-loops
    neg_hash = el_hash(maybe_neg)

    neg = np.concatenate(
      [neg, maybe_neg[:, ~np.in1d(neg_hash, el1d)]],
      axis=1
    )

  # May have gotten some extras
  neg = neg[:, :batch_size]
  return torch.tensor(neg).long()




'''ref: Euler: https://github.com/iHeartGraph/Euler/tree/main/euler'''

# # # # # # # # # # # # # # # # # # # # # # # # # #
#          Generators for data splits for         #
#           training/testing/validation           #
#       Each returns tuple (pos, neg, embeds)     #
#      of pos and negative edge lists, and the    #
#      embeddings they reference, respectively    #
# # # # # # # # # # # # # # # # # # # # # # # # # #

'''
Assumes edge indices have already been masked
Literally just generates negative samples given a list of
true positive samples. Doesn't even need embeddings bc assumes
user already has those and the order doesn't change

If using for validation, set num_pos to a list of lengths of neg
samples to generate for balanced training
'''
def lightweight_lp(eis, num_nodes, nratio=1, num_pos=None):
  negs = []
  pos = lambda i : eis[i].size(1) if type(num_pos) == type(None) else num_pos[i]

  for i in range(len(eis)):
    ei = eis[i]
    negs.append(fast_negative_sampling(ei, pos(i), num_nodes))

  return negs

'''
Simplest one. Just returns eis and random negative sample
for each time step AT each timestep
'''
def link_detection(data, partition_fn, zs, start=0, end=None,
                    include_tr=True, batched=False, nratio=1):
  if batched:
      raise NotImplementedError("Sorry, batching is a TODO")

  end = end if end else start+len(zs)
  negs = []

  if partition_fn == None:
    partition_fn = lambda x : data.eis[x]

  for t in range(start, end):
    ei = tp = partition_fn(t)

    # Also avoid edges from training set (assuming this is val or test
    # calling the function)
    if include_tr:
        ei = torch.cat([ei, data.tr(t)], dim=1)
    neg = fast_negative_sampling(ei, int(tp.size(1)*nratio), data.num_nodes)
    negs.append(neg)

  return [remove_self_loops(partition_fn(i))[0] for i in range(start, end)], negs, zs

'''
Using embeddings from timestep t, predict links in timestep t+1
same as link prediction, just offset edge lists and embeddings by -1
'''
def link_prediction(data, partition_fn, zs, start=0, end=None,
                            include_tr=True, batched=False, nratio=1):
  # Uses every edge in the next snap shot, so no partition fn needed
  p, n, z = link_detection(
    data, partition_fn, zs, start, end,
    include_tr, batched, nratio
  )

  p = p[1:]
  n = n[1:]
  z = z[:-1]

  return p, n, z

'''
Predict links that weren't present in prev batch appearing in next batch
(Compute heavy. May want to precalculate this/only run on test set)
'''
def new_link_prediction(data, partition_fn, zs, start=0, end=None, include_tr=True, batched=False):
  if batched:
    raise NotImplementedError("Sorry, batching is a TODO")

  p, n = [], []
  b = None

  if partition_fn == None:
    partition_fn = lambda x : data.eis[x]

  end = end if end else start+len(zs)

  for i in range(start, end):
    # Use full adj matrix for new link pred
    ei = remove_self_loops(partition_fn(i))[0]

    a = b
    b = to_dense_adj(ei, max_num_nodes=data.num_nodes)[0].bool()

    if type(a) == type(None):
        continue

    # Generates new links in next time step
    new_links = (~a).logical_and(a.logical_or(b))
    new_links, _ = dense_to_sparse(new_links)

    p.append(new_links)
    n.append(
      fast_negative_sampling(
      ei, p[-1].size(1), data.num_nodes
      )
    )

  return p, n, zs[:-1]




"""### utils.py - get_score"""

from sklearn.metrics import roc_auc_score, average_precision_score# \

'''
Returns AUC and AP scores given true and false scores
'''
def get_score(pscore, nscore):
  ntp = pscore.size(0)
  ntn = nscore.size(0)

  score = torch.cat([pscore, nscore]).numpy()
  labels = np.zeros(ntp + ntn, dtype=np.int32)
  labels[:ntp] = 1

  ap = average_precision_score(labels, score)
  auc = roc_auc_score(labels, score)

  return [auc, ap]




"""# DspGNN Utils"""
"""## Edge List Preprocessing"""

def data_readcsv(p, time_win_aggr, norm_func="min-max", skip_header=False):
  if skip_header:
    df = pd.read_csv(p, header=None, names=['src', 'dst', 'attr', 'ts'], skiprows=[0])
  else:
    df = pd.read_csv(p, header=None, names=['src', 'dst', 'attr', 'ts'])
  df = df.sort_values(by=['ts']).reset_index(drop=True)
  df = df[['src', 'dst', 'ts']]
  ## for dt aggregating
  min_ts = df['ts'].min()
  max_ts = df['ts'].max()
  ## Snapshot Split By Same NB Edges
  time_win_length = len(df) / (time_win_aggr-1)
  df['ts'] = df.index // time_win_length
  return df

def data_prepare_dataframes(df, degree_norm=False):
  ## for unique node mapping
  unique_values = set(df['src']) | set(df['dst'])
  mapping = {}
  for idx, value in enumerate(unique_values):
    mapping[value] = idx
  ## for edge dataframe
  links_df = df[['ts', 'src', 'dst']].copy()
  links_df['src'] = links_df['src'].map(mapping)
  links_df['dst'] = links_df['dst'].map(mapping)
  return links_df, len(unique_values)

def df_to_edgelists(links_df):
  ''' Raw edgelist (un-indexed) for each time '''
  times = links_df.ts.sort_values().unique().astype(int) # unique timestamps
  edgelists = []
  edgelists_attr = []
  for t in range(len(times)):
    edges_t = links_df[links_df['ts'] == t]
    edgelists.append(torch.tensor(edges_t[['src', 'dst']].values).T)
    edgelists_attr.append(torch.tensor(edges_t.drop(['src', 'dst'], axis=1).values).T) # In case that attr is used for forward
  return edgelists

'''Dataset Selection'''
def dataset_selection(args):
  egc_df = None
  args.t_0 = 0

  if args.data == "bca":
    args.data_path = "data/bca/soc-sign-bitcoinalpha.csv"
    args.T = 137
    args.t_train = 95
    args.t_valid = 14
    args.t_test = 28

  elif args.data == "bco":
    args.data_path = "data/bco/soc-sign-bitcoinotc.csv"
    args.T = 136
    args.t_train = 95
    args.t_valid = 13
    args.t_test = 28

  elif args.data == "mls":
    args.data_path = "data/mls/movielens-latest-small-ratings.csv"
    args.T = 90
    args.t_train = 63
    args.t_valid = 9
    args.t_test = 18
    args.reg_e_targets = 1

  elif args.data == "uci":
    args.data_path = "data/uci/uci.csv"
    args.T = 88
    args.t_train = 62
    args.t_valid = 9
    args.t_test = 17


  args.idx_start_train = 1 + args.t_0
  args.idx_end_train = args.idx_start_train + args.t_train
  args.idx_end_valid = args.idx_end_train + args.t_valid
  args.idx_end_test = args.idx_end_valid + args.t_test

  print("Current Dataset: %s \nTotal Snapshots: %d \nTrain Snapshots: %d \nValid Snapshots: %d \nTest  Snapshots: %d \n"%(
      args.data, args.T, args.t_train, args.t_valid, args.t_test
  ))

  return args

def load_eis(edgelists, num_nodes):
  splits = [edge_tvt_split(ei) for ei in edgelists]
  data = TData(
      x=torch.eye(num_nodes), eis=edgelists, masks=splits,
      num_nodes=num_nodes, dynamic_feats=False, T=len(edgelists)
  )
  return data

def args2eis(args):
  if args.data[:2] == "bc":
    rawdata = data_readcsv(args.data_path, time_win_aggr=args.T)
  elif args.data in ["mls", "uci"]:
    rawdata = data_readcsv(args.data_path, time_win_aggr=args.T, skip_header=True)
  links_df, num_nodes = data_prepare_dataframes(rawdata)
  edgelists = df_to_edgelists(links_df)
  data = load_eis(edgelists, num_nodes)
  return data




"""## [GENERAL MODULE] Spectral Design"""

def spectral_design(args, A, nb_edges, debug=False, sp_pente=1):
  num_nodes = A.shape[0]
  undi_W = 1.0*A # save a undirected for later extract edge weights
  W = 1.0 * np.logical_or(A, A.T).astype(int)

  d = W.sum(axis=0)
  # normalized Laplacian matrix.
  dis=1/np.sqrt(d)
  dis[np.isinf(dis)]=0
  dis[np.isnan(dis)]=0
  D=np.diag(dis)
  nL=np.eye(D.shape[0])-(W.dot(D)).T.dot(D)
  V1,U1 = np.linalg.eigh(nL)
  V1[V1<0]=0

  spec_l = [W]
  ## low pass filter
  dbb=(V1.max()-V1)/V1.max()
  db=dbb**3
  A0=U1.dot(np.diag(db).dot(U1.T))
  spec_l.append(A0)
  V_ = V1.copy()#V1[V1>0.001].copy()
  Vmin = V1.min()
  Vmax = V1.max()
  Vscale = Vmax - Vmin
  # gms = [np.percentile(V_, 25), np.percentile(V_, 50), np.percentile(V_, 75)]
  # gms = [(np.percentile(V_, 15) + Vscale * 0.25 + Vmin)/2, (np.percentile(V_, 50) + Vscale * 0.50 + Vmin)/2, (np.percentile(V_, 85) + Vscale * 0.75 + Vmin)/2]
  gms = [Vscale * 0.25 + Vmin, Vscale * 0.50 + Vmin, Vscale * 0.75 + Vmin]
  # print("Spectral Centers by GM: ", gms)#gm.means_)

  # band pass filters
  # ff=np.linspace(0,V1.max(),5)
  for f in gms:
    db4=np.exp(-(((V1-f))**2)*sp_pente)
    A2=U1.dot(np.diag(db4).dot(U1.T))
    A2[np.where(np.abs(A2)<0.001)]=0
    spec_l.append(A2)

  ## high pass filter
  dbb=np.linspace(0,1,db.shape[0])
  A1=U1.dot(np.diag(dbb).dot(U1.T))
  spec_l.append(A1)

  Nt = A1.shape[-1]

  # W = W #- np.eye(Nt)
  W_2hop = undi_W @ undi_W
  W_3hop = undi_W @ undi_W @ undi_W

  supports = []
  for spectral_weights in spec_l:
    xsp = spectral_weights #(np.ones((num_nodes,num_nodes)) - np.eye(num_nodes))
    xsp = np.abs(xsp)

    ## No Self Loop Maskage
    if args.masktype == "nomask":
      mask = np.ones((Nt,Nt)) - np.eye(Nt)
    elif args.masktype == "1hop":
      mask = undi_W - np.eye(Nt)
    elif args.masktype == "2hops":
      mask = W_2hop + undi_W - np.eye(Nt)*2
    elif args.masktype == "3hops":
      mask = W_3hop + W_2hop + undi_W - np.eye(Nt)*3

    mask[mask > .1] = 1.
    mask[mask <= .1] = 0.
    xsp = xsp * mask

    ## row nomrlaize
    row_sums = xsp.sum(axis=1, keepdims=True)
    xsp = xsp / (row_sums+0.000001)

    supports.append(xsp)

  if debug:
    plt.figure(figsize=[20,20])
    plt.imshow(mask, cmap='jet'); plt.colorbar(); plt.show()
    print(xsp)
    # print(threshold)
  supports = np.array(supports)
  return supports, V1


def build_adj_matrix(edges_tensor):
  unique_nodes = torch.unique(edges_tensor)
  ### Active Node mapping
  node_to_idx = {node.item(): idx for idx, node in enumerate(unique_nodes)}
  n = len(unique_nodes)
  adj_matrix = np.zeros((n, n))
  for i in range(edges_tensor.size(1)):
    src, dst = edges_tensor[:, i].tolist()
    i, j = node_to_idx[src], node_to_idx[dst]
    adj_matrix[i][j] = 1.
    # adj_matrix[j][i] = 1.  # to sym
  return adj_matrix, node_to_idx


def extract_edge_weights_HH(edges, adj_matrix_updated, node_to_idx, debug=False):
  adj_matrix_updated = torch.tensor(adj_matrix_updated)


  indices = torch.nonzero(adj_matrix_updated, as_tuple=False)[:, 1:] # whatever which spec, only need i,j indx
  # i_positions = indices[:, 1]
  # j_positions = indices[:, 2]

  edges, _ = torch.unique(indices, dim=0, return_inverse=True)
  edges = edges.T

  keys = list(node_to_idx.values())
  values = list(node_to_idx.keys())

  extracted_weight = adj_matrix_updated[:, edges[0,:], edges[1,:]]

  if debug:
    # plt.imshow(adj_matrix_updated[0])
    print('indice shape:', indices.shape)
    print('edges  shape:', edges.shape)
    print('weight shape:', extracted_weight.shape)
    # print(edges)

  for k, v in zip(keys, values):
    edges = torch.where(edges == k, torch.tensor(v), edges)

  return edges, extracted_weight


def precompute_spectral_supports(args, edgelists, edge_coef=5, debug=False): # using args.data
  spectral_weights = []
  eigen_vals = []
  edges_hh = []
  totalt = len(edgelists)
  # for t in [totalt-1]:#range(totalt):
  for t in range(totalt):
    if t % (totalt // 4) == 0 and t != 0:
      print("Current time: %d / %d"%(t,totalt))
    raw_adj, ni = build_adj_matrix(edgelists[t])

    nb_edges = edge_coef * edgelists[t].shape[-1]

    adj_matrix_updated, eigen_val = spectral_design(args, raw_adj, nb_edges, debug)
    e_t_hh, e_t_weights = extract_edge_weights_HH(edgelists[t], torch.tensor(adj_matrix_updated), ni, debug)
    edges_hh.append(e_t_hh)
    spectral_weights.append(e_t_weights.float())
    eigen_vals.append(torch.tensor(eigen_val).float())
    del(raw_adj)
    del(adj_matrix_updated)

  print("Computation Finished!")
  return edges_hh, spectral_weights, eigen_vals




"""# Main Funcs (=euler_test.py)"""

os.chdir('...')

torch.set_num_threads(8)

fmt_score = lambda x : 'AUC: %0.4f AP: %0.4f' % (x[0], x[1])

def train(args, model, data, epochs=1, pred=False, nratio=1, lr=0.01):
    # print(lr)
  end_tr = data.T - args.TEST_TS
  opt = Adam(model.parameters(), lr=lr)

  best = (0, None)
  no_improvement = 0


  for e in range(epochs):

    if e % 20 == 0:
      print('Check: Epoch = %3d'%e)

    model.train()
    opt.zero_grad()
    zs = None

    # Get embedding
    zs = model(data.x, data.eis, data.tr)[:end_tr]


    if not pred:
      p,n,z = link_detection(data, data.tr, zs, nratio=nratio)
    else:
      p,n,z = link_prediction(data, data.tr, zs, nratio=nratio)

    loss = model.loss_fn(p,n,z)
    loss.backward()
    opt.step()

    # Done by VGRNN to improve convergence
    torch.nn.utils.clip_grad_norm_(model.parameters(), 10)

    trloss = loss.item()
    with torch.no_grad():
      model.eval()

      zs = model(data.x, data.eis, data.tr)[:end_tr]


      if not pred:
        p,n,z = link_detection(data, data.va, zs)
        st, sf = model.score_fn(p,n,z)
        sscores = get_score(st, sf)
        # print('[%d] Loss: %0.4f  \n\tSt %s ' %(e, trloss, fmt_score(sscores) ), end='')
        avg = sscores[0] + sscores[1]

      else:
        dp,dn,dz = link_prediction(data, data.va, zs, include_tr=False)
        dt, df = model.score_fn(dp,dn,dz)
        dscores = get_score(dt, df)

        dp,dn,dz = new_link_prediction(data, data.va, zs)
        dt, df = model.score_fn(dp,dn,dz)
        dnscores = get_score(dt, df)
        # print('[%d] Loss: %0.4f  \n\tPr  %s  \n\tNew %s' %(e, trloss, fmt_score(dscores), fmt_score(dnscores) ),end='')
        avg = (dscores[0] + dscores[1] )
      if args.wandb_log:
        wandb.log({"epoch": e, "train loss": trloss, "valid auc": dscores[0],
              "valid ap": dscores[1], "valid new auc": dnscores[0], "valid new ap": dnscores[1]})

      ## update best
      if avg > best[0]:
        if e > 1:
          print('at epoch %4d   |   '%e, 'vald_auc : %.4f  |  vald_ap :  %.4f'%(dscores[0],dscores[1]))
        best = (avg, deepcopy(model))
        no_improvement = 0

        # Inductive
        if not pred:
          zs = model(data.x, data.eis, data.tr)[end_tr-1:]
        # Transductive
        else:
          zs = model(data.x, data.eis, data.all)[end_tr-1:]

        if not pred:
          zs = zs[1:]
          p,n,z = link_detection(data, data.te, zs, start=end_tr)
          t, f = model.score_fn(p,n,z)
          sscores = get_score(t, f)
          print('''\nTest Scores:Static LP:  %s\n'''% fmt_score(sscores))

          wandb.summary['Test AUC'] = sscores[0]
          wandb.summary['Test AP'] = sscores[1]
          wandb.finish()

          return {'auc': sscores[0], 'ap': sscores[1]}

        else:
          p,n,z = link_prediction(data, data.all, zs, start=end_tr-1)
          t, f = model.score_fn(p,n,z)
          dscores = get_score(t, f)
          p,n,z = new_link_prediction(data, data.all, zs, start=end_tr-1)
          t, f = model.score_fn(p,n,z)
          nscores = get_score(t, f)
          print('''Test scores:Dynamic LP:     %s |  Dynamic New LP: %s \n''' %(fmt_score(dscores),fmt_score(nscores)))
        ####

      ## no progress on val set; break after a certain number of epochs
      else:
        no_improvement += 1
        # Though it's not reflected in the code, the authors for VGRNN imply in the supplimental material that after 500 epochs, early stopping may kick in
        if no_improvement >= args.patience:
          print("Early stopping...\n")
          break

  #### Testing
  model = best[1]
  with torch.no_grad():
    model.eval()

    # Inductive
    if not pred:
      zs = model(data.x, data.eis, data.tr)[end_tr-1:]
    # Transductive
    else:
      zs = model(data.x, data.eis, data.all)[end_tr-1:]

    if not pred:
      zs = zs[1:]
      p,n,z = link_detection(data, data.te, zs, start=end_tr)
      t, f = model.score_fn(p,n,z)
      sscores = get_score(t, f)
      print('''\nFinal scores:Static LP:  %s\n'''% fmt_score(sscores))

      wandb.summary['Test AUC'] = sscores[0]
      wandb.summary['Test AP'] = sscores[1]
      wandb.finish()

      return {'auc': sscores[0], 'ap': sscores[1]}

    else:
      p,n,z = link_prediction(data, data.all, zs, start=end_tr-1)
      t, f = model.score_fn(p,n,z)
      dscores = get_score(t, f)

      p,n,z = new_link_prediction(data, data.all, zs, start=end_tr-1)
      t, f = model.score_fn(p,n,z)
      nscores = get_score(t, f)

      total_params = sum(p.numel() for p in model.parameters())
      print('total params:', total_params)

      if args.wandb_log:
        wandb.summary['Test AUC Pred'] = dscores[0]
        wandb.summary['Test AP Pred'] = dscores[1]
        wandb.summary['Test AUC New Pred'] = nscores[0]
        wandb.summary['Test AP New Pred'] = nscores[1]
        wandb.summary['ModelParams'] = total_params
        wandb.finish()

      print('''Final scores:Dynamic LP:     %s |  Dynamic New LP: %s \n''' %(fmt_score(dscores),fmt_score(nscores)))
      return {'pred-auc': dscores[0], 'pred-ap': dscores[1], 'new-auc': nscores[0], 'new-ap': nscores[1],}
  del model
  del best
  gc.collect()



"""# Models"""

## EvolveGCN with Euler implements
from torch_geometric.utils.loop import add_remaining_self_loops
def convert_to_dense(data, mask, start=0, end=None):
  end = data.T if not end else end

  adjs = []
  for t in range(start, end):
    ei = data.get_masked_edges(t, mask)
    ei = add_remaining_self_loops(ei, num_nodes=data.num_nodes)[0]

    a = to_dense_adj(ei, max_num_nodes=data.num_nodes)[0]
    d = a.sum(dim=1)
    d = 1/torch.sqrt(d)
    d = torch.diag(d)
    ahat = d @ a @ d

    adjs.append(ahat)

  return adjs, [torch.eye(data.num_nodes) for _ in range(len(adjs))]


'''
https://github.com/iHeartGraph/Euler/blob/main/benchmarks/models/loss_fns.py
'''
def full_adj_nll(ei, z):
    A = to_dense_adj(ei, max_num_nodes=z.size(0))[0]
    A_tilde = z@z.T

    temp_size = A.size(0)
    temp_sum = A.sum()
    posw = float(temp_size * temp_size - temp_sum) / temp_sum
    norm = temp_size * temp_size / float((temp_size * temp_size - temp_sum) * 2)
    nll_loss_mat = F.binary_cross_entropy_with_logits(input=A_tilde, target=A, pos_weight=posw, reduction='none')
    nll_loss = -1 * norm * torch.mean(nll_loss_mat, dim=[0,1])
    return - nll_loss

'''
Implimenting DropEdge https://openreview.net/forum?id=Hkx1qkrKPr
'''
class DropEdge(nn.Module):
    def __init__(self, p):
        super().__init__()
        self.p = p

    def forward(self, ei):
        if self.training:
            mask = torch.rand(ei.size(1))
            return ei[:, mask > self.p]
        return ei


class Recurrent(nn.Module):
  def __init__(self, feat_dim, out_dim=16, hidden_dim=32, hidden_units=1, lstm="GRU"):
    super(Recurrent, self).__init__()
    self.lstm = lstm
    if self.lstm == "LSTM":
      self.t_encoder = nn.LSTM(feat_dim, hidden_dim, num_layers=hidden_units)
    elif self.lstm == "LINE":
      self.t_encoder = nn.Linear(feat_dim, hidden_dim)
    self.drop = nn.Dropout(0.25)
    self.lin = nn.Linear(hidden_dim, out_dim)
    self.out_dim = out_dim
  '''
  Expects (t, batch, feats) input
  Returns (t, batch, embed) embeddings of nodes at timesteps 0-t
  '''
  def forward(self, xs, h_0):
    xs = self.drop(xs)
    if self.lstm == "LSTM":
      if type(h_0) != type(None):
        xs, h = self.t_encoder(xs, h_0)
      else:
        xs, h = self.t_encoder(xs)
    elif self.lstm == "LINE":
      xs = self.t_encoder(xs)
    xs = self.drop(xs)
    return self.lin(xs)




"""## VGRNN"""

"""<Variational Graph Recurrent Neural Networks> https://arxiv.org/abs/1908.09710
ref: https://github.com/iHeartGraph/Euler/blob/main/benchmarks/loaders/load_vgrnn.py"""

from torch_geometric.nn import GCNConv, GraphConv

'''
Using same GRU as VGRNN paper, where linear layers are replaced with
graph conv layers
'''
class GraphGRU(nn.Module):
    def __init__(self, in_size, hidden_size, n_layers=1, asl=True):
        super(GraphGRU, self).__init__()

        self.hidden_size = hidden_size
        self.n_layers = n_layers

        # GRU parameters:
        # Update gate
        self.weight_xz = []
        self.weight_hz = []

        # Reset gate
        self.weight_xr = []
        self.weight_hr = []

        # Activation vector
        self.weight_xh = []
        self.weight_hh = []

        for i in range(self.n_layers):
            if i==0:
                self.weight_xz.append(GCNConv(in_size, hidden_size, add_self_loops=asl))
                self.weight_hz.append(GCNConv(hidden_size, hidden_size, add_self_loops=asl))
                self.weight_xr.append(GCNConv(in_size, hidden_size, add_self_loops=asl))
                self.weight_hr.append(GCNConv(hidden_size, hidden_size, add_self_loops=asl))
                self.weight_xh.append(GCNConv(in_size, hidden_size, add_self_loops=asl))
                self.weight_hh.append(GCNConv(hidden_size, hidden_size, add_self_loops=asl))
            else:
                self.weight_xz.append(GCNConv(hidden_size, hidden_size, add_self_loops=asl))
                self.weight_hz.append(GCNConv(hidden_size, hidden_size, add_self_loops=asl))
                self.weight_xr.append(GCNConv(hidden_size, hidden_size, add_self_loops=asl))
                self.weight_hr.append(GCNConv(hidden_size, hidden_size, add_self_loops=asl))
                self.weight_xh.append(GCNConv(hidden_size, hidden_size, add_self_loops=asl))
                self.weight_hh.append(GCNConv(hidden_size, hidden_size, add_self_loops=asl))

        self.sig = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.drop = nn.Dropout(0.25)


    '''
    Calculates h_out for 1 timestep
    '''
    def forward_once(self, x, ei, h):
        h_out = None

        for i in range(self.n_layers):
            if i == 0:
                z_g = self.sig(self.weight_xz[i](x, ei) + self.weight_hz[i](h, ei))
                r_g = self.sig(self.weight_xr[i](x, ei) + self.weight_hr[i](h, ei))
                h_hat = self.tanh(self.weight_xh[i](x, ei) + self.weight_hh[i](r_g * h, ei))
                h_out = z_g * h[i] + (1-z_g) * h_hat

            else:
                z_g = self.sig(self.weight_xz[i](h_out, ei) + self.weight_hz[i](h, ei))
                r_g = self.sig(self.weight_xr[i](h_out, ei) + self.weight_hr[i](h, ei))
                h_hat = self.tanh(self.weight_xh[i](h_out, ei) + self.weight_hh[i](r_g * h, ei))
                h_out = z_g * h[i] + (1-z_g) * h_hat

            h_out = self.drop(h_out)

        # Some people save every layer of the GRU but that seems pointless to me..
        # but I dunno. I'm breaking with tradition, I guess
        return h_out


    '''
    Calculates h_out for all timesteps. Returns a
    (t, batch, hidden) tensor

    xs is a 3D batch of features over time
    eis is a list of edge-lists
    h is the initial hidden state. Defaults to zero
    '''
    def forward(self, xs, eis, mask_fn=lambda x:x, h=None):
        h_out = []

        if type(h) == type(None):
            h = torch.zeros(xs.size(1), self.hidden_size)

        for t in range(len(eis)):
            x = xs[t]
            ei = mask_fn(t)

            h = self.forward_once(x, ei, h)
            h_out.append(h)

        return torch.stack(h_out)



# This file contains the VGRNN class, which is updated to use
# Pytorch Geometric since the original uses older, depreciated
# (slower) functions. Used for speed tests rather than evaluation
# as we just use what the authors reported at face value
class VGAE(nn.Module):
    def __init__(self, x_dim, hidden_dim, embed_dim):
        super(VGAE, self).__init__()

        self.c1 = GCNConv(x_dim, hidden_dim, add_self_loops=True)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(0.25)

        self.mean = GCNConv(hidden_dim, embed_dim, add_self_loops=True)
        self.std = GCNConv(hidden_dim, embed_dim, add_self_loops=True)

        self.soft = nn.Softplus()

    def forward(self, x, ei, ew=None):
        x = self.c1(x, ei, edge_weight=ew)
        x = self.relu(x)
        # x = self.drop(x)

        mean = self.mean(x, ei)
        if self.eval:
            return mean, torch.zeros((1))

        std = self.soft(self.std(x, ei))

        z = self._reparam(mean, std)
        kld = 0.5 * torch.sum(torch.exp(std) + mean**2 - 1. - std)

        return z, kld

    def _reparam(self, mean, std):
        eps1 = torch.FloatTensor(std.size()).normal_()
        eps1 = Variable(eps1)
        return eps1.mul(std).add_(mean)

class VGAE_Prior(VGAE):
    def forward(self, x, ei, pm, ps, ew=None):
        x = self.c1(x, ei, edge_weight=ew)
        x = self.relu(x)
        x = self.drop(x)

        mean = self.mean(x, ei)
        if self.eval:
            return mean, torch.zeros((1))

        std = self.soft(self.std(x, ei))

        z = self._reparam(mean, std)
        kld = self._kld_gauss(mean, std, pm, ps)

        return z, kld

    '''
    Copied straight from the VGRNN code
    '''
    def _kld_gauss(self, mean_1, std_1, mean_2, std_2):
        '''
        # Only take KLD for nodes that exist in this timeslice
        # (Assumes nodes with higher IDs appear later in the timeline
        # and that once a node appears it never dissapears. A lofty assumption,
        # I know, but this is what the authors did and it seems to work)

        (makes no difference, just slows down training so removed)

        mean_1 = mean_1[:num_nodes]
        mean_2 = mean_2[:num_nodes]
        std_1 = std_1[:num_nodes]
        std_2 = std_2[:num_nodes]
        '''

        num_nodes = mean_1.size()[0]
        kld_element =  (2 * torch.log(std_2 + self.eps) - 2 * torch.log(std_1 + self.eps) +
                        (torch.pow(std_1 + self.eps ,2) + torch.pow(mean_1 - mean_2, 2)) /
                        torch.pow(std_2 + self.eps ,2) - 1)
        return (0.5 / num_nodes) * torch.mean(torch.sum(kld_element, dim=1), dim=0)



'''
Model based on that used by the VGRNN paper
Basically the same, but without the variational part
(though that could easilly be added to make it identical)
'''
class GAE_RNN(nn.Module):
    def __init__(self, x_dim, h_dim, z_dim, grnn=True, variational=True, adj_loss=False):

        super(GAE_RNN, self).__init__()

        self.h_dim = h_dim
        self.grnn = grnn

        self.phi_x = nn.Sequential(
            nn.Linear(x_dim, h_dim),
            nn.ReLU()
        )

        self.encoder = GAE(
            h_dim*2,
            embed_dim=z_dim,
            hidden_dim=h_dim
        ) if not variational else VGAE(
            h_dim*2,
            embed_dim=z_dim,
            hidden_dim=h_dim
        )

        self.phi_z = nn.Sequential(
            nn.Linear(z_dim, h_dim),
            nn.ReLU()
        )

        self.recurrent = nn.GRUCell(
            h_dim*2, h_dim
        ) if not grnn else GraphGRU(
            h_dim*2, h_dim
        )

        self.variational = variational
        self.kld = torch.zeros((1))
        self.adj_loss = adj_loss

    '''
    Iterates through list of xs, and eis passed in (if dynamic_feats is false
    assumes xs is a single 2d tensor that doesn't change through time)
    '''
    def forward(self, xs, eis, mask_fn, ews=None, start_idx=0):
        zs = []
        h = None
        self.kld = torch.zeros((1))

        for i in range(len(eis)):
            ei = mask_fn(start_idx + i)
            h,z = self.forward_once(xs, ei, h)
            zs.append(z)

        return torch.stack(zs)


    '''
    Runs net for one snapshot
    '''
    def forward_once(self, x, ei, h):
        if type(h) == type(None):
            h = torch.zeros((x.size(0), self.h_dim))

        x = self.phi_x(x)
        gcn_x = torch.cat([x,h], dim=1)

        if self.variational:
            z, kld = self.encoder(gcn_x, ei)
            self.kld += kld
        else:
            z = self.encoder(gcn_x, ei)

        h_in = torch.cat([x, self.phi_z(z)], dim=1)

        if self.grnn:
            h = self.recurrent.forward_once(h_in, ei, h)
        else:
            h = self.recurrent(h_in, h)

        return h, z


    '''
    Inner product given edge list and embeddings at time t
    '''
    def decode(self, src, dst, z):
        dot = (z[src] * z[dst]).sum(dim=1)
        return torch.sigmoid(dot)


    '''
    Given confidence scores of true samples and false samples, return
    neg log likelihood
    '''
    def calc_loss(self, t_scores, f_scores):
        EPS = 1e-6
        pos_loss = -torch.log(t_scores+EPS).mean()
        neg_loss = -torch.log(1-f_scores+EPS).mean()

        # KLD loss is always 0 if not variational
        return pos_loss + neg_loss


    '''
    Expects a list of true edges and false edges from each time
    step. Note: edge lists need not be the same length. Requires
    less preprocessing but doesn't utilize GPU/tensor ops as effectively
    as the batched fn
    '''
    def loss_fn(self, ts, fs, zs):
        tot_loss = torch.zeros((1))
        T = len(ts)

        for i in range(T):
            if not self.adj_loss:
                t_src, t_dst = ts[i]
                f_src, f_dst = fs[i]
                z = zs[i]

                tot_loss += self.calc_loss(
                    self.decode(t_src, t_dst, z),
                    self.decode(f_src, f_dst, z)
                )
            else:
                tot_loss += full_adj_nll(ts[i], zs[i])

        return tot_loss + self.kld


    '''
    Get scores for true/false embeddings to find ROC/AP scores.
    Essentially the same as loss_fn but with no NLL
    '''
    def score_fn(self, ts, fs, zs):
        tscores = []
        fscores = []

        T = len(ts)

        for i in range(T):
            t_src, t_dst = ts[i]
            f_src, f_dst = fs[i]
            z = zs[i]

            tscores.append(self.decode(t_src, t_dst, z))
            fscores.append(self.decode(f_src, f_dst, z))

        tscores = torch.cat(tscores, dim=0)
        fscores = torch.cat(fscores, dim=0)

        return tscores, fscores


class VGRNN(GAE_RNN):
    def __init__(self, x_dim, h_dim, z_dim, adj_loss=True, pred=True):
        super(VGRNN, self).__init__(x_dim, h_dim, z_dim, grnn=True, variational=True, adj_loss=adj_loss)

        self.encoder = VGAE_Prior(
            h_dim*2,
            hidden_dim=h_dim,
            embed_dim=z_dim
        )

        self.prior = nn.Sequential(
            nn.Linear(h_dim, h_dim),
            nn.ReLU()
        )

        self.prior_mean = nn.Sequential(nn.Linear(h_dim, z_dim))
        self.prior_std = nn.Sequential(
            nn.Linear(h_dim, z_dim),
            nn.Softplus()
        )

        # Whether we return priors or means during eval
        self.pred = pred

    '''
    Runs net for one timeslice
    '''
    def forward_once(self, x, ei, h):
        if type(h) == type(None):
            h = torch.zeros((x.size(0), self.h_dim))

        x = self.phi_x(x)
        gcn_x = torch.cat([x,h], dim=1)

        prior = self.prior(h)
        prior_std = self.prior_std(prior)
        prior_mean = self.prior_mean(prior)

        z, kld = self.encoder(gcn_x, ei, pm=prior_mean, ps=prior_std)
        self.kld += kld

        h_in = torch.cat([x, self.phi_z(z)], dim=1)
        h = self.recurrent.forward_once(h_in, ei, h)

        # Regardless of if self.pred == True Z is means if self.eval == True
        z = prior_mean if self.pred and self.eval else z
        return h, z


"""## Euler
(Euler / benchmarks / models / euler_serial)
"""

### EulerGCN
class GAE(nn.Module):
    def __init__(self, feat_dim, embed_dim=16, hidden_dim=32):
        super(GAE, self).__init__()

        #self.lin = nn.Sequential(nn.Linear(feat_dim, hidden_dim), nn.ReLU())
        self.c1 = GCNConv(feat_dim, hidden_dim, add_self_loops=True)
        self.relu = nn.ReLU()
        self.c2 = GCNConv(hidden_dim, embed_dim, add_self_loops=True)
        self.drop = nn.Dropout(0.25)
        self.de = DropEdge(0.8)

    def forward(self, x, ei, ew=None):
        ei = self.de(ei)
        x = self.c1(x, ei, edge_weight=ew)
        x = self.relu(x)
        x = self.drop(x)

        return self.c2(x, ei, edge_weight=ew)


class EulerGCN(nn.Module):
    def __init__(self, x_dim, h_dim, z_dim, gru_hidden_units=1,
                dynamic_feats=False, dense_loss=False,
                use_predictor=False, use_w=True, lstm=False,
                neg_weight=0.5):
        super(EulerGCN, self).__init__()

        self.weightless = not use_w
        self.kld_weight = 0
        self.dynamic_feats = dynamic_feats
        self.neg_weight = neg_weight
        self.cutoff = None
        self.z_dim = z_dim
        self.drop = nn.Dropout(0.05)

        self.gcn = GAE(
            x_dim,
            hidden_dim=h_dim,
            embed_dim=h_dim if gru_hidden_units > 0 else z_dim
        )

        self.gru = Recurrent(
            h_dim, out_dim=z_dim,
            hidden_dim=h_dim,
            hidden_units=gru_hidden_units,
            lstm=lstm
        ) if gru_hidden_units > 0 else None

        self.use_predictor = use_predictor
        self.predictor = nn.Sequential(
            nn.Linear(z_dim, 1),
            nn.Sigmoid()
        ) if use_predictor else None

        self.sig = nn.Sigmoid()

        self.dense_loss=dense_loss
        msg = "dense" if self.dense_loss else 'sparse'
        print("Using %s loss" % msg)

    '''
    Iterates through list of xs, and eis passed in (if dynamic_feats is false
    assumes xs is a single 2d tensor that doesn't change through time)
    '''
    def forward(self, xs, eis, mask_fn, ew_fn=None, start_idx=0,
                include_h=False, h_0=None):
        embeds = self.encode(xs, eis, mask_fn, ew_fn, start_idx)

        if type(self.gru) == type(None):
            return embeds
        else:
            return self.gru(torch.tanh(embeds), h_0)


    '''
    Split proceses in two to make it easier to combine embeddings with
    different masks (ie allow train set to influence test set embeds)
    '''
    def encode(self, xs, eis, mask_fn, ew_fn=None, start_idx=0):
        embeds = []

        for i in range(len(eis)):
            ei = mask_fn(start_idx + i)
            ew = None if not ew_fn or self.weightless else ew_fn(start_idx + i)
            x = xs if not self.dynamic_feats else xs[start_idx + i]

            z = self.gcn(x,ei,ew)
            embeds.append(z)

        return torch.stack(embeds)


    '''
    Inner product given edge list and embeddings at time t
    '''
    def decode(self, src, dst, z, as_probs=False):
        if self.use_predictor:
            return self.predictor(
                self.drop(z[src]) * self.drop(z[dst])
            )

        dot = (self.drop(z[src]) * self.drop(z[dst])).sum(dim=1)
        logits = self.sig(dot)

        if as_probs:
            return self.__logits_to_probs(logits)
        return logits


    '''
    Given confidence scores of true samples and false samples, return
    neg log likelihood
    '''
    def calc_loss(self, t_scores, f_scores):
        EPS = 1e-6
        pos_loss = -torch.log(t_scores+EPS).mean()
        neg_loss = -torch.log(1-f_scores+EPS).mean()

        return (1-self.neg_weight) * pos_loss + self.neg_weight * neg_loss


    '''
    Expects a list of true edges and false edges from each time
    step. Note: edge lists need not be the same length. Requires
    less preprocessing but doesn't utilize GPU/tensor ops as effectively
    as the batched fn
    '''
    def loss_fn(self, ts, fs, zs):
        tot_loss = torch.zeros((1))
        T = len(ts)

        for i in range(T):
            t_src, t_dst = ts[i]
            f_src, f_dst = fs[i]
            z = zs[i]

            if not self.dense_loss:
                tot_loss += self.calc_loss(
                    self.decode(t_src, t_dst, z),
                    self.decode(f_src, f_dst, z)
                )
            else:
                tot_loss += full_adj_nll(ts[i], z)

        return tot_loss.true_divide(T)

    '''
    Get scores for true/false embeddings to find ROC/AP scores.
    Essentially the same as loss_fn but with no NLL

    Returns logits unless as_probs is True
    '''
    def score_fn(self, ts, fs, zs, as_probs=False):
        tscores = []
        fscores = []

        T = len(ts)

        for i in range(T):
            t_src, t_dst = ts[i]
            f_src, f_dst = fs[i]
            z = zs[i]

            tscores.append(self.decode(t_src, t_dst, z))
            fscores.append(self.decode(f_src, f_dst, z))

        tscores = torch.cat(tscores, dim=0)
        fscores = torch.cat(fscores, dim=0)

        if as_probs:
            tscores=self.__logits_to_probs(tscores)
            fscores=self.__logits_to_probs(fscores)

        return tscores, fscores


    '''
    Converts from log odds (what the encode method outputs) to probabilities
    '''
    def __logits_to_probs(self, logits):
        odds = torch.exp(logits)
        probs = odds.true_divide(1+odds)
        return probs




"""## DspGNN"""

from torch import nn
from torch_geometric.nn import GCNConv

def dsp_dropedge(matrix, drop_ratio=0.8):
    total_columns = matrix.shape[1]
    keep_columns = int(total_columns * (1 - drop_ratio))
    random_indices = np.random.choice(total_columns, keep_columns, replace=False)
    drop_matrix = matrix[:, random_indices].clone()
    return drop_matrix

class SPGNN(nn.Module):
  def __init__(self, args, ablation_test_spectral=False, ablation_test_hyperhop=False, ablation_test_linear_combi=False):
    super(SPGNN, self).__init__()
    ''' Hidden state of all nodes at t=0 '''
    self.N, self.d_x, self.d_h, self.nb_spec = args.N, args.d_x, args.d_h, args.nb_spectral_supports
    self.spec_support = args.spec_support
    self.norma = args.norma

    d_h = self.d_h
    d_x = self.d_x
    self.sage_conv1 = GraphConv(d_x, d_h, aggr='mean', bias=True)
    self.sage_conv2 = GraphConv(d_x, d_h, aggr='mean', bias=True)
    self.sage_conv3 = GraphConv(d_x, d_h, aggr='mean', bias=True)
    self.sage_conv4 = GraphConv(d_x, d_h, aggr='mean', bias=True)
    self.sage_conv5 = GraphConv(d_x, d_h, aggr='mean', bias=True)
    self.sage_conv6 = GraphConv(d_x, d_h, aggr='mean', bias=True)
    self.feat_proj = nn.Linear(in_features=self.d_x, out_features=d_h)

    self.sage_combine_projection = nn.Linear(in_features=(self.nb_spec+1)*d_h, out_features=1*d_h)
    ''' Weights: [d_hid + d_agg, d_hid, K], K is the number of agg timesteps '''
    self.proj_linear = nn.Linear(2*d_h, d_h)
    self.dropout = nn.Dropout(p=0.5)
    self.relu = nn.ReLU()
    self.init_params()

  def init_params(self):
    for param in self.parameters():
      if len(param.shape) > 1:
        nn.init.xavier_uniform_(param)
      else:
        nn.init.constant_(param, 0.)

  def forward(self, H_t_in, ei_t, ew_t):
    ''' Generate H_t by catting previous H[t-K:t-1] (train and forecast)'''
    if args.dropedge:
        ei_t = dsp_dropedge(ei_t, args.dropedge)
        ew_t = dsp_dropedge(ew_t, args.dropedge)
    agg_feats1 = self.sage_conv1(H_t_in, ei_t, ew_t[0])
    agg_feats2 = self.sage_conv2(H_t_in, ei_t, ew_t[1])
    agg_feats3 = self.sage_conv3(H_t_in, ei_t, ew_t[2])
    agg_feats4 = self.sage_conv4(H_t_in, ei_t, ew_t[3])
    agg_feats5 = self.sage_conv5(H_t_in, ei_t, ew_t[4])
    agg_feats6 = self.sage_conv5(H_t_in, ei_t, ew_t[5])
    feat_ht_in = self.feat_proj(H_t_in)
    agg_feats_concat = torch.cat((feat_ht_in, agg_feats1, agg_feats2, agg_feats3,
                    agg_feats4, agg_feats5, agg_feats6), dim=1)
    agg_feats_projected = self.sage_combine_projection(agg_feats_concat)
    agg_feats_projected = self.relu(agg_feats_projected)
    agg_feats_projected = self.dropout(agg_feats_projected)
    H_t = self.proj_linear(torch.cat((feat_ht_in, agg_feats_projected), dim=1))
    if self.norma:
      H_t = F.normalize(H_t, p=2, dim=1)
    return H_t
    # return torch.unsqueeze(H_t, 0) # [N,d] -> [1,N,d]


class DspGNN(nn.Module):
  def __init__(self, args, x_dim, h_dim, eis, ews, gru_hidden_units=1,
          use_predictor=False, use_w=True, lstm=False,
          neg_weight=0.5, supports=None, dropedge=0):
    super(DspGNN, self).__init__()

    self.neg_weight = neg_weight
    self.h_dim = h_dim
    self.drop = nn.Dropout(0.2)

    self.eis = eis
    self.ews = ews

    self.gcn = SPGNN(args, dropedge)
    self.rnn = Recurrent(h_dim, out_dim=h_dim, hidden_dim=h_dim, hidden_units=gru_hidden_units, lstm=lstm)

    self.use_predictor = use_predictor
    self.predictor = nn.Sequential(nn.Linear(h_dim, 1), nn.Sigmoid()) if use_predictor else None

    self.sig = nn.Sigmoid()

  '''
  Iterates through list of xs, and eis passed in (if dynamic_feats is false
  assumes xs is a single 2d tensor that doesn't change through time)
  '''
  def forward(self, xs, eis, mask_fn, ew_fn=None, start_idx=0, h_0=None):
    embeds = []
    for i in range(len(eis)):
      x = xs
      ei = self.eis[i]
      ew = self.ews[i]
      z = self.gcn(x,ei,ew)
      embeds.append(z)
    embeds = torch.stack(embeds)
    # return torch.tanh(embeds) #self.rnn(torch.tanh(embeds), h_0)
    return self.rnn(torch.tanh(embeds), h_0)

  '''
  Inner product given edge list and embeddings at time t
  '''
  def decode(self, src, dst, z):
    if self.use_predictor:
      return self.predictor(self.drop(z[src]) * self.drop(z[dst]))
    else:
      dot = (self.drop(z[src]) * self.drop(z[dst])).sum(dim=1)
      logits = self.sig(dot)
      return logits

  '''
  Given confidence scores of true samples and false samples, return
  neg log likelihood
  '''
  def calc_loss(self, t_scores, f_scores, EPS=1e-6):
    pos_loss = -torch.log(t_scores+EPS).mean()
    neg_loss = -torch.log(1-f_scores+EPS).mean()
    return (1-self.neg_weight) * pos_loss + self.neg_weight * neg_loss

  '''
  Expects a list of true edges and false edges from each time
  step. Note: edge lists need not be the same length. Requires
  less preprocessing but doesn't utilize GPU/tensor ops as effectively
  as the batched fn
  '''
  def loss_fn(self, ts, fs, zs):
    tot_loss = torch.zeros((1))
    T = len(ts)

    for i in range(T):
      t_src, t_dst = ts[i]
      f_src, f_dst = fs[i]
      z = zs[i]
      tot_loss += self.calc_loss(self.decode(t_src, t_dst, z), self.decode(f_src, f_dst, z))
    return tot_loss.true_divide(T)

  '''
  Get scores for true/false embeddings to find ROC/AP scores.
  Essentially the same as loss_fn but with no NLL

  Returns logits unless as_probs is True
  '''
  def score_fn(self, ts, fs, zs, as_probs=False):
    tscores = []
    fscores = []
    T = len(ts)

    for i in range(T):
      t_src, t_dst = ts[i]
      f_src, f_dst = fs[i]
      z = zs[i]
      tscores.append(self.decode(t_src, t_dst, z))
      fscores.append(self.decode(f_src, f_dst, z))

    tscores = torch.cat(tscores, dim=0)
    fscores = torch.cat(fscores, dim=0)
    return tscores, fscores





"""# Run Me for Training"""

args = Namespace()
wandb.finish()

# @title Logging Sys.
wandb_project = '...'
args.wandb_log = True # @param {type:"boolean"}
args.debug = False # @param {type:"boolean"}
args.exp_type = '...' # @param {type:"string"}

#### args
args.task = "Ecls"

# @title Training Configs
args.lr = 0.001 # @param {type:"number"}
args.patience = 20 # @param {type:"slider", min:5, max:100, step:5}
args.max_epoch = 1000 # @param {type:"slider", min:2, max:1000, step:50}

args.MAX_DECREASE = 2

args.dropedge = 0 # @param {type:"slider", min:0, max:0.9, step:0.05}

seeds = [0,1,2,3,4]


args.predict = True # @param {type:"boolean"}
args.lstm = "LSTM"
args.norma = True # @param {type:"boolean"}

# @title Hyper Params Research
datas = ["bca", "bco", "mls", "uci"] # @param
d_hs = [8, 16, 32] 
models = ["Euler", "VGRNN", "DspGNN"]
if args.debug:
  d_hs = [4]
  args.max_epoch = 3


## Used only for DspGNN
args.dsp_edge_coef = 2 # @param # used multip coef
args.nb_spectral_supports = 6

args.masktype = "1hop"
for seed in seeds:
  args.seed = seed
  set_seed(seed)

  for d_h in d_hs:
    args.d_h = d_h

    for dataname in datas:
      args.data = dataname
      args = dataset_selection(args)

      for modelarchi in models:

        ## Most inside loop
        set_seed(seed)

        args.modelarchi = modelarchi
        if args.modelarchi != "DspGNN":
          args.spec_support = False
          args.motif = "..." # @param
        else:
          args.motif = "..." # @param
          args.spec_support = True
        args.modelname = args.exp_type + '-' + args.modelarchi + '-' + args.motif

        wandb.finish()
        data = args2eis(args)
        args.N = data.num_nodes
        args.d_x = args.N
        args.TEST_TS = args.t_test

        if args.modelarchi == 'DspGNN':
          train_edgelists = [data.tr(t) for t in range(data.T)]
          edges_hh, spectral_weights, eigen_vals = precompute_spectral_supports(args, train_edgelists, args.dsp_edge_coef, debug=False)
          model = DspGNN(args, data.x.size(1), args.d_h, lstm=args.lstm, dropedge=args.dropedge, eis=edges_hh, ews=spectral_weights)

        elif args.modelarchi == 'Euler':
          model = EulerGCN(data.x.size(1), args.d_h*2, args.d_h, lstm=args.lstm)

        elif args.modelarchi == 'VGRNN':
          model = VGRNN(data.x.size(1), args.d_h*2, args.d_h)


        if args.wandb_log:
            wandb.init(project=wandb_project
              , name=args.modelname + "_sd%d"%seed
              , config=vars(args))

        print(args)
        train(args, model, data, epochs=args.max_epoch, pred=args.predict, lr=args.lr)

