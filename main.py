# -*- coding: utf-8 -*-
"""# Installation"""

from google.colab import drive
drive.mount('/content/drive')

import os
os.chdir('...')
# !ls

import torch

def format_pytorch_version(version):
  return version.split('+')[0]

TORCH_version = torch.__version__
TORCH = format_pytorch_version(TORCH_version)

def format_cuda_version(version):
  return 'cu' + version.replace('.', '')

CUDA_version = torch.version.cuda
CUDA = format_cuda_version(CUDA_version)

# !pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-{TORCH}+{CUDA}.html
# !pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-{TORCH}+{CUDA}.html
# !pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-{TORCH}+{CUDA}.html
# !pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-{TORCH}+{CUDA}.html
# !pip install torch-geometric
# !pip install torch-geometric-temporal

# !pip install wandb
import wandb

"""## Libraries"""

'''commun libraries'''

import os
import collections
import time
from argparse import Namespace

import random
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, precision_recall_curve
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

import itertools
from datetime import datetime


'''torch related'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GATConv
from torch_geometric.nn.conv import GraphConv

def set_seed(seed):
  os.environ['PYTHONHASHSEED']=str(seed)
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  torch.use_deterministic_algorithms(True)

"""# Dataset Preprocess

## [ Funcs ] Node & Link Dataframes Preprocessing
"""

def coevognn_protocol_sampling_for_node_reg(links_df, gen_subsample_mask, degree_norm):
  return None

def data_egc_dt_snapshot_preprocess_dataframes(egc, egc_aggregate_timespan):
  egc['time'] = (egc['time'] - 1) / 86400 + 1 # timestamp to day int
  egc['time'] = egc['time'] / egc_aggregate_timespan  # aggregate to snapshots
  egc['time'] = egc['time'].astype(int)
  ## Mapping names to node idx
  combined_values = pd.concat([egc['src'], egc['dst']]).unique()
  combined_index = pd.factorize(combined_values)[0]
  mapping_dict = dict(zip(combined_values, combined_index))
  egc['src_id'] = egc['src'].map(mapping_dict)
  egc['dst_id'] = egc['dst'].map(mapping_dict)
  ## Normalizing features
  egc['label_nb'] = np.log10(egc['nb']+1)
  egc['label_nb'] = egc['label_nb']/egc['label_nb'].max()
  egc['label_value'] = np.log10(egc['value']+1)
  egc['label_value'] = egc['label_value']/egc['label_value'].max()
  return egc, mapping_dict

def data_egc_dt_prepare_dataframes(df, gen_subsample_mask=True, degree_norm=False):
  ## for unique node mapping
  unique_values = set(df['src_id']) | set(df['dst_id'])
  mapping = {}
  for idx, value in enumerate(unique_values):
    mapping[value] = idx
  nodes_df = pd.DataFrame(list(mapping.items()), columns=['node_name', 'node_index'])
  ## for edge dataframe
  df = df[['time', 'src_id', 'dst_id', 'label_nb', 'label_value']]
  df.columns = ['ts', 'src', 'dst', 'attr', 'attr2']
  links_df = df[['ts', 'src', 'dst', 'attr', 'attr2']].copy()
  links_df['src'] = links_df['src'].map(mapping)
  links_df['dst'] = links_df['dst'].map(mapping)
  nodes_attr = coevognn_protocol_sampling_for_node_reg(links_df, gen_subsample_mask, degree_norm)
  return nodes_df, links_df, nodes_attr

'''nodes: array(0..N-1); times: array(0..T-1); edgelists:  ;'''

def load_temporalgraphs(nodes_df, links_df):
  ''' Raw edgelist (un-indexed) for each time '''
  times = links_df.ts.sort_values().unique().astype(int) # unique timestamps
  nodes = nodes_df.node_index.sort_values().unique().astype(int) # unique node index

  edgelists = []
  edgelists_attr = []
  for t in range(len(times)):
    edges_t = links_df[links_df['ts'] == t]
    edgelists.append(torch.tensor(edges_t[['src', 'dst']].values).T)
    edgelists_attr.append(torch.tensor(edges_t.drop(['src', 'dst'], axis=1).values).T) # In case that attr is used for forward
  return nodes, times, edgelists, edgelists_attr

"""## [ Funcs ] Negative Sampling"""

'''
Following functions get_edges_ids, sample_edges, get_non_existing_edges are from EvolveGCN
ref; EvolveGCN
'''

def get_edges_ids(sp_idx, tot_nodes):
  return sp_idx[0]*tot_nodes + sp_idx[1]

def sample_edges(num_edges, idx, tot_nodes, existing_nodes=None, smart=True):
  if smart:
    from_id = np.random.choice(idx[0], size=num_edges, replace=True)
    to_id = np.random.choice(existing_nodes, size=num_edges, replace=True)
    if num_edges>1:
      edges = np.stack([from_id,to_id])
    else:
      edges = np.concatenate([from_id,to_id])
    return edges
  else:
    if num_edges > 1:
      edges = np.random.randint(0,tot_nodes,(2,num_edges))
    else:
      edges = np.random.randint(0,tot_nodes,(2,))
    return edges

def get_non_existing_edges(adj, number, tot_nodes, smart_sampling, existing_nodes=None):
  t0 = time.time()
  idx = adj.t().numpy()
  true_ids = get_edges_ids(idx,tot_nodes)
  true_ids = set(true_ids) # return to unique id of edge in the graph

  ## the maximum of possible negative edges
  ## would be all edges that don't exist between nodes that have edges
  num_edges = min(number, idx.shape[1] * (idx.shape[1]-1) - len(true_ids))
  ## Sample and convert edges to ids
  edges = sample_edges(num_edges*4, idx, tot_nodes, existing_nodes, smart_sampling)
  edge_ids = edges[0] * tot_nodes + edges[1]

  out_ids = set()
  num_sampled = 0
  sampled_indices = []
  for i in range(num_edges*4):
    eid = edge_ids[i]
    ## ignore if any of these conditions happen:
    ## edge already sampled, self connection, or real edge sampled
    if eid in out_ids or edges[0,i] == edges[1,i] or eid in true_ids:
      continue
    ## add the eid and the index to a list
    out_ids.add(eid)
    sampled_indices.append(i)
    num_sampled += 1
    ## if we have sampled enough edges break
    if num_sampled >= num_edges:
      break
  edges = edges[:,sampled_indices]
  edges = torch.tensor(edges).t()
  vals = torch.zeros(edges.size(0),dtype = torch.long)
  return edges # {'idx': edges, 'vals': vals}

'''Integrated util for final deterministic negative edge sampling'''
def negative_edge_sampling(edgelists, idx_end_train, times, tot_nodes,
                train_neg_sample_coef, seed):
  # np.random.seed(seed)
  sampled_negative_edges = []
  nodelists = []
  for t in times:
    e_t = edgelists[t]
    if e_t.shape[1] > 1:
      ## if train, ~10x negative sampling as in CoEvoGNN
      # if t in range(0, idx_end_train):
      if t < idx_end_train:
        negative_edges_nb = e_t.shape[1] * train_neg_sample_coef
      ## if valid and test
      if t >= idx_end_train:
        negative_edges_nb = e_t.shape[1]
      existing_nodes = e_t.T.flatten().unique()
      e_t_sym = torch.cat([e_t, torch.stack([e_t[1, :], e_t[0, :]], dim=0)], dim=1) ## convert to sym mode for NS
      nodelists.append(existing_nodes)
      current_negatives = get_non_existing_edges(e_t_sym.T, number=negative_edges_nb,
                    tot_nodes=tot_nodes, smart_sampling=True, existing_nodes=existing_nodes).T
      if current_negatives.shape[1] >= e_t.shape[1]: # shape: [2, |Et|]
        sampled_negative_edges.append(current_negatives)
      else: # if theres is no enough edges in the current snapshot
        sampled_negative_edges.append(get_non_existing_edges(e_t_sym.T, number=negative_edges_nb,
                    tot_nodes=tot_nodes, smart_sampling=False, existing_nodes=existing_nodes).T
        )
    else:
      #e_t.shape[0]
      sampled_negative_edges.append(current_negatives[:,0].reshape(2,1)) # when there is no edge, sample at least 1 negative from latest
  return sampled_negative_edges, nodelists

"""## [ Funcs ] Spectral Analysing"""

def calc_spectral_supports(A, sp_pente=0.125):
  num_nodes = A.shape[0]

  W = 1.0*A
  d = W.sum(axis=0)

  # normalized Laplacian matrix.
  dis=1/np.sqrt(d)
  dis[np.isinf(dis)]=0
  dis[np.isnan(dis)]=0
  D=np.diag(dis)
  nL=np.eye(D.shape[0])-(W.dot(D)).T.dot(D)
  V1,U1 = np.linalg.eigh(nL)
  V1[V1<0]=0

  # spec_gcn = (W.dot(D)).T.dot(D)
  # spec_l = [spec_gcn]
  spec_self = np.eye(D.shape[0]) # This is a placeholder filter
  spec_l = [spec_self]

  # low pass filter
  # print('Creating low pass kernel')
  dbb=(V1.max()-V1)/V1.max()
  db=dbb**3
  A0=U1.dot(np.diag(db).dot(U1.T))
  spec_l.append(A0)
  # sio.savemat('pubmedA0.mat',{'A0':A0})
  # print(V1)
  V_ = V1.copy()#V1[V1>0.001].copy()
  Vmin = V1.min()
  Vmax = V1.max()
  Vscale = Vmax - Vmin
  gms = [np.percentile(V_, 25), np.percentile(V_, 50), np.percentile(V_, 75)]
  print("Spectral Centers by GM: ", gms)#gm.means_)

  # band pass filters
  # ff=np.linspace(0,V1.max(),5)
  for f in gms:
    db4=np.exp(-(((V1-f))**2)*1)
    A2=U1.dot(np.diag(db4).dot(U1.T))
    A2[np.where(np.abs(A2)<0.001)]=0
    spec_l.append(A2)

  # high pass filter
  dbb=np.linspace(0,1,db.shape[0])
  A1=U1.dot(np.diag(dbb).dot(U1.T))
  spec_l.append(A1)

  supports = []
  for spectral_weights in spec_l:
    xsp = spectral_weights * (np.ones((num_nodes,num_nodes)) - np.eye(num_nodes))
    xsp = np.abs(xsp)
    min_val = np.min(xsp)
    max_val = np.max(xsp)
    xsp = (xsp - min_val) / (max_val - min_val + 0.00001)
    supports.append(xsp)
  supports = np.array(supports)
  return supports, V1


def build_adj_matrix(edges_tensor):
  unique_nodes = torch.unique(edges_tensor)
  node_to_idx = {node.item(): idx for idx, node in enumerate(unique_nodes)}
  n = len(unique_nodes)
  adj_matrix = np.zeros((n, n))
  for i in range(edges_tensor.size(1)):
    src, dst = edges_tensor[:, i].tolist()
    i, j = node_to_idx[src], node_to_idx[dst]
    adj_matrix[i][j] = 1
    adj_matrix[j][i] = 1  # to sym
  return adj_matrix, node_to_idx

def spectral_design(A):
  supports, V = calc_spectral_supports(A, sp_pente=0.125)
  adj_matrix_updated = torch.tensor(supports)
  assert len(adj_matrix_updated.shape) == 3
  return adj_matrix_updated, V

def extract_edge_weights(edges, adj_matrix_updated, node_to_idx):
  new_edge_weights = []
  for edge in edges.T:
    i, j = node_to_idx[edge[0].item()], node_to_idx[edge[1].item()]
    weight = adj_matrix_updated[:,i,j]  # get weights from spectral filtered adj
    new_edge_weights.append(weight)  # [edge, weight]
  return new_edge_weights

def extract_edge_weights_HH(edges, adj_matrix_updated, node_to_idx):
  adj_matrix_updated = torch.tensor(adj_matrix_updated)
  non_zero_indices = torch.nonzero(adj_matrix_updated, as_tuple=True)
  non_zero_values = adj_matrix_updated[non_zero_indices]
  medi_mask = adj_matrix_updated.sum(axis=0) > (torch.max(non_zero_values) * 0.2)#torch.quantile(non_zero_values, 0.2, interpolation='nearest')
  upper_triangle = torch.triu(medi_mask)
  indices = torch.nonzero(upper_triangle, as_tuple=False)
  i_positions = indices[:, 0]
  j_positions = indices[:, 1]

  keys = list(node_to_idx.values())
  values = list(node_to_idx.keys())

  i_list = torch.zeros_like(i_positions)
  for i in range(len(keys)):
    mask = (i_positions == keys[i])
    i_list[mask] = values[i]

  j_list = torch.zeros_like(j_positions)
  for i in range(len(keys)):
    mask = (j_positions == keys[i])
    j_list[mask] = values[i]

  edges = torch.stack([i_list, j_list])
  extracted_weight = adj_matrix_updated[:, i_positions, j_positions]
  return edges, extracted_weight

def precompute_spectral_supports(args, egc_df=None): # using args.data
  edgelists = data_processing_pipeline(args, egc_df, skip_ns=True)
  spectral_weights = []
  eigen_vals = []
  edges_hh = []
  totalt = len(edgelists)
  for t in range(totalt):
    # print(t)
    # print(edgelists[t])
    print("Current time: %d / %d"%(t,totalt))
    raw_adj, ni = build_adj_matrix(edgelists[t])
    adj_matrix_updated, eigen_val = spectral_design(raw_adj)
    if args.HH:
      e_t_hh, e_t_weights = extract_edge_weights_HH(edgelists[t], torch.tensor(adj_matrix_updated), ni)
      edges_hh.append(e_t_hh)
      spectral_weights.append(e_t_weights.float())
    else:
      e_t_weights = extract_edge_weights(edgelists[t], torch.tensor(adj_matrix_updated), ni)
    # print(e_t_hh)
    # print(e_t_weights)
    eigen_vals.append(torch.tensor(eigen_val).float())
  if args.HH:
    torch.save(edges_hh, 'spec/HH/%s_edge_hh.pt'%args.data)
    torch.save(spectral_weights, 'spec/HH/%s_edge_spec_weights.pt'%args.data)
    torch.save(eigen_vals, 'spec/HH/%s_snapshot_eigenval.pt'%args.data)
  else:
    torch.save(spectral_weights, 'spec/%s_edge_spec_weights.pt'%args.data)
    torch.save(eigen_vals, 'spec/%s_snapshot_eigenval.pt'%args.data)
  print("Computation Finished!")

"""# Model Archi

## [ Mods ] Models - CoEvoSAGE(-LSTM) - DspGNN
"""

"""
Model of CoEvoSage
"""

class CoEvoSAGE(nn.Module):
  def __init__(self, N, hid_emb_dim, K, num_sample_neighbors=20):
    super(CoEvoSAGE, self).__init__()
    ''' Hidden state of all nodes at t=0 '''
    self.K, self.N, self.hid_emb_dim, self.max_nbnb = K, N, hid_emb_dim, num_sample_neighbors
    self.sage_conv = SAGEConv(hid_emb_dim, hid_emb_dim, aggr='mean')
    ''' Weights: [d_hid + d_agg, d_hid, K], K is the number of agg timesteps '''
    self._agg_emb_dim = self.hid_emb_dim
    self.weight_W_ks = nn.ParameterList(
      [nn.Parameter(torch.FloatTensor(self.hid_emb_dim + self._agg_emb_dim, self.hid_emb_dim))
        for _ in range(self.K)])
    self.init_params()

  def init_params(self):
    for param in self.parameters():
      if len(param.shape) > 1:
        nn.init.xavier_uniform_(param)
      else:
        nn.init.constant_(param, 0.)

  def _step_k(self, H_k, edgelist_t): # self_feats = H_k; agg_feats = self.sage_conv(self_feats, edgelist_t)
    agg_feats = self.sage_conv(H_k, edgelist_t)
    return torch.cat((H_k, agg_feats), dim=1)

  def _step(self, H_K_prev, edgelists):
    ''' Transform previous K timesteps of H '''
    _H_t_ks = [torch.mm(self._step_k(H_K_prev[k], edgelists[k]), self.weight_W_ks[k])
                for k in range(len(H_K_prev))]
    ''' Fuse K transformed previous H'''
    H_t = F.relu(torch.sum(torch.stack(_H_t_ks), 0)) # Nonlinearity outside Sigma
    H_t = F.normalize(H_t, p=2, dim=1) # Row-wise L2 normalization
    return H_t

  def forward(self, H_K_prev, edgelists):
    ''' Generate H_t by catting previous H[t-K:t-1] (train and forecast)'''
    H_t = self._step(H_K_prev, edgelists)
    return torch.unsqueeze(H_t, 0) # [N,d] -> [1,N,d]



class CoEvoSAGELSTM(nn.Module):
  def __init__(self, N, d_h, K, num_sample_neighbors=20, aug_conv=False, convmod="SAGE"):
    super(CoEvoSAGELSTM, self).__init__()
    ''' Hidden state of all nodes at t=0 '''
    self.K, self.N, self.d_h, self.max_nbnb = K, N, d_h, num_sample_neighbors
    self.aug_conv = aug_conv
    if convmod == "SAGE":
      self.sage_conv1 = SAGEConv(d_h, d_h, aggr='mean')
    elif convmod == "GAT":
      self.sage_conv1 = GATConv(d_h, d_h, aggr='mean')
    self.lstm = nn.LSTM(input_size=d_h, hidden_size=d_h, batch_first=True)
    ''' Weights: [d_hid + d_agg, d_hid, K], K is the number of agg timesteps '''
    self._agg_emb_dim = self.d_h
    self.proj_linear = nn.Linear(2*d_h, d_h) # snapshot level proj
    self.weight_linear = nn.Linear(d_h, 1) # weight of current k
    self.init_params()

  def init_params(self):
    for param in self.parameters():
      if len(param.shape) > 1:
        nn.init.xavier_uniform_(param)
      else:
        nn.init.constant_(param, 0.)

  def _step_k(self, H_k, edgelist_t): # self_feats = H_k; agg_feats = self.sage_conv(self_feats, edgelist_t)
    agg_feats = self.sage_conv1(H_k, edgelist_t)
    return self.proj_linear(torch.cat((H_k, agg_feats), dim=1))

  def _step(self, H_K_prev, edgelists):
    ''' lstm solution '''
    _H_t_ks = [self._step_k(H_K_prev[k], edgelists[k]) for k in range(len(H_K_prev))]
    H_K_stack = torch.stack(_H_t_ks, dim=1) # [N, d] to [N, K, d]
    H_t, (hn, cn) = self.lstm(H_K_stack) # [N, K, d] for batch_first = True
    H_t = hn[-1] # last step only
    H_t = F.relu(H_t)
    # H_t = F.normalize(H_t, p=2, dim=1)
    return H_t

  def forward(self, H_K_prev, edgelists):
    ''' Generate H_t by catting previous H[t-K:t-1] (train and forecast)'''
    H_t = self._step(H_K_prev, edgelists)
    return torch.unsqueeze(H_t, 0) # [N,d] -> [1,N,d]



from torch_geometric_temporal.nn.recurrent import EvolveGCNH
class EvolveGCN(torch.nn.Module):
  def __init__(self, N, d, K):
    super(EvolveGCN, self).__init__()
    self.recurrent = EvolveGCNH(N, d)
    self.K = K
    # self.lin = nn.Linear(d,d)
    self.sigmoid = nn.Sigmoid()
    self.init_params()

  def init_params(self):
    for param in self.parameters():
      if len(param.shape) > 1:
        nn.init.xavier_uniform_(param)
      else:
        nn.init.constant_(param, 0.)

  def _step(self, H_K_prev, edgelists):
    ''' lstm solution '''
    h = H_K_prev[0]
    for k in range(len(H_K_prev)):
      h = F.relu(self.recurrent(h, edgelists[k]))
      h = F.normalize(h, p=2, dim=1)
      # h = self.sigmoid(h)
      return h

  def forward(self, H_K_prev, edgelists):
    ''' Generate H_t by catting previous H[t-K:t-1] (train and forecast)'''
    H_t = self._step(H_K_prev, edgelists)
    return torch.unsqueeze(H_t, 0) # [N,d] -> [1,N,d]

class DspGNN(nn.Module):
  def __init__(self, args, ablation_test_spectral_sup=False):
    super(DspGNN, self).__init__()
    ''' Hidden state of all nodes at t=0 '''
    self.K, self.N, self.d_h, self.nb_spec =\
      args.K, args.N, args.d_h, args.nb_spectral_supports
    self.spec_support, self.spec_eigenft = args.spec_support, args.spec_eigenft

    d_h = self.d_h
    self.sage_conv1 = GraphConv(d_h, d_h, aggr='mean', project=True, bias=False)
    self.sage_conv2 = GraphConv(d_h, d_h, aggr='mean', project=True, bias=False)
    self.sage_conv3 = GraphConv(d_h, d_h, aggr='mean', project=True, bias=False)
    self.sage_conv4 = GraphConv(d_h, d_h, aggr='mean', project=True, bias=False)
    self.sage_conv5 = GraphConv(d_h, d_h, aggr='mean', project=True, bias=False)
    self.sage_conv6 = GraphConv(d_h, d_h, aggr='mean', project=True, bias=False)
    if args.spec_eigenft:
      self.d_spec = args.spec_eigenft
      self.spec_proj = nn.Linear(self.d_spec, d_h)
      self.sage_combine_projection = nn.Linear(in_features=(self.nb_spec+2)*d_h, out_features=3*d_h)
    else:
      self.sage_combine_projection = nn.Linear(in_features=(self.nb_spec+1)*d_h, out_features=3*d_h)
      self.d_spec = False
    ''' Weights: [d_hid + d_agg, d_hid, K], K is the number of agg timesteps '''
    self.ablation_test_spectral_sup = ablation_test_spectral_sup
    # self.proj_linear = nn.Linear(4*d_h, 2*d_h)
    # self.lstm = nn.LSTM(input_size=2*d_h, hidden_size=d_h, batch_first=True)
    self.proj_linear = nn.Linear(4*d_h, d_h)
    self.lstm = nn.LSTM(input_size=d_h, hidden_size=d_h, batch_first=True)

    self.dropout = nn.Dropout(p=0.5)
    self.relu = nn.ReLU()
    self.init_params()

  def init_params(self):
    for param in self.parameters():
      if len(param.shape) > 1:
        nn.init.xavier_uniform_(param)
      else:
        nn.init.constant_(param, 0.)

  def _step_k(self, H_k, edgelist_t, ew_t=None, eig_t=None):
    if self.ablation_test_spectral_sup:
      ew_t = torch.ones_like(ew_t)
    if self.d_spec:
      spectral_density_projected = self.spec_proj(eig_t).repeat(self.N, 1)

    agg_feats1 = self.sage_conv1(H_k, edgelist_t, ew_t[0])
    agg_feats2 = self.sage_conv2(H_k, edgelist_t, ew_t[1])
    agg_feats3 = self.sage_conv3(H_k, edgelist_t, ew_t[2])
    agg_feats4 = self.sage_conv4(H_k, edgelist_t, ew_t[3])
    agg_feats5 = self.sage_conv5(H_k, edgelist_t, ew_t[4])
    agg_feats6 = self.sage_conv6(H_k, edgelist_t, ew_t[5])
    if self.d_spec:
      agg_feats_concat = torch.cat((H_k, agg_feats1, agg_feats2, agg_feats3,
                      agg_feats4, agg_feats5, agg_feats6,
                      spectral_density_projected), dim=1)
    else:
      agg_feats_concat = torch.cat((H_k, agg_feats1, agg_feats2, agg_feats3,
                      agg_feats4, agg_feats5, agg_feats6), dim=1)
    agg_feats_projected = self.sage_combine_projection(agg_feats_concat)
    agg_feats_projected = self.relu(agg_feats_projected)
    agg_feats_projected = self.dropout(agg_feats_projected)

    return self.proj_linear(torch.cat((H_k, agg_feats_projected), dim=1))

  def _step(self, H_K_prev, edgelists, spec_attr_prev):
    ''' lstm solution '''
    spec_weight, spec_eig = spec_attr_prev
    _H_t_ks = [self._step_k(H_K_prev[k], edgelists[k], spec_weight[k], spec_eig[k].float()) for k in range(len(H_K_prev))]
    H_K_stack = torch.stack(_H_t_ks, dim=1) # [N, d] to [N, K, d]
    H_t, (hn, cn) = self.lstm(H_K_stack) # [N, K, d] for batch_first = True
    H_t = hn[-1] # last step only
    H_t = self.relu(H_t)
    return H_t

  def forward(self, H_K_prev, edgelists, spec_attr_prev):
    ''' Generate H_t by catting previous H[t-K:t-1] (train and forecast)'''
    H_t = self._step(H_K_prev, edgelists, spec_attr_prev)
    return torch.unsqueeze(H_t, 0) # [N,d] -> [1,N,d]

"""## [ Mods ] Commun Modules"""

class MLP_Regressor(nn.Module):
  """2-layers Regression with GeLU"""
  def __init__(self, hidden_dims, output_dims, expansion_factor=0.5, dropout=0.5):
    super(MLP_Regressor, self).__init__()
    self.dh = hidden_dims
    self.do = output_dims
    self.expansion_factor = expansion_factor
    self.dropout = dropout
    self.lin0 = nn.Linear(hidden_dims, int(expansion_factor * hidden_dims))
    self.lin1 = nn.Linear(int(expansion_factor * hidden_dims), output_dims)
    # self.reset_parameters()

  def forward(self, x, ts):
    x = self.lin0(x)
    x = F.gelu(x)
    x = F.dropout(x, p=self.dropout, training=self.training)
    x = self.lin1(x)
    x = F.dropout(x, p=self.dropout, training=self.training)
    return x


class Linear_Regressor(nn.Module):
  def __init__(self, hidden_dims, output_dims):
    super(Linear_Regressor, self).__init__()
    self.dh = hidden_dims
    self.do = output_dims
    self.weight_M = nn.Parameter(
        torch.FloatTensor(self.dh, self.do))
    self.init_params()
    self.relu = torch.nn.ReLU()

  def init_params(self):
    for param in self.parameters():
      nn.init.xavier_uniform_(param)

  def forward(self, x, ts):
    # if self.eval:
    #   return self.relu(torch.matmul(x, self.weight_M))
    # else:
    return torch.matmul(x, self.weight_M)

class LinkPredictor(nn.Module):
  def __init__(self, args=None):
    super(LinkPredictor, self).__init__()
    self.parametric = args.e_parametric if args is not None else False
    if not self.parametric: # non-parametric
      self.sim = nn.CosineSimilarity(dim=1, eps=1e-6)
    elif self.parametric: # non-parametric
      self.sim = nn.Linear(args.d_h*2, 1, bias=False)
      # self.sim = MLP_Regressor(args.d_h*2, 1, 0.5, 0.2)

  def pos_neg_sim(self, H_t, real_edges_t, negative_edges_t):
    if not self.parametric:
      real_sim = self.sim(H_t[real_edges_t[0]], H_t[real_edges_t[1]])
      ns_sim = self.sim(H_t[negative_edges_t[0]], H_t[negative_edges_t[1]])
    else:
      real_sim = self.sim(torch.cat((H_t[real_edges_t[0]], H_t[real_edges_t[1]]), dim=1))
      ns_sim = self.sim(torch.cat((H_t[negative_edges_t[0]], H_t[negative_edges_t[1]]), dim=1))
    return real_sim, ns_sim

  def loss_ns(self, node_vs, H_t, real_edges_t, negative_edges_t, train_neg_sample_coef=10):
    assert len(H_t.shape) == 2
    ## Loss of possitive and negative edges
    real_sim, ns_sim = self.pos_neg_sim(H_t, real_edges_t, negative_edges_t)
    all_sim = torch.cat([real_sim, ns_sim])
    all_labels = torch.cat([torch.ones_like(real_sim), torch.zeros_like(ns_sim)])
    loss = F.binary_cross_entropy(
      torch.clamp(all_sim, min=0, max=1),
      torch.clamp(all_labels, min=0, max=1)
      )
    return loss

"""## [ Funcs ] Snapshot-level Inference"""

def snapshot_link_pred(H_t, real_edges_t, negative_edges_t, e_pred, verbose=False):
  assert len(H_t.shape) == 3
  real_sim, ns_sim = e_pred.pos_neg_sim(H_t[0], real_edges_t, negative_edges_t)
  preds = torch.cat((real_sim, ns_sim))
  reals = torch.cat((torch.ones_like(real_sim), torch.zeros_like(ns_sim)))
  return reals, preds

def snapshot_reg_process(H_t, current_df, args, mode='edge'):
  '''H_t size = [1, N, d]'''
  inputs = []
  true_attrs = []
  assert mode=='edge' ## edge regression
  ts = current_df['ts'].astype(int).values
  src = current_df['src'].astype(int).values
  dst = current_df['dst'].astype(int).values
  attr = current_df['attr'].values
  H_inputs = torch.cat((H_t[0][src], H_t[0][dst]), axis=1)
  assert args.reg_e_targets == 1
  true_attrs = current_df['attr'].values.reshape(-1, 1)
  true_attrs = torch.tensor(true_attrs).float()
  return H_inputs, true_attrs

"""## [ Funcs ] Metrics Calcul"""

def eval_attr(X_ts_pred, X_ts_real, N, args):
  '''Calculate time step by time step, return array of shape [T_eval]'''
  maes, rmses = [], []
  ''' Sum absolute/squared errors of X - X^hat '''
  loss_mae = nn.L1Loss(reduction='sum')
  loss_mse = nn.MSELoss(reduction='sum')
  T_eval = len(X_ts_pred)#.shape[0]
  numel_t = N * 2

  for t in range(T_eval):
    _num_samples = np.prod(X_ts_real[t].shape)
    assert args.task == "ereg"
    _mae = loss_mae(X_ts_pred[t], X_ts_real[t]).item() / _num_samples
    _mse = loss_mse(X_ts_pred[t], X_ts_real[t]).item() / _num_samples
    maes.append(_mae)
    rmses.append(np.sqrt(_mse))
  return maes, rmses

def eval_stru(G_ts_pred, G_ts_real):
  T_eval = len(G_ts_pred)
  accs, f1s, aucs = [], [], []
  for t in range(T_eval):
    pred = np.array(G_ts_pred[t] > np.median(G_ts_pred[t])).astype(int)
    real = G_ts_real[t]
    accs.append(accuracy_score(real, pred))
    aucs.append(roc_auc_score(real, pred))
    f1s.append(f1_score(real, pred))
  return accs, f1s, aucs

def forward_inference(H_K_prev, elist_prev, current_links_df, current_nodes_df,
            t, enc, reg, args, spec_attr_prev=None):
  '''forward pass'''
  if args.spec_support:
    H_t = enc(H_K_prev, elist_prev, spec_attr_prev)  # Get hidden states for current time step using last K time step
  else:
    H_t = enc(H_K_prev, elist_prev)  # Get hidden states for current time step using last K time step
  '''predict attr'''
  if args.task == "ereg":
    inputs_V = None
    inputs_E, true_attrs = snapshot_reg_process(H_t, current_links_df, args, mode='edge')
    predicted_attrs = reg(inputs_E, t)
  if args.task == "vreg":
    inputs_V, true_attrs = snapshot_reg_process(H_t, current_nodes_df, args, mode='node')
    inputs_E, _ = snapshot_reg_process(H_t, current_links_df, args, mode='edge') ### inputs_E will be used for link CLF
    predicted_attrs = reg(inputs_V, t)
  return H_t, inputs_V, inputs_E, true_attrs, predicted_attrs

def evaluation(start_ts, end_ts, H_K_prev, links_df, nodes_attr, edgelists,
        negative_edgelists, enc, reg, eclf, args, additional_attr=None):
  G_ts_real = []
  G_ts_pred = []
  X_ts_real = []
  X_ts_pred = []
  N = args.N
  for t in range(start_ts, end_ts):
    ''' Forward & Regression '''
    ## elist_prev, ts, args.K
    elist_prev = edgelists[t-args.K:t]
    if args.spec_support:
      spec_attr_prev = (additional_attr[0][t-args.K:t],
                additional_attr[1][t-args.K:t]) # Spectral weights, Eigenvalues
    else:
      spec_attr_prev = None
    H_t, inputs_V, inputs_E, true_attrs, predicted_attrs =\
      forward_inference(H_K_prev, elist_prev, links_df[links_df['ts']==t], nodes_attr[nodes_attr['ts']==t],
                t, enc, reg, args, spec_attr_prev)

    H_K_prev = torch.cat((H_K_prev[1:], H_t.detach()), dim=0)
    X_ts_real.append(true_attrs)
    X_ts_pred.append(predicted_attrs)

    ''' Edge Prediction '''
    reals_t, preds_t = snapshot_link_pred(H_t, edgelists[t], negative_edgelists[t], eclf)
    G_ts_real.append(reals_t)
    G_ts_pred.append(preds_t)
  if args.global_mean: # Combine all prediction into one snapshot
    accs, aucs, f1s = eval_stru(torch.cat(G_ts_pred).unsqueeze(0), torch.cat(G_ts_real).unsqueeze(0))
    maes, rmses = eval_attr(torch.cat(X_ts_pred).unsqueeze(0), torch.cat(X_ts_real).unsqueeze(0), N, args)
  else: # Mean of each snapshot
    accs, aucs, f1s = eval_stru(G_ts_pred, G_ts_real)
    maes, rmses = eval_attr(X_ts_pred, X_ts_real, N, args)
  return accs, aucs, f1s, maes, rmses, H_K_prev

def customize_score_func(rmse, f1, args):
  return rmse

"""# Main Func"""

def data_processing_pipeline(args, egc_df=None, skip_ns=False):
  """
  Processing
  """
  set_seed(args.seed)
  # if args.data[:2] == "bc":
  #   rawdata = data_bc_readcsv_norm(args.data_path, time_win_aggr=args.T)
  # if args.data == "mls":
  #   rawdata = data_bc_readcsv_norm(args.data_path, time_win_aggr=args.T, skip_header=True)
  # nodes_df, links_df, nodes_attr = data_bc_prepare_dataframes(rawdata, args.vreg_gen_subsample_mask, args.vreg_degree_norm)
  if args.data[:3] == "egc":
    nodes_df, links_df, nodes_attr = data_egc_dt_prepare_dataframes(egc_df, args.vreg_gen_subsample_mask, args.vreg_degree_norm)

  ## Extracting Edgelist per snapshot
  nodes, times, edgelists, edgelists_attr = load_temporalgraphs(nodes_df, links_df)
  args.N = len(nodes)
  if skip_ns:
    return edgelists
  ## Sampling negative edges and active nodelist per snapshot
  negative_edgelists, nodelists = negative_edge_sampling(edgelists, idx_end_train=args.idx_end_train, times=times, tot_nodes=args.N,
                  train_neg_sample_coef=args.train_neg_sample_coef, seed=args.seed)
  return args, nodes_df, links_df, nodes_attr, nodes, times, edgelists, edgelists_attr, negative_edgelists, nodelists


def train_test_pipeline(args, nodes_df, links_df, nodes_attr, nodes, times, edgelists, edgelists_attr, negative_edgelists, nodelists):
  """
  Training
  """
  ## Train test looping
  set_seed(args.seed)
  print(args)
  if args.wblog:
    wandb.init(project=args.wblog_project
      , name=args.data + '_' + args.wblog_type + '_' + args.task + '_' + args.model + "_sd%d"%args.seed
      , config=vars(args))

  ''' Load Spectral Weights and eigenvalues '''
  if args.spec_support:
    edgelists_attr, edge_weights, snapshot_eig = edgelists_attr
    additional_attr = (edge_weights, snapshot_eig)
  else:
    additional_attr = None

  if args.model == "CoEvoSAGELSTM":
    if args.convmod == "GAT": ## GAT
      enc = CoEvoSAGELSTM(N=args.N, d_h=args.d_h,
            K=args.K, num_sample_neighbors=args.num_sample_neighbors,
            convmod="GAT")
    else: ## GraphSAGE
      enc = CoEvoSAGELSTM(N=args.N, d_h=args.d_h,
            K=args.K, num_sample_neighbors=args.num_sample_neighbors)
  elif args.model == "CoEvoSage":
    enc = CoEvoSAGE(N=args.N, hid_emb_dim=args.d_h,
          K=args.K, num_sample_neighbors=args.num_sample_neighbors)
  elif args.model == "DspGNN":
    enc = DspGNN(args, ablation_test_spectral_sup=args.ablation_test_spectral_sup)
  elif args.model == "EvolveGCN":
    enc = EvolveGCN(args.N, args.d_h, args.K)


  if args.regressor == "1L":
    Regressor = Linear_Regressor
  elif args.regressor == "2L":
    Regressor = MLP_Regressor

  if args.task == "ereg":
    reg = Regressor(hidden_dims=2*args.d_h, output_dims=args.reg_e_targets)
  elif args.task == "vreg":
    reg = Regressor(hidden_dims=args.d_h, output_dims=args.reg_v_targets)

  eclf = LinkPredictor()  # Non-parametric model
  models = [enc, reg, eclf]
  params = [param for model in models for param in model.parameters() if param.requires_grad]
  optimizer = torch.optim.SGD(params, lr=args.lr, weight_decay=args.wdecay)

  print('\nTraining CoEvoGNN ({:,})...'.format(args.epochs), flush=True)
  best_syn_score = customize_score_func(10000, 0, args)
  best_ep = 0
  best_val_f1 = 0
  best_val_rmse = 100000
  best_train_loss = 100000
  for ep in range(args.epochs):
    s_t = time.time()
    # set_seed(args.seed)
    H_K_prev = torch.rand((args.K, args.N, args.d_h)) # np.load(args.H_0_npf)
    for model in models:
      model.train()
    for param in params:
      param.requires_grad = True

    ''' Train: Time Step Loop '''
    loss_T = 0
    loss_attr_T = 0
    loss_stru_T = 0
    for t in range(args.idx_start_train+args.K+1, args.idx_end_train):  # Loop through each time step
      # pass
      ''' Initializations '''
      optimizer.zero_grad()

      ''' Forward & Regression '''
      elist_prev = edgelists[t-args.K:t]

      if args.spec_support:
        spec_attr_prev = (additional_attr[0][t-args.K:t],
                  additional_attr[1][t-args.K:t]) # Spectral weights, Eigenvalues
      else:
        spec_attr_prev = None

      H_t, inputs_V, inputs_E, true_attrs, pred_attrs =\
        forward_inference(H_K_prev, elist_prev, links_df[links_df['ts']==t], nodes_attr[nodes_attr['ts']==t],
                  t, enc, reg, args, spec_attr_prev)
      H_K_prev = torch.cat((H_K_prev[1:], H_t.detach()), dim=0) # Update the previous K hidden states

      ''' Loss '''
      loss_attr = args.loss_alpha * F.mse_loss(pred_attrs, true_attrs) / args.t_train
      loss_stru = eclf.loss_ns(nodelists[t], H_t[0], edgelists[t], negative_edgelists[t], train_neg_sample_coef=10) / args.t_train
      if args.loss_accum:
        loss_attr_T += loss_attr
        loss_stru_T += loss_stru
      else:
        loss = loss_attr + loss_stru
        loss.backward()  # Perform backward pass using total loss
        loss_T += loss.item()
        loss_attr_T += loss_attr.item()
        loss_stru_T += loss_stru.item()

        for model in models:
          nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()
    if args.loss_accum:
      loss_T = loss_attr_T + loss_stru_T
      loss_T.backward()  # Perform backward pass using total loss
      for model in models:
        nn.utils.clip_grad_norm_(model.parameters(), 5)
      optimizer.step()
      new_train_loss = loss_T.item()
      loss_attr_T = loss_attr_T.item()
      loss_stru_T = loss_stru_T.item()
    else:
      new_train_loss = loss_T



    ''' Valid & Test'''
    for model in models:
      model.eval()
    for param in params:
      param.requires_grad = False

    # set_seed(args.seed)
    val_accs, val_aucs, val_f1s, val_maes, val_rmses, H_K_prev = evaluation(
      args.idx_end_train, args.idx_end_valid, H_K_prev, links_df, nodes_attr, edgelists, negative_edgelists, enc, reg, eclf, args,
      additional_attr)
    print("VALID Epoch : %d | MAE %.4f   RMSE %.4f   "%(ep, np.mean(val_maes), np.mean(val_rmses)),
          "ACC %.4f    F1 %.4f    AUC %.4f"%(np.mean(val_accs), np.mean(val_f1s), np.mean(val_aucs)))

    ''' Test '''
    tst_accs, tst_aucs, tst_f1s, tst_maes, tst_rmses, _ = evaluation(
      args.idx_end_valid, args.idx_end_test-1, H_K_prev, links_df, nodes_attr, edgelists, negative_edgelists, enc, reg, eclf, args,
      additional_attr)
    print("TEST  Epoch : %d | MAE %.4f   RMSE %.4f   "%(ep, np.mean(tst_maes), np.mean(tst_rmses)),
          "ACC %.4f    F1 %.4f    AUC %.4f"%(np.mean(tst_accs), np.mean(tst_f1s), np.mean(tst_aucs)))


    ''' Epoch summary '''
    e_t = time.time()
    _ep_time = e_t - s_t
    print('Ep.{:03d} Train Loss: {:6.3f} = {:6.3f} + {:6.3f} ({:5.1f} secs)'.format(ep + 1, loss_T, loss_attr_T, loss_stru_T, _ep_time), flush=True)
    if args.wblog:
      wandb.log({"epoch": ep, "train loss": new_train_loss, "last train loss reg": loss_attr, "last train loss clf": loss_stru,
        "valid rmse": np.mean(val_rmses), "valid f1": np.mean(val_f1s), "valid acc": np.mean(val_accs),
        "test rmse": np.mean(tst_rmses), "test f1": np.mean(tst_f1s), "test acc": np.mean(tst_accs),
        "epoch time": _ep_time})


    ''' Print snapshot-level details '''
    if args.snapshot_level_verbose:
      print("DETAILS")
      print("VALID Epoch-level-RMSE: ", val_rmses, "\nVALID Epoch-level-F1  : ", val_f1s)
      print("TEST  Epoch-level-RMSE: ", tst_rmses, "\nTEST  Epoch-level-F1  : ", tst_f1s)

    # print(H_t)

    ''' Check valid if is best epoch'''
    new_val_rmse = np.mean(val_rmses)
    new_val_f1 = np.mean(val_f1s)
    new_syn_score = customize_score_func(new_val_rmse, new_val_f1, args)
    if new_syn_score <= best_syn_score and new_train_loss <= best_train_loss:
    # if new_val_rmse <= best_val_rmse and new_train_loss <= best_train_loss and new_val_f1 >= best_val_f1:
      print("!!!!!!!! best epo !!!!!!!!")
      bests = [ep, np.mean(tst_maes), np.mean(tst_rmses), np.mean(tst_accs), np.mean(tst_f1s), np.mean(tst_aucs)]
      best_val_f1 = new_val_f1
      best_val_rmse = new_val_rmse
      best_train_loss = new_train_loss
      best_ep = ep
      best_syn_score = new_syn_score
    print("/////////////////////////////////////////")

    if (ep - best_ep >= args.earlystop) or (ep >= args.epochs-1):
      if args.wblog:
        wandb.summary['Final Valid best RMSE'] = best_val_rmse
        wandb.summary['Final Valid best F1'] = best_val_f1
        wandb.summary['Final Test MAE'] = bests[1]
        wandb.summary['Final Test RMSE'] = bests[2]
        wandb.summary['Final Test ACC'] = bests[3]
        wandb.summary['Final Test AUC'] = bests[4]
        wandb.summary['Final Test F1'] = bests[5]
        wandb.finish()
      print("!!!!!!!! EARLY STOP !!!!!!!!")
      break


  print("Main Program Finished")
  print("BEST  Epoch : %d | MAE %.4f   RMSE %.4f    ACC %.4f    F1 %.4f    AUC %.4f"%tuple(bests))
  return bests, (enc, reg, eclf)

"""# Arguments"""

'''Dataset Selection'''
def dataset_selection(args):
  egc_df = None
  args.t_0 = 0

  if args.data == "bca":
    args.data_path = "data/soc-sign-bitcoinalpha.csv"
    args.T = 137
    args.t_train = 95
    args.t_valid = 14
    args.t_test = 28
    args.reg_e_targets = 1

  elif args.data == "bco":
    args.data_path = "data/soc-sign-bitcoinotc.csv"
    args.T = 136
    args.t_train = 95
    args.t_valid = 13
    args.t_test = 28
    args.reg_e_targets = 1

  elif args.data == "egc_dt":
    # args.T = int(args.T/7)
    args.data_path = "data/egc_full_combined.csv"
    args.egc_aggregate_timespan = 7
    if args.egc_aggregate_timespan == 7:
      args.T = 131
      args.t_train = 92
      args.t_valid = 13
      args.t_test = 26
      args.reg_e_targets = 2

  elif args.data == "mls":
    args.data_path = "data/movielens-latest-small-ratings.csv"
    args.T = 90
    args.t_train = 63
    args.t_valid = 9
    args.t_test = 18
    args.reg_e_targets = 1

  args.idx_start_train = 1 + args.t_0
  args.idx_end_train = args.idx_start_train + args.t_train
  args.idx_end_valid = args.idx_end_train + args.t_valid
  args.idx_end_test = args.idx_end_valid + args.t_test

  print("Current Dataset: %s \nTotal Snapshots: %d \nTrain Snapshots: %d \nValid Snapshots: %d \nTest  Snapshots: %d \n"%(
      args.data, args.T, args.t_train, args.t_valid, args.t_test
  ))
  return args

### log_type = debug, base, inno, tune

def commum_args_CoEvoSage(args, runs, log_type, wblog, e_param):
  '''Program Related'''
  args.exps = runs
  args.snapshot_level_verbose = True
  args.vreg_gen_subsample_mask = False#True#False
  args.t_0 = 0

  '''Model Related'''
  args.model = "CoEvoSage"
  args.K = 2
  args.d_h = 64
  args.regressor = "1L"
  args.loss_accum = True #False

  '''Training Related'''
  args.epochs = 200
  args.lr = 0.01
  args.wdecay = 1e-6

  args.e_parametric = False
  args.loss_alpha = 100

  args.earlystop = 10
  args.global_mean = False
  args.num_sample_neighbors = 20

  '''logging_sys'''
  args.wblog = wblog
  args.wblog_type = log_type
  args.wblog_project = "EGC-dt"
  if log_type == "debug":
    args.epochs = 5

  '''Execution'''
  ## logging
  if args.wblog:
    wandb.login()
  return args

def args_baseline_bc_coevosage(data="bca", task="ereg" ,runs=2, log_type='debug', wblog=True, e_param=False):
  '''Args'''
  args = Namespace()
  args.data = data
  args.task = task
  args = commum_args_CoEvoSage(args, runs, log_type, wblog, e_param)

  args.vreg_degree_norm = False
  args.train_neg_sample_coef = 10 #related to CoEvoGNN rw_R rw_Q
  return args

# def args_baseline_egcdt_coevosage(data="egc_dt", task="ereg", runs=1, log_type='debug', wblog=True, e_param=False):
#   '''Args'''
#   args = Namespace()
#   args.data = data
#   args.task = task
#   args = commum_args_CoEvoSage(args, runs, log_type, wblog, e_param)

#   args.egc_aggregate_timespan = 7
#   args.vreg_degree_norm = "log"
#   args.train_neg_sample_coef = 2 #related to CoEvoGNN rw_R rw_Q
#   return args

def args_baseline_bc_dspgnn(data="bca", task="ereg" ,runs=2, log_type='debug', wblog=True, e_param=False):
  '''Args'''
  args = Namespace()
  args.data = data
  args.task = task
  args = commum_args_DspGNN(args, runs, log_type, wblog, e_param)

  args.vreg_degree_norm = False
  args.train_neg_sample_coef = 10 #related to CoEvoGNN rw_R rw_Q
  return args

# def args_baseline_egcdt_dspgnn(data="egc_dt", task="ereg", runs=1, log_type='debug', wblog=True, e_param=False):
#   '''Args'''
#   args = Namespace()
#   args.data = data
#   args.task = task
#   args = commum_args_DspGNN(args, runs, log_type, wblog, e_param)

#   args.egc_aggregate_timespan = 7
#   args.vreg_degree_norm = "log"
#   args.earlystop = 10 # This model converges quickly
#   args.train_neg_sample_coef = 2 #related to CoEvoGNN rw_R rw_Q
#   return args

def commum_args_DspGNN(args, runs, log_type, wblog, e_param):
  '''Program Related'''
  args.exps = runs
  args.snapshot_level_verbose = True
  args.vreg_gen_subsample_mask = True#False
  args.t_0 = 0

  '''Model Related'''
  args.model = "DspGNN"
  args.K = 6
  args.d_h = 64
  args.regressor = "1L"
  args.loss_accum = False
  args.spec_support = True
  args.spec_eigenft = False
  args.ablation_test_spectral_sup = False
  args.nb_spectral_supports = 6

  '''Training Related'''
  args.epochs = 100
  args.lr = 0.003
  args.wdecay = 1e-6

  args.e_parametric = False
  args.loss_alpha = 100

  args.earlystop = 10
  args.global_mean = False
  args.num_sample_neighbors = 20

  '''logging_sys'''
  args.wblog = wblog
  args.wblog_type = log_type
  args.wblog_project = "EGC-dt"
  if log_type == "debug":
    args.epochs = 5

  '''Execution'''
  ## logging
  if args.wblog:
    wandb.login()
  return args

def runmain(args):
  bests_l = []
  args = dataset_selection(args)

  if args.model == "DspGNN":
    load_edge_weights = torch.load('spec/%s_edge_spec_weights.pt'%args.data)
    load_snapshot_eig = torch.load('spec/%s_snapshot_eigenval.pt'%args.data)
    if args.HH:
      load_edge_weights = torch.load('spec/HH/%s_edge_spec_weights.pt'%args.data)

  for seed in range(args.exps):
    args.seed = seed
    set_seed(args.seed)
    args, nodes_df, links_df, nodes_attr, nodes, times, edgelists, edgelists_attr, negative_edgelists, nodelists = data_processing_pipeline(args)
    if args.HH:
      edgelists = torch.load('spec/HH/%s_edge_hh.pt'%args.data)
    ## DspGNN Spectral supported edge weight
    if args.model == "DspGNN":
      if args.spec_support:
        edgelists_attr = (edgelists_attr, load_edge_weights, load_snapshot_eig)
    bests, trained_models = train_test_pipeline(args, nodes_df, links_df, nodes_attr, nodes, times, edgelists, edgelists_attr, negative_edgelists, nodelists)
    bests_l.append(bests)
  return bests, trained_models

"""# .< Start Testing >."""

wandb.login(key = "...")

"""# Final Test Runned"""

debuging = False

logtype = '''...'''
for d in ["bca"]: # "bco" #, "mls"
  runs = 5
  wblog = True
  args = args_baseline_bc_dspgnn(data=d, task="ereg",
                  runs=runs, log_type=logtype, wblog=wblog)
  args.K, args.lr = [6, 0.003]
  args.spec_support = True
  args.loss_accum = False
  args.ablation_test_spectral_sup = False
  args.snapshot_level_verbose = False

  print("Debugging ?", debuging)
  if debuging:
    args.epochs = 10 #20
    args.debug = True
    args.seed = 42

  args.HH = True

  runmain(args)

wandb.finish()