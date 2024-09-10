import torch
import torch.nn.functional as F
import yaml
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
import numpy as np
from tqdm import tqdm
import os,inspect, random, pickle
from numpy import random,mat
import scipy.sparse
import argparse
import torch
import torch.nn as nn
from torch.optim import Adam
import os
import numpy as np
import time
import datetime
import path
import shutil
import config


def gnn_init(params,G):
    emb_model_name = params['gridemb_model']
    n_emb_layers = params['n_gridemb_layers']

    d_emb_hidden = params['d_gridemb_hidden']

    d_emb_hidden[0] = G.number_of_nodes()
    if emb_model_name == 'GAT':
        n_emb_head = params['n_gridemb_head']
        return GAT(n_emb_layers, d_emb_hidden, n_emb_head)
    else:
        return GCN(n_emb_layers, d_emb_hidden)

class VariationalGCN(torch.nn.Module):
    def __init__(self,encoder):
        super(VariationalGCN, self).__init__()
        self.encoder = encoder
        hidden_dim1, hidden_dim2 = encoder.layers[-1].out_channels, encoder.layers[-1].out_channels // 2
        self.mu = GCNConv(hidden_dim1, hidden_dim2)
        self.std = GCNConv(hidden_dim1, hidden_dim2)
    def forward(self,x,edge_index,edge_attr):
        x = self.encoder(x,edge_index,edge_attr).relu()
        return self.mu(x,edge_index,edge_attr),self.std(x,edge_index,edge_attr)
class GCN(torch.nn.Module):
    def __init__(self, layers, hidden):
        super().__init__()
        self.name = 'GCN'
        self.layers = nn.ModuleList()
        for layer  in range(layers):
           self.layers.append(GCNConv(hidden[layer], hidden[layer+1]))

    def forward(self, x,edge_index,edge_attr):

        num_layers  =  len(self.layers)
        for i in range(num_layers-1):
            x = self.layers[i](x,edge_index,edge_attr)
            x = F.relu(x)
            x = F.dropout(x, training=self.training)

        return self.layers[-1](x, edge_index, edge_attr)

    # def gae_forward(self, x,edge_index):
    #
    #     num_layers  =  len(self.layers)
    #     for i in range(num_layers-1):
    #         x = self.layers[i](x,edge_index)
    #         x = F.relu(x)
    #         x = F.dropout(x, training=self.training)
    #
    #     return x, self.layers[-1](x, edge_index)
    def saveemb(self, data,f):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        # x = self.conv2(x, edge_index)

        X=F.log_softmax(x, dim=1)
        torch.save(X.cpu(),f)


    def load(self,param_dir):
        model_dict = torch.load(param_dir)
        self.layers.state_dict = model_dict
    def save(self,param_dir):
        torch.save(self.layers.state_dict(),param_dir)

    def saveemb(self,x,edge_index,f):
        X = self.forward(x,edge_index)
        torch.save(X.cpu(),f)

    def loademb(self,f):
        X = torch.load(f).cuda()
        return X