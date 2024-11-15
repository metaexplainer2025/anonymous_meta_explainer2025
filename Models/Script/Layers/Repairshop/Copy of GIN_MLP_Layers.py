from torch_geometric.utils import dropout
from torch_geometric.loader import DataLoader
import torch
import torch.nn as nn
from torch.autograd import Variable
import math
import torch_geometric
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch_geometric.utils.convert import to_scipy_sparse_matrix
import networkx as nx
import numpy as np
from torch_geometric.nn import GCNConv
import sys 
from torch_geometric.datasets import TUDataset
from torch_geometric.nn import global_add_pool


class GIN_MLPs(nn.Module):
    def __init__(self, num_slp_layers, mlp_input_dim, mlp_hid_dim, mlp_output_dim, mlp_act_fun, Bias):
        super(GIN_MLPs, self).__init__()
        self.mlp_input_dim = mlp_input_dim
        self.mlp_hid_dim = mlp_hid_dim
        self.mlp_output_dim = mlp_output_dim
        self.num_slp_layers = num_slp_layers
        self.Bias = Bias

        if mlp_act_fun == 'ReLu':
            self.mlp_act_fun = F.relu
        elif mlp_act_fun == 'eLu':
            self.mlp_act_fun = nn.functional.elu
        elif mlp_act_fun == 'tanh':
            self.mlp_act_fun = torch.tanh

        self.gin_mlp_layers = torch.nn.ModuleList()
        self.gin_batch_normalization = torch.nn.ModuleList()


        if self.num_slp_layers == 1:
            self.gin_mlp_layers.append(nn.Linear(in_features=self.mlp_input_dim, out_features=self.mlp_output_dim, bias=self.Bias))
            self.gin_batch_normalization.append(nn.BatchNorm1d(self.mlp_output_dim))
        elif self.num_slp_layers > 1:
            for i in range(self.num_slp_layers):
                if i == 0:
                    self.gin_mlp_layers.append(nn.Linear(in_features=self.mlp_input_dim, out_features=self.mlp_hid_dim, bias=self.Bias))
                    self.gin_batch_normalization.append(nn.BatchNorm1d(num_features=self.mlp_hid_dim))
                elif i == self.num_slp_layers-1:
                    self.gin_mlp_layers.append(nn.Linear(in_features=self.mlp_hid_dim, out_features=self.mlp_output_dim, bias=self.Bias))
                    self.gin_batch_normalization.append(nn.BatchNorm1d(self.mlp_output_dim))
                elif 0 < i < self.num_slp_layers-1:
                    self.gin_mlp_layers.append(nn.Linear(in_features=self.mlp_hid_dim, out_features=self.mlp_hid_dim, bias=self.Bias))
                    self.gin_batch_normalization.append(nn.BatchNorm1d(self.mlp_hid_dim))
        else:
            print("please enter layer config")

    def forward(self, h):

        for i in range(self.num_slp_layers):
            if i == 0:
                layer = self.gin_mlp_layers[i](h)
                #layer = layer.permute(0, 2, 1)
                bnorm = self.gin_batch_normalization[i](layer)
                #bnorm = bnorm.permute(0, 2, 1)
                h = self.mlp_act_fun(bnorm)
            else:
                h = self.mlp_act_fun(self.gin_mlp_layers[i](h))
        return h

#mlp_example = GIN_MLPs(num_slp_layers=2, mlp_input_dim=7, mlp_hid_dim=7, mlp_output_dim=7, mlp_act_fun="ReLu", Bias=True)
#x = torch.rand(20,7)
#y = mlp_example(x)
#print(y.size())