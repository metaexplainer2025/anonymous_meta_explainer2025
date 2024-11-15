import sys 
py_path = '/data/cs.aau.dk/ey33jw/Explainability_Methods/Models/Script/Layers/'
sys.path.insert(0,py_path)
from torch_geometric.typing import Adj, OptPairTensor, Size, SparseTensor
import torch
import torch.nn as nn
import matrix_util as Mat_Util
import Batched_GraphSage_Layer as batched_graphsage_layer
import torch.nn.functional as F
class Batched_DiffPool_Embedding_Layer(nn.Module):
    '''
   #     Z, new features size
    '''
    def __init__(self, input_dim, embedding_num_block_layers, hid_dim, embedded_dim, Bias, normalize_graphsage, dropout,
                 aggregation, concat):
        super(Batched_DiffPool_Embedding_Layer, self).__init__()
        self.input_dim = input_dim
        self.embedding_num_block_layers = embedding_num_block_layers
        self.hid_dim = hid_dim
        self.embedded_dim = embedded_dim
        self.Bias = Bias
        self.normalize_graphsage = normalize_graphsage
        self.dropout = dropout
        self.aggregation = aggregation
        self.act_fun = F.relu
        self.concat = concat


        self.DiffPool_Embedding = nn.ModuleList()

        if self.concat:
            self.hid_dim = 2*self.hid_dim

        self.DiffPool_Embedding.append(batched_graphsage_layer.GNN_Batched_GraphSage_Layer(input_dim=self.input_dim, output_dim=self.hid_dim,
                                                                                           Bias=self.Bias, normalize_graphsage=self.normalize_graphsage,
                                                                                           dropout=self.dropout, aggregation=self.aggregation, concat=self.concat))


        for i in range(embedding_num_block_layers):
            self.DiffPool_Embedding.append(batched_graphsage_layer.GNN_Batched_GraphSage_Layer(input_dim=self.hid_dim, output_dim=self.hid_dim,
                                                                                               Bias=self.Bias, normalize_graphsage=self.normalize_graphsage,
                                                                                               dropout=self.dropout, aggregation=self.aggregation, concat=self.concat))

        self.DiffPool_Embedding.append(batched_graphsage_layer.GNN_Batched_GraphSage_Layer(input_dim=self.hid_dim, output_dim=self.embedded_dim,
                                                                                           Bias=self.Bias, normalize_graphsage=self.normalize_graphsage,
                                                                                           dropout=self.dropout, aggregation=self.aggregation, concat=self.concat))



    def forward(self, input_tensor, tilda_adjacency_matrix):

        for layer in self.DiffPool_Embedding:
            input_tensor = self.act_fun(layer(input_tensor, tilda_adjacency_matrix))

        return input_tensor
