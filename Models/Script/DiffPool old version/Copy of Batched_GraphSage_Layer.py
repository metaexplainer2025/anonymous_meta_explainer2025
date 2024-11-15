import sys 
py_path = '/content/drive/MyDrive/Explainability Methods/Models/Script/Layers/'
sys.path.insert(0,py_path)
from torch_geometric.typing import Adj, OptPairTensor, Size, SparseTensor
import torch
import torch.nn as nn
import matrix_util as Mat_Util
import torch.nn.functional as F
class GNN_Batched_GraphSage_Layer(nn.Module):
    '''
        #    A single GraphSage Layer: Graph Sampling and Aggregate
    '''
    def __init__(self, input_dim, output_dim, Bias, normalize_embedding, dropout, aggregation):
        super(GNN_Batched_GraphSage_Layer, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.Bias = Bias
        self.dropout = dropout
        self.normalize_embedding = normalize_embedding
        self.aggregation = aggregation
        #self.add_self = add_self
        
        if self.aggregation == 'mean':
            self.learnable_weights = nn.Linear(self.input_dim*2, self.output_dim, bias=self.Bias)
        else:
            self.learnable_weights = nn.Linear(self.input_dim, self.output_dim, bias=self.Bias)

        self.normalize = F.normalize
    
    
    def forward(self, new_features, tilda_adjacency_matrix):
        #tilda_adjacency_matrix, new_features = self.computational_matricess(batched_graphs)
        new_features = new_features.to(torch.float32)

    
        if self.aggregation == 'mean':
            tilda_adjacency_matrix = tilda_adjacency_matrix / tilda_adjacency_matrix.sum(-2, keepdim=True)
        num_node_per_graph = tilda_adjacency_matrix.size(1)

        tilda_adjacency_matrix_neghborhood = torch.bmm(tilda_adjacency_matrix, new_features) # Y = A~ * X

        neighborhood_aggregated = torch.cat((tilda_adjacency_matrix_neghborhood, new_features), 2)

        node_linear = self.learnable_weights(neighborhood_aggregated) # Y * W
        
        if self.normalize_embedding:
            node_linear = self.normalize(node_linear, p=2, dim=1)
        


        return node_linear