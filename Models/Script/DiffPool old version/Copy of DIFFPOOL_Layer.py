import sys
import torch
import torch.nn as nn
import matrix_util as Mat_Util
import torch.nn.functional as F
py_path = '/content/drive/MyDrive/Explainability Methods/Models/Script/Layers/'
sys.path.insert(0,py_path)
import GraphSage_Layer as graphsage_layer
import DIFFPOOL_Assignment_Layer as diffpool_assignment_layer
import DIFFPOOL_Embedding_Layer as diffpool_embedding_layer
class DiffPool_Layer(nn.Module):
    def __init__(self, input_dim_size, new_feat_dim_size, new_num_nodes, Bias, normalize_embedding, dropout, aggregation):
        super(DiffPool_Layer, self).__init__()
        self.input_dim_size = input_dim_size
        self.new_feat_dim_size = new_feat_dim_size
        self.new_num_nodes = new_num_nodes
        self.Bias = Bias
        self.normalize_embedding = normalize_embedding
        self.dropout = dropout
        self.aggregation = aggregation

        self.new_embed = diffpool_embedding_layer.DiffPool_Embedding_Layer(input_dim_size=input_dim_size, new_feat_dim_size=new_feat_dim_size, Bias=self.Bias, normalize_embedding=self.normalize_embedding, dropout=self.dropout, aggregation=self.aggregation)
        self.new_assign = diffpool_assignment_layer.DiffPool_Assignment_Layer(input_dim_size=self.input_dim_size, new_num_nodes=self.new_num_nodes, Bias=self.Bias , normalize_embedding=self.normalize_embedding , dropout=self.dropout, aggregation=self.aggregation)
    
    def forward(self, input_tensor, tilda_adjacency_matrix):
        #x, edge_index, batch, y = batched_graph.x, batched_graph.edge_index, batched_graph.batch, batched_graph.y 

        z_l = self.new_embed(input_tensor, tilda_adjacency_matrix)
        s_l = self.new_assign(input_tensor, tilda_adjacency_matrix)

        new_X = torch.mm(s_l.transpose(-1, -2), z_l)
        new_adjacency = (s_l.transpose(-1, -2)).mm(tilda_adjacency_matrix).mm(s_l)

        return new_X, new_adjacency