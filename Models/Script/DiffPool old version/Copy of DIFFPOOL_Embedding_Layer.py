import sys
import torch
import torch.nn as nn
import matrix_util as Mat_Util
import torch.nn.functional as F
py_path = '/content/drive/MyDrive/Explainability Methods/Models/Script/Layers/'
sys.path.insert(0,py_path)
import GraphSage_Layer as graphsage_layer
class DiffPool_Embedding_Layer(nn.Module):
    '''
        Z, new features size
    '''
    def __init__(self, input_dim_size, new_feat_dim_size, Bias, normalize_embedding, dropout, aggregation):
        super(DiffPool_Embedding_Layer, self).__init__()
        self.input_dim_size = input_dim_size
        self.new_feat_dim_size = new_feat_dim_size
        self.Bias = Bias
        self.normalize_embedding = normalize_embedding
        self.dropout = dropout
        self.aggregation = aggregation
        self.embedding_layer = graphsage_layer.GNN_GraphSage_Layer(input_dim=self.input_dim_size, output_dim=self.new_feat_dim_size, Bias=self.Bias, normalize_embedding=self.normalize_embedding, dropout=self.dropout, aggregation=self.aggregation)
        self.act_fun = F.relu


    def forward(self, input_tensor, tilda_adjacency_matrix):
        #x, edge_index, batch, y = batched_graph.x, batched_graph.edge_index, batched_graph.batch, batched_graph.y 
        z_l_init = self.embedding_layer(input_tensor, tilda_adjacency_matrix)
        z_l_init = self.act_fun(z_l_init)

        return z_l_init