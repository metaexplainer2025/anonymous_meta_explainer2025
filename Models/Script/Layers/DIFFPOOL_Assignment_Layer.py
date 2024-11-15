import sys
import torch
import torch.nn as nn
import matrix_util as Mat_Util
import torch.nn.functional as F
py_path = '/data/cs.aau.dk/ey33jw/Explainability_Methods/Models/Script/Layers/'
sys.path.insert(0,py_path)
import GraphSage_Layer as graphsage_layer
class DiffPool_Assignment_Layer(nn.Module):
    '''
        S, new clusters, new number of nodes
    '''
    def __init__(self, input_dim_size, new_num_nodes, Bias, normalize_embedding, dropout, aggregation):
        super(DiffPool_Assignment_Layer, self).__init__()
        self.input_dim_size = input_dim_size
        self.new_num_nodes = new_num_nodes
        self.Bias = Bias
        self.normalize_embedding = normalize_embedding
        self.dropout = dropout
        self.aggregation = aggregation
        self.assinment_layer = graphsage_layer.GNN_GraphSage_Layer(input_dim=self.input_dim_size, output_dim=self.new_num_nodes, Bias=self.Bias, normalize_embedding=self.normalize_embedding, dropout=self.dropout, aggregation=self.aggregation)
        self.act_fun = F.relu
    
    def forward(self, input_tensor, tilda_adjacency_matrix):
        #x, edge_index, batch, y = batched_graph.x, batched_graph.edge_index, batched_graph.batch, batched_graph.y 
        s_l_init = self.assinment_layer(input_tensor, tilda_adjacency_matrix)
        s_l_init = self.act_fun(s_l_init)

        s_l = F.softmax(s_l_init, dim=-1)
        
        return s_l