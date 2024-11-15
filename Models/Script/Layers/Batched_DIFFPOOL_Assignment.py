import sys 
py_path = '/data/cs.aau.dk/ey33jw/Explainability_Methods/Models/Script/Layers/'
sys.path.insert(0,py_path)
from torch_geometric.typing import Adj, OptPairTensor, Size, SparseTensor
import torch
import torch.nn as nn
import matrix_util as Mat_Util
import Batched_GraphSage_Layer as batched_graphsage_layer
import torch.nn.functional as F
class Batched_DiffPool_Assignment_Layer(nn.Module):
    '''
    #    S, new clusters, new number of nodes
    '''
    def __init__(self, input_dim, assignment_num_block_layers, hid_dim, assigned_dim, Bias, normalize_graphsage, dropout,
                 aggregation, concat):
        super(Batched_DiffPool_Assignment_Layer, self).__init__()
        self.input_dim = input_dim
        self.assignment_num_block_layers = assignment_num_block_layers
        self.hid_dim = hid_dim
        self.assigned_dim = assigned_dim
        self.assignment_prediction_layer_input_dim = assigned_dim
        #########################################################.  General Parameters
        self.Bias = Bias
        self.normalize_graphsage = normalize_graphsage
        self.dropout = dropout
        self.aggregation = aggregation
        self.act_fun = F.relu
        self.concat=concat

        self.diffPool_assignment = nn.ModuleList()

        self.diffPool_assignment.append(batched_graphsage_layer.GNN_Batched_GraphSage_Layer(input_dim=self.input_dim, output_dim=self.hid_dim,
                                                                                            Bias=self.Bias, normalize_graphsage=self.normalize_graphsage,
                                                                                            dropout=self.dropout, aggregation=self.aggregation, concat=self.concat))

        for i in range(assignment_num_block_layers):
            self.diffPool_assignment.append(batched_graphsage_layer.GNN_Batched_GraphSage_Layer(input_dim=self.hid_dim, output_dim=self.hid_dim,
                                                                                                Bias=self.Bias, normalize_graphsage=self.normalize_graphsage,
                                                                                                dropout=self.dropout, aggregation=self.aggregation, concat=self.concat))

        self.diffPool_assignment.append(batched_graphsage_layer.GNN_Batched_GraphSage_Layer(input_dim=self.hid_dim, output_dim=self.assigned_dim,
                                                                                            Bias=self.Bias, normalize_graphsage=self.normalize_graphsage,
                                                                                            dropout=self.dropout, aggregation=self.aggregation, concat=self.concat))


    def forward(self, input_tensor, tilda_adjacency_matrix):

        for layer in self.diffPool_assignment:
            input_tensor = self.act_fun(layer(input_tensor, tilda_adjacency_matrix))

        dense_prediction = F.softmax(input_tensor, dim=-1)

        return dense_prediction
