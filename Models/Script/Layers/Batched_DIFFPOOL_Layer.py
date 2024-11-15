import sys 
py_path = '/data/cs.aau.dk/ey33jw/Explainability_Methods/Models/Script/Layers/'
sys.path.insert(0,py_path)
from torch_geometric.typing import Adj, OptPairTensor, Size, SparseTensor
import torch
import torch.nn as nn
import matrix_util as Mat_Util
import Batched_DIFFPOOL_Embedding as batched_diffpool_embedding
import Batched_DIFFPOOL_Assignment as batched_diffpool_assignment
class Batched_DiffPool_Layer(nn.Module):
    def __init__(self, embedding_input_dim, embedding_num_block_layers, embedding_hid_dim, embedding_output_dim, assignment_input_dim,
                 assignment_num_block_layers, assignment_hid_dim, assignment_output_dim, concat, Weight_Initializer, Bias,
                 dropout_rate, normalize_graphsage, aggregation, act_fun):
        super(Batched_DiffPool_Layer, self).__init__()
        ################################################.  General Parameters
        self.Bias = Bias
        self.dropout_rate = dropout_rate
        self.aggregation = aggregation
        self.Weight_Initializer = Weight_Initializer
        self.concat = concat
        self.normalize_graphsage = normalize_graphsage

        ################################################.  Embedding and Assignment

        self.embedding_input_dim = embedding_input_dim
        self.embedding_num_block_layers = embedding_num_block_layers
        self.embedding_hid_dim = embedding_hid_dim
        self.embedding_output_dim = embedding_output_dim

        self.assignment_input_dim = assignment_input_dim
        self.assignment_num_block_layers = assignment_num_block_layers
        self.assignment_hid_dim = assignment_hid_dim
        self.assignment_output_dim = assignment_output_dim      # new number of nodes

        ################################################.  Embedding
        self.diffpool_embedding = batched_diffpool_embedding.Batched_DiffPool_Embedding_Layer(input_dim=self.embedding_input_dim,
                                                                                              embedding_num_block_layers=self.embedding_num_block_layers,
                                                                                              hid_dim=self.embedding_hid_dim,
                                                                                              embedded_dim=self.embedding_output_dim,
                                                                                              concat=self.concat,Bias=self.Bias,
                                                                                              normalize_graphsage=self.normalize_graphsage,
                                                                                              dropout=self.dropout_rate,
                                                                                              aggregation=self.aggregation)

        ################################################.  Assignment
        self.diffpool_assignment = batched_diffpool_assignment.Batched_DiffPool_Assignment_Layer(input_dim=self.assignment_input_dim,
                                                                                                 assignment_num_block_layers=self.assignment_num_block_layers,
                                                                                                 hid_dim=self.assignment_hid_dim,
                                                                                                 assigned_dim=self.assignment_output_dim,
                                                                                                 concat=self.concat, Bias=self.Bias,
                                                                                                 normalize_graphsage=self.normalize_graphsage,
                                                                                                 dropout=self.dropout_rate,
                                                                                                 aggregation=self.aggregation)
    def forward(self, features, adjacecny):
        features = features.to(torch.float32)

        embedding_output = self.diffpool_embedding(features, adjacecny)

        assignment_output = self.diffpool_assignment(features, adjacecny)

        #features = torch.matmul(torch.transpose(assignment_output, 1, 2), embedding_output)
        #adjacecny = torch.transpose(assignment_output, 1, 2) @ adjacecny @ assignment_output



        return embedding_output, assignment_output