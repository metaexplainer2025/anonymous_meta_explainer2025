from torch_geometric.utils import dropout
from torch_geometric.loader import DataLoader
import torch
import torch.nn as nn
from torch.autograd import Variable
import math
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch_geometric.utils.convert import to_scipy_sparse_matrix
from torch_geometric.utils.train_test_split_edges import torch_geometric
import networkx as nx
import numpy as np
from torch_geometric.nn import GCNConv
import sys 
from torch_geometric.datasets import TUDataset
py_path = '/content/drive/MyDrive/Explainability Methods/Models/Script/Layers/'
sys.path.insert(0,py_path)
import Batched_GraphSage_Layer as batched_graphsage_layer
import Batched_DIFFPOOL_Assignment as batched_diffpool_assignment
import Batched_DIFFPOOL_Embedding as batched_diffpool_embedding
import Batched_DIFFPOOL_Layer as batched_diffpool_layer



class GlobalMeanPool(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x, batch):
        return gnn.global_mean_pool(x, batch)
################################################################################
class IdenticalPool(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x, batch):
        return x

################################################################################
class DIFFPOOL_Model(nn.Module):
    '''
        DIFFPOOL Mode
    '''
    def __init__(self, diffpool_layers_dim, diffpool_layers_new_num_nodes, Weight_Initializer, Bias, num_classes, dropout_rate, normalize_embedding, aggregation, act_fun):
        
        super(DIFFPOOL_Model, self).__init__()
        self.diffpool_layers_dim = diffpool_layers_dim
        self.diffpool_layers_new_num_nodes = diffpool_layers_new_num_nodes
        self.dropout_rate = dropout_rate
        self.Bias = Bias
        self.normalize_embedding = normalize_embedding
        self.aggregation = aggregation
        self.num_classes = num_classes

                                
        self.diffpool_layer_1 = batched_diffpool_layer.Batched_DiffPool_Layer(input_dim_size=self.diffpool_layers_dim[0][0], new_feat_dim_size=self.diffpool_layers_dim[0][1], new_num_nodes=self.diffpool_layers_new_num_nodes[0], Bias=self.Bias, normalize_embedding=self.normalize_embedding, dropout=self.dropout_rate, aggregation=self.aggregation)

        self.graph_sage_1 = batched_graphsage_layer.GNN_Batched_GraphSage_Layer(input_dim=self.diffpool_layers_dim[0][1], output_dim=self.diffpool_layers_dim[1][0], Bias=self.Bias, normalize_embedding=self.normalize_embedding, dropout=self.dropout_rate, aggregation=self.aggregation)

        self.diffpool_layer_2 = batched_diffpool_layer.Batched_DiffPool_Layer(input_dim_size=self.diffpool_layers_dim[1][0], new_feat_dim_size=self.diffpool_layers_dim[1][1], new_num_nodes=self.diffpool_layers_new_num_nodes[1], Bias=self.Bias, normalize_embedding=self.normalize_embedding, dropout=self.dropout_rate, aggregation=self.aggregation)

        self.graph_sage_2 = batched_graphsage_layer.GNN_Batched_GraphSage_Layer(input_dim=self.diffpool_layers_dim[1][1], output_dim=self.diffpool_layers_dim[1][1], Bias=self.Bias, normalize_embedding=self.normalize_embedding, dropout=self.dropout_rate, aggregation=self.aggregation)

        self.lin1 = torch.nn.Linear(in_features=self.diffpool_layers_dim[1][1], out_features=self.diffpool_layers_dim[1][1])
        self.lin2 = torch.nn.Linear(in_features=self.diffpool_layers_dim[1][1], out_features=self.diffpool_layers_dim[1][1])
        self.lin3 = torch.nn.Linear(in_features=self.diffpool_layers_dim[1][1], out_features=self.num_classes)
        
        if act_fun == 'ReLu':
            self.act_fun = F.relu
            print('ReLu is Selected.')
        elif act_fun == 'eLu':
            self.act_fun = nn.functional.elu
            print('eLu is Selected.')
        elif act_fun == 'tanh':
            self.act_fun = torch.tanh
            print('tanh is Selected.')
        self.act_fun_softmax = F.softmax       
        

        mean = 0
        std = 0.1
        self.initialize_weights(Weight_Initializer, Bias, mean, std)
    

    def initialize_weights(model, Weight_Initializer, Bias, mean, std):
        # 1. Xavier Normal_.  2. Kaiming Normal_.  3. Uniform (0,0.1std)
        if Weight_Initializer == 1:                                             #.      1. Xavier Normal_.
            for i,layers in enumerate(model.children()):
                if isinstance(layers, torch.nn.ModuleList):
                    for j, layer in enumerate(layers.modules()):
                        if isinstance(layer, batched_graphsage_layer.GNN_Batched_GraphSage_Layer):
                            torch.nn.init.xavier_normal_(layer.learnable_weights.weight)
                            if Bias:
                                layer.learnable_weights.bias.data.zero_()
                        else:
                            pass
                elif isinstance(layers, batched_graphsage_layer.GNN_Batched_GraphSage_Layer):
                    torch.nn.init.xavier_normal_(layers.learnable_weights.weight)
                    if Bias:
                        layers.learnable_weights.bias.data.zero_()
                elif isinstance(layers, batched_diffpool_layer.Batched_DiffPool_Layer):
                    torch.nn.init.xavier_normal_(layers.new_assign.assinment_layer.learnable_weights.weight)
                    torch.nn.init.xavier_normal_(layers.new_embed.embedding_layer.learnable_weights.weight)
                    if Bias:
                        torch.nn.init.zeros_(layers.new_assign.assinment_layer.learnable_weights.bias)
                        torch.nn.init.zeros_(layers.new_embed.embedding_layer.learnable_weights.bias)
                elif isinstance(layers, torch.nn.Linear):
                    torch.nn.init.xavier_normal_(layers.weight)
                    if Bias:
                        torch.nn.init.zeros_(layers.bias)
                    

        if Weight_Initializer == 2:                                             #.      2. Kaiming Normal_.
            for i,layers in enumerate(model.children()):
                if isinstance(layers, torch.nn.ModuleList):
                    for j, layer in enumerate(layers.modules()):
                        if isinstance(layer, batched_graphsage_layer.GNN_Batched_GraphSage_Layer):
                            torch.nn.init.kaiming_normal_(layer.learnable_weights.weight)
                            if Bias:
                                layer.learnable_weights.bias.data.zero_()
                        else:
                            pass
                elif isinstance(layers, batched_graphsage_layer.GNN_Batched_GraphSage_Layer):
                    torch.nn.init.kaiming_normal_(layers.learnable_weights.weight)
                    if Bias:
                        layers.learnable_weights.bias.data.zero_()
                elif isinstance(layers, batched_diffpool_layer.Batched_DiffPool_Layer):
                    torch.nn.init.kaiming_normal_(layers.new_assign.assinment_layer.learnable_weights.weight)
                    torch.nn.init.kaiming_normal_(layers.new_embed.embedding_layer.learnable_weights.weight)
                    if Bias:
                        torch.nn.init.zeros_(layers.new_assign.assinment_layer.learnable_weights.bias)
                        torch.nn.init.zeros_(layers.new_embed.embedding_layer.learnable_weights.bias)
                elif isinstance(layers, torch.nn.Linear):
                    torch.nn.init.kaiming_normal_(layers.weight)
                    if Bias:
                        torch.nn.init.zeros_(layers.bias)
                    
                            
        if Weight_Initializer == 3:                                             #.      3. Uniform (0,0.1std)
            for i,layers in enumerate(model.children()):
                if isinstance(layers, torch.nn.ModuleList):
                    for j, layer in enumerate(layers.modules()):
                        if isinstance(layer, batched_graphsage_layer.GNN_Batched_GraphSage_Layer):
                            torch.nn.init.normal_(layer.learnable_weights.weight.data, mean, std)
                            if Bias:
                                layer.learnable_weights.bias.data.zero_()
                        else:
                            pass
                elif isinstance(layers, batched_graphsage_layer.GNN_Batched_GraphSage_Layer):
                    torch.nn.init.normal_(layers.learnable_weights.weight, mean=mean, std=std)
                    if Bias:
                        layers.learnable_weights.bias.data.zero_()
                elif isinstance(layers, batched_diffpool_layer.Batched_DiffPool_Layer):
                    torch.nn.init.normal_(layers.new_assign.assinment_layer.learnable_weights.weight, mean=mean, std=std)
                    torch.nn.init.normal_(layers.new_embed.embedding_layer.learnable_weights.weight, mean=mean, std=std)
                    if Bias:
                        torch.nn.init.zeros_(layers.new_assign.assinment_layer.learnable_weights.bias)
                        torch.nn.init.zeros_(layers.new_embed.embedding_layer.learnable_weights.bias)
                elif isinstance(layers, torch.nn.Linear):
                    torch.nn.init.normal_(layers.weight, mean=mean, std=std)
                    if Bias:
                        torch.nn.init.zeros_(layers.bias)
                    

    def computational_matricess(self, batched_graphs, edge_mask):
        if edge_mask == None:
            joint_tilda_adjacency_matrix = torch.tensor(to_scipy_sparse_matrix(batched_graphs.edge_index).todense()) + torch.eye(len(torch.tensor(to_scipy_sparse_matrix(batched_graphs.edge_index).todense())))
        else:
            joint_tilda_adjacency_matrix = torch.tensor(csr_matrix((np.array(edge_mask), (np.array(batched_graphs.edge_index[0]), np.array(batched_graphs.edge_index[1])))).todense()) + torch.eye(len(torch.tensor(to_scipy_sparse_matrix(batched_graphs.edge_index).todense())))
        joint_tilda_adjacency_matrix = joint_tilda_adjacency_matrix.type(torch.float32)
        if batched_graphs.batch == None:
            batch_size = 1
        else:
            batch_size = batched_graphs.num_graphs

        #print("whole_graphs_adjacency.size()[0]: ", whole_graphs_adjacency.size()[0])
        new_number_of_nodes = int(joint_tilda_adjacency_matrix.size()[0] / batch_size)
        #print(batch_size)
        adjacency_list = []
        feature_list = []
        for i in range(batch_size):
            start = i * new_number_of_nodes
            end = (i + 1) * new_number_of_nodes
            adjacency_list.append(joint_tilda_adjacency_matrix[start:end, start:end])
            feature_list.append(batched_graphs.x[start:end, :])
        adjacency_list = list(map(lambda x: torch.unsqueeze(x, 0), adjacency_list))
        feature_list = list(map(lambda x: torch.unsqueeze(x, 0), feature_list))
        new_adjacecny = torch.cat(adjacency_list, dim=0)
        new_features = torch.cat(feature_list, dim=0)
        new_adjacecny = new_adjacecny.view(batch_size, new_number_of_nodes,new_number_of_nodes)

        return new_adjacecny, new_features  

    
    def forward(self, batched_graphs, edge_mask):
        new_adjacecny, new_features = self.computational_matricess(batched_graphs, edge_mask)
        
        #if batch is not None:
        #    max_dim = 1
        #else:
        #    max_dim = 0

        new_features = new_features.to(torch.float32)

        new_X, new_adjacency_1 = self.diffpool_layer_1(new_features, new_adjacecny)
        graph_sage1_output = self.graph_sage_1(new_X, new_adjacency_1)

        new_X_2, new_adjacency_2 = self.diffpool_layer_2(graph_sage1_output, new_adjacency_1)
        graph_sage2_output = self.graph_sage_2(new_X_2, new_adjacency_2)
        #print("graph_sage2_output: ",graph_sage2_output.size())

        #graph_sage2_output = graph_sage2_output.view((len(graph), self.diffpool_layers_new_num_nodes[1], self.diffpool_layers_dim[1][1]))
        graph_sage2_output, q = torch.max(graph_sage2_output, dim=1, keepdim=True)
        #print(graph_sage2_output)
        #graph_sage2_output= graph_sage2_output.sum(dim=1, keepdim=True)

        linear1_output = self.lin1(graph_sage2_output)
        linear1_output = self.act_fun(linear1_output)

        linear2_output = self.lin2(linear1_output)
        linear2_output = self.act_fun(linear2_output)

        linear3_output = self.lin3(linear2_output)
        linear3_output = self.act_fun(linear3_output)
        linear3_output = torch.squeeze(self.act_fun_softmax(linear3_output, dim=2))
        #print(linear3_output)

        
        return new_X, new_adjacency_1, graph_sage1_output, new_X_2, new_adjacency_2, graph_sage2_output, linear1_output, linear2_output, linear3_output








#dataset = TUDataset(root='data/TUDataset', name='MUTAG')
#batch_size = 32
#new_num_nodes = 10
#print("new_num_nodes: ", new_num_nodes)

#node_feat_size = len(dataset[0].x[0])
#hid_dim = 7

#diffpool_model_example = DIFFPOOL_Model(diffpool_layers_dim=[[node_feat_size, hid_dim], [hid_dim, node_feat_size]], #diffpool_layers_new_num_nodes=[new_num_nodes, new_num_nodes], 
#                                        Weight_Initializer=1, Bias=True, num_classes=2, dropout_rate=0, normalize_embedding=True, #aggregation####='mean', 
#                                        act_fun='ReLu')
#
#batched_dataset = DataLoader(dataset, batch_size=batch_size, shuffle=False)

#for batched_graph in batched_dataset:
#    x, edge_index, batch, y = batched_graph.x, batched_graph.edge_index, batched_graph.batch, batched_graph.y
#    new_X, new_adjacency_1, graph_sage1_output, new_X_2, new_adjacency_2, graph_sage2_output, linear1_output, linear2_output, linear3_output = #diffpool_model_example(batched_graph)
#    print("Final Output: ", linear3_output.size())
