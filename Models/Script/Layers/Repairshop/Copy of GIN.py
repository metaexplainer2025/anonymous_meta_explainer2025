import torch
import torch.nn as nn
from torch.autograd import Variable
import math
import torch_geometric
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch_geometric.utils.convert import to_scipy_sparse_matrix
#from torch_geometric.utils.train_test_split_edges import torch_geometric
from torch_geometric.utils import dropout
from torch_geometric.loader import DataLoader
import networkx as nx
import numpy as np
from torch_geometric.nn import GCNConv
import sys 
from torch_geometric.datasets import TUDataset
from torch_geometric.nn import global_add_pool
from scipy.sparse import csr_matrix
py_path = '/content/drive/MyDrive/Explainability Methods/Models/Script/Layers/'
sys.path.insert(0,py_path)
import GIN_MLP_Layers as gin_mlp_layers


class GlobalSUMPool(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x, batch):
        return torch_geometric.nn.global_add_pool(x, batch)
################################################################################
class IdenticalPool(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x, batch):
        return x

################################################################################
class GIN_Model(nn.Module):
    """
    An aggregation-based (message passing) graph learning neural network.
    Please Note that in each MLP, mlp_input_dim & mlp_output_dim should have same value.
    Args(example):
        num_mlp_layers=4, 
        Bias=classifier_bias, 
        num_slp_layers=2, 
        mlp_input_dim=13, 
        mlp_hid_dim=20,
        mlp_output_dim=13, 
        mlp_act_fun="ReLu", 
        num_classes=3, 
        dropout_rate=classifier_dropout,
        Weight_Initializer=1
    """
    def __init__(self, num_mlp_layers, Bias, num_slp_layers, mlp_input_dim, mlp_hid_dim, mlp_output_dim, mlp_act_fun, num_classes, dropout_rate, Weight_Initializer):
        super(GIN_Model, self).__init__()

        self.mlp_input_dim = mlp_input_dim
        self.mlp_hid_dim = mlp_hid_dim
        self.mlp_output_dim = mlp_output_dim
        self.num_slp_layers = num_slp_layers
        self.mlp_act_fun = mlp_act_fun
        self.lin_act_fun = mlp_act_fun
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        self.Weight_Initializer = Weight_Initializer

        self.num_mlp_layers = num_mlp_layers
        self.Bias = Bias

        self.eps = nn.Parameter(torch.zeros(self.num_mlp_layers),requires_grad=True)
        self.gin_mlp_layers = nn.ModuleList()
        self.global_summing = GlobalSUMPool()

        self.lin1 = nn.Linear(in_features=self.mlp_output_dim * (self.num_mlp_layers + 1), out_features=self.mlp_input_dim * (self.num_mlp_layers + 1))
        self.lin2 = nn.Linear(in_features=self.mlp_input_dim * (self.num_mlp_layers + 1), out_features=self.num_classes)
        self.dorpout = nn.Dropout(p=dropout_rate)
        self.act_fun_softmax = F.softmax
        self.the_first_layer = nn.Linear(in_features=self.mlp_input_dim, out_features=self.mlp_output_dim, bias=self.Bias)

        for i in range(self.num_mlp_layers):
            self.gin_mlp_layers.append(gin_mlp_layers.GIN_MLPs(num_slp_layers=self.num_slp_layers, mlp_input_dim=self.mlp_input_dim, 
                                                               mlp_hid_dim=self.mlp_hid_dim, mlp_output_dim=self.mlp_output_dim, 
                                                               mlp_act_fun=self.mlp_act_fun, Bias=self.Bias))

        if self.lin_act_fun == 'ReLu':
            self.lin_act_fun = F.relu
            #print('ReLu is Selected.')
        elif self.lin_act_fun == 'eLu':
            self.lin_act_fun = nn.functional.elu
            #print('eLu is Selected.')
        elif self.lin_act_fun == 'tanh':
            self.lin_act_fun = torch.tanh
            #print('tanh is Selected.')

        mean = 0
        std = 0.1
        self.initialize_weights(self.Weight_Initializer, Bias, mean, std)

    def initialize_weights(model, Weight_Initializer, Bias, mean, std):
        # 1. Xavier Normal_.  2. Kaiming Normal_.  3. Uniform (0,0.1std)
        if Weight_Initializer == 1:                                             #.      1. Xavier Normal_.
            for i, modules in enumerate(model.children()):
                if isinstance(modules, torch.nn.ModuleList):
                    for module in modules:
                        if isinstance(module, gin_mlp_layers.GIN_MLPs):
                            for final_module in module.children():
                                if isinstance(final_module, torch.nn.ModuleList):
                                    for layers in final_module:
                                        if isinstance(layers, torch.nn.Linear):
                                            torch.nn.init.xavier_normal_(layers.weight)
                                            #layers.weight.data.zero_()
                                            #print(layers.weight)
                                            if Bias:
                                                layers.bias.data.zero_()
                                                #print("ok")
                                        elif isinstance(layers, torch.nn.BatchNorm1d):
                                            #print("ok")
                                            pass
                elif isinstance(modules, torch.nn.Linear):
                    #print("predict layer")
                    #modules.weight.data.zero_()
                    #print(modules.weight)
                    torch.nn.init.xavier_normal_(modules.weight)
                elif isinstance(modules, GlobalSUMPool):
                    #print("GlobalSUMPool")
                    #print(modules)
                    pass
                elif isinstance(modules, nn.Dropout):
                    #print("Dropout")
                    #print(modules)
                    pass
                else:
                    pass

        if Weight_Initializer == 2:                                             #.      2. Kaiming Normal_.
            for i, modules in enumerate(model.children()):
                if isinstance(modules, torch.nn.ModuleList):
                    for module in modules:
                        if isinstance(module, gin_mlp_layers.GIN_MLPs):
                            for final_module in module.children():
                                if isinstance(final_module, torch.nn.ModuleList):
                                    for layers in final_module:
                                        if isinstance(layers, torch.nn.Linear):
                                            torch.nn.init.kaiming_normal_(layers.weight)
                                            #layers.weight.data.zero_()
                                            #print(layers.weight)
                                            if Bias:
                                                layers.bias.data.zero_()
                                                #print("ok")
                                        elif isinstance(layers, torch.nn.BatchNorm1d):
                                            #print("ok")
                                            pass
                elif isinstance(modules, torch.nn.Linear):
                    #print("predict layer")
                    #modules.weight.data.zero_()
                    #print(modules.weight)
                    torch.nn.init.kaiming_normal_(modules.weight)
                elif isinstance(modules, GlobalSUMPool):
                    #print("GlobalSUMPool")
                    #print(modules)
                    pass
                elif isinstance(modules, nn.Dropout):
                    #print("Dropout")
                    #print(modules)
                    pass
                else:
                    pass

        if Weight_Initializer == 3:                                             #.      3. Uniform (0,0.1std)
            for i, modules in enumerate(model.children()):
                if isinstance(modules, torch.nn.ModuleList):
                    for module in modules:
                        if isinstance(module, gin_mlp_layers.GIN_MLPs):
                            for final_module in module.children():
                                if isinstance(final_module, torch.nn.ModuleList):
                                    for layers in final_module:
                                        if isinstance(layers, torch.nn.Linear):
                                            torch.nn.init.normal_(layers.weight, mean=mean, std=std)
                                            #layers.weight.data.zero_()
                                            #print(layers.weight)
                                            if Bias:
                                                layers.bias.data.zero_()
                                                #print("ok")
                                        elif isinstance(layers, torch.nn.BatchNorm1d):
                                            #print("ok")
                                            pass
                elif isinstance(modules, torch.nn.Linear):
                    #print("predict layer")
                    #modules.weight.data.zero_()
                    #print(modules.weight)
                    torch.nn.init.normal_(modules.weight, mean=mean, std=std)
                elif isinstance(modules, GlobalSUMPool):
                    #print("GlobalSUMPool")
                    #print(modules)
                    pass
                elif isinstance(modules, nn.Dropout):
                    #print("Dropout")
                    #print(modules)
                    pass
                else:
                    pass


    def gin_neighborhood_aggregation(self, new_adjacecny, new_features):

        pooled = torch.bmm(new_adjacecny, new_features)


        return pooled


    def gin_layer_process_eps(self, hid_rep, layer, new_adjacecny):

        pooled = self.gin_neighborhood_aggregation(new_adjacecny, hid_rep)


        pooled = pooled + (1 + self.eps[layer])*hid_rep
        pooled_rep = self.gin_mlp_layers[layer](pooled)


        return pooled_rep

    def merging_process(self, one_mlp, graph_sizes):
        new=[]
        start=0
        for j in range(len(graph_sizes)):
            end = start + graph_sizes[j]
            new.append(one_mlp[start:end])
            start = end
        return new

    def reshape_mlps_outputs(self, mlps_output_embeds, graph_sizes):
        merged_mlps_output_embeds = []
        for i in range(len(graph_sizes)):
            merged_mlps_output_embeds.append([])

        for i in range(len(mlps_output_embeds)):
            for j in range(len(mlps_output_embeds[i])):
                merged_mlps_output_embeds[j].extend(mlps_output_embeds[i][j])
        return merged_mlps_output_embeds


    def computational_matrix(self, batched_graphs, edge_mask):

        if edge_mask == None:
            joint_tilda_adjacency_matrix = torch.tensor(to_scipy_sparse_matrix(batched_graphs.edge_index).todense())
        else:
            joint_tilda_adjacency_matrix = torch.tensor(csr_matrix((np.array(edge_mask), (np.array(batched_graphs.edge_index[0]), np.array(batched_graphs.edge_index[1])))).todense())
        joint_tilda_adjacency_matrix = joint_tilda_adjacency_matrix.type(torch.float32)


        if batched_graphs.batch is not None:
            graph_sizes = [len(batched_graphs[i].x) for i in range(len(batched_graphs))]
            batch_size = batched_graphs.num_graphs
        else:
            graph_sizes = [len(batched_graphs.x)]
            batch_size = 1
        max_number_of_nodes_in_batch_of_graphs = max(graph_sizes)


        adjacency_list = []
        feature_list = []
        start = 0
        for i in range(batch_size):
            end = start + graph_sizes[i]
            un_padded_adj = joint_tilda_adjacency_matrix[start:end, start:end]
            adj_off_set = max_number_of_nodes_in_batch_of_graphs - un_padded_adj.size()[0]
            if un_padded_adj.size()[0] <= max_number_of_nodes_in_batch_of_graphs:
                un_padded_adj = F.pad(un_padded_adj, (0, adj_off_set, 0, adj_off_set), mode='constant', value=0)
                un_padded_adj = un_padded_adj.type(torch.float32)
                num_nodes = max_number_of_nodes_in_batch_of_graphs
            un_padded_adj = un_padded_adj.type(torch.float32)
            adjacency_list.append(un_padded_adj)
            un_padded_feat = batched_graphs.x[start:end, :]
            node_feat_off_set = max_number_of_nodes_in_batch_of_graphs - graph_sizes[i]
            un_padded_feat = F.pad(un_padded_feat, (0, 0, 0, node_feat_off_set), mode='constant', value=0)
            un_padded_feat = un_padded_feat.type(torch.float32)
            un_padded_feat.require_grad = True
            feature_list.append(un_padded_feat)
            start = end

        adjacency_list = list(map(lambda x: torch.unsqueeze(x, 0), adjacency_list))
        feature_list = list(map(lambda x: torch.unsqueeze(x, 0), feature_list))

        new_adjacecny = torch.cat(adjacency_list, dim=0)
        new_features = torch.cat(feature_list, dim=0)

        return new_adjacecny, new_features


    def forward(self, batched_graphs, edge_mask):

        if batched_graphs.batch is not None:
            graph_sizes = [len(batched_graphs[i].x) for i in range(len(batched_graphs))]
            num_graphs = batched_graphs.num_graphs
        else:
            graph_sizes = [len(batched_graphs.x)]
            num_graphs = 1
        new_adjacecny, new_features = self.computational_matrix(batched_graphs, edge_mask)
        new_features = self.the_first_layer(new_features)
        mlps_output_embeds = []
        mlps_output_embeds_pooled = []
        mlps_output_embeds.append(new_features)
        sum_pooling_output = torch.sum(new_features, dim=1)
        mlps_output_embeds_pooled.append(sum_pooling_output)

        for layer in range(self.num_mlp_layers):
            new_features = self.gin_layer_process_eps(new_features, layer, new_adjacecny)
            each_MLP_batch_graphs_summed = torch.sum(new_features, dim=1)
            mlps_output_embeds.append(new_features)
            mlps_output_embeds_pooled.append(each_MLP_batch_graphs_summed)

        mlps_output_embeds_pooled_stacked = torch.stack(mlps_output_embeds_pooled)

        mlps_output_embeds_pooled_stacked = mlps_output_embeds_pooled_stacked.view(num_graphs, (self.num_mlp_layers+1)*sum_pooling_output.size()[1])

        mlps_output_embeds_stacked = torch.stack(mlps_output_embeds)
        lin1_output = self.lin1(mlps_output_embeds_pooled_stacked)
        lin1_output = self.lin_act_fun(lin1_output)

        lin1_output_dropouted = self.dorpout(lin1_output)

        lin2_output = self.lin2(lin1_output_dropouted)
        lin2_output_softmaxed = self.act_fun_softmax(lin2_output, dim=1)

        return mlps_output_embeds, mlps_output_embeds_stacked, mlps_output_embeds_pooled, mlps_output_embeds_pooled_stacked, lin1_output, lin1_output_dropouted, lin2_output, lin2_output_softmaxed


        

#dataset = TUDataset(root='data/TUDataset', name='MUTAG')
#batch_size = 20 
#node_feat_size = len(dataset[0].x[0])
#batched_dataset = DataLoader(dataset, batch_size=batch_size, shuffle=False)
#gin_model_example = GIN_Model(num_mlp_layers=4, Bias=True, num_slp_layers=2, mlp_input_dim=node_feat_size, mlp_hid_dim=7, mlp_output_dim=7, #mlp_act_fun="ReLu", num_classes=2, dropout_rate=0.5, Weight_Initializer=3)
#print(gin_model_example)

#for batched_graphs in batched_dataset:
#    #x, edge_index, batch, y = batched_graphs.x, batched_graphs.edge_index, batched_graphs.batch, batched_graphs.y
#    mlps_output_embeds, mlps_output_embeds_stacked, mlp_outputs_globalSUMpooled, merged_mlps_output_embeds_reshaped, lin1_output, #lin1_output_dropouted, lin2_output, lin2_output_softmaxed = gin_model_example(batched_graphs)
#    print("lin2_output_softmaxed: ", lin2_output_softmaxed.size())
#    print(lin2_output_softmaxed)
#    break