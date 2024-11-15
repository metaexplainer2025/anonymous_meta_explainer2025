from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F


class DGCNN_MLP(nn.Module):
    def __init__(self, num_class, last_gnn_layer_dim, mlp_act_fun, dropout_rate, hid_channels, conv1d_kernels, dgcnn_k,
                 ffn_layer_size, Bias, strides):

        super(DGCNN_MLP, self).__init__()
        self.dgcnn_k = dgcnn_k
        self.last_gnn_layer_dim = last_gnn_layer_dim
        self.hid_channels = hid_channels
        self.conv1d_kernels = conv1d_kernels
        self.padding = 0
        self.ffn_layer_size = ffn_layer_size
        self.num_class = num_class
        self.Bias = Bias
        self.strides = strides

        self.conv1d_1 = nn.Conv1d(in_channels=1, out_channels=self.hid_channels[0], kernel_size=self.conv1d_kernels[0],
                                  stride=self.strides[0], padding=self.padding, bias=self.Bias)

        self.maxpool1d = nn.MaxPool1d(kernel_size=2, stride=2)

        self.conv1d_2 = nn.Conv1d(in_channels=self.hid_channels[0], out_channels=self.hid_channels[1],
                                  kernel_size=self.conv1d_kernels[1], stride=self.strides[1], bias=self.Bias)

        dim_conv1_out = int((self.dgcnn_k*self.last_gnn_layer_dim + 2*self.padding -2)/(self.strides[0]) + 1) # self.node_feat_size should change to the last GNN layer dimension
        dim_conv1_out = dim_conv1_out/2

        dim_conv2_out = int((dim_conv1_out - self.conv1d_kernels[1])/(self.strides[1])+1)

        self.linear1 = nn.Linear(self.hid_channels[1]*dim_conv2_out, self.ffn_layer_size, bias=self.Bias)
        self.linear2 = nn.Linear(self.ffn_layer_size, self.num_class, bias=self.Bias)

        if mlp_act_fun == 'ReLu':
            self.mlp_act_fun = F.relu
        self.soft_fun = F.softmax
        self.dropout_linear1 = nn.Dropout(p=dropout_rate)


    def forward(self, sortpooled_embedings, graph_sizes):

        to_conv1d_1 = sortpooled_embedings.view((-1, 1, self.dgcnn_k * self.last_gnn_layer_dim)) # self.node_feat_size should change to the last GNN layer dimension

        conv1d_1_res = self.conv1d_1(to_conv1d_1)
        output_conv1d_1 = self.mlp_act_fun(conv1d_1_res)

        maxpooled_output_conv1d_1 = self.maxpool1d(output_conv1d_1)

        conv1d_2_res = self.conv1d_2(maxpooled_output_conv1d_1)
        output_conv1d_2 = self.mlp_act_fun(conv1d_2_res)

        all_but_last_two_dims = output_conv1d_2.size()[:-2]

        to_dense = output_conv1d_2.view(*all_but_last_two_dims, 1, -1)
        to_dense = torch.squeeze(to_dense, 1)

        ffn_1 = self.linear1(to_dense)
        ffn_1 = self.mlp_act_fun(ffn_1)

        dropout_ffn_1 = self.dropout_linear1(ffn_1)

        ffn_2 = self.linear2(dropout_ffn_1)
        softmaxed_ffn_2 = self.soft_fun(ffn_2, dim=1)

        return output_conv1d_1, maxpooled_output_conv1d_1, output_conv1d_2, to_dense, ffn_1, dropout_ffn_1, ffn_2, softmaxed_ffn_2