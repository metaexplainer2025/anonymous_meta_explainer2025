import torch
import torch.nn as nn
import torch.nn.functional as F


class SortPooling_for_BMM(nn.Module):
    def __init__(self, sort_pooling_k):
        super(SortPooling_for_BMM, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.sort_pooling_k = sort_pooling_k

    def forward(self, output_of_dgcnn_layer):

        max_values = output_of_dgcnn_layer.max(dim=2).values
        sorted_indices = max_values.argsort(dim=1, descending=True)
        sorted_output_of_dgcnn_layer = torch.gather(output_of_dgcnn_layer, 1,
                                                    sorted_indices.unsqueeze(-1).expand(-1, -1,
                                                                                        output_of_dgcnn_layer.size(-1)))
        top_k_node_features = sorted_output_of_dgcnn_layer[:, :self.sort_pooling_k, :]

        Batch_Size, num_Nodes, num_Node_Features = top_k_node_features.shape
        if num_Nodes == self.sort_pooling_k:
            return top_k_node_features

        if num_Nodes < self.sort_pooling_k:
            pad_offset = self.sort_pooling_k - num_Nodes
            top_k_node_features = F.pad(top_k_node_features, pad=(0, 0, 0, pad_offset), mode="constant", value=0)
        else:
            top_k_node_features = top_k_node_features[:, :self.sort_pooling_k, :]

        return top_k_node_features