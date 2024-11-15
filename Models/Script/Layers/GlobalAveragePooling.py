import torch
import torch.nn as nn
import torch_geometric
class GlobalAveragePooling(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        batch_size, num_nodes, num_features = x.size()

        x_reshaped = x.view(batch_size * num_nodes, num_features)
        batch_tensor = torch.arange(batch_size, device=x.device).repeat_interleave(num_nodes)
        pooled_features = torch_geometric.nn.global_mean_pool(x_reshaped, batch_tensor)

        return pooled_features