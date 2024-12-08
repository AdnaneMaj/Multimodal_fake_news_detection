import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool,global_max_pool

class KMGCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim,pool:str):
        """
        Two-layer Graph Convolutional Network for classification
        
        Args:
        input_dim (int): Dimension of input node features
        hidden_dim (int): Dimension of hidden layer
        output_dim (int): Dimension of output (number of classes)
        """
        super().__init__()
        self.pool = pool
        self.pool_dict = {'avg':global_mean_pool,'max':global_max_pool}
        
        # First GCN layer
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.bn1 = torch.nn.BatchNorm1d(hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim // 2)
        self.bn2 = torch.nn.BatchNorm1d(hidden_dim // 2)
        self.dropout = torch.nn.Dropout(0.3)

        # Fully Connected Layers
        self.fc1 = torch.nn.Linear(hidden_dim//2, 128)
        self.fc2 = torch.nn.Linear(128, 64)
        self.fc3 = torch.nn.Linear(64, output_dim)

    def forward(self, data):
        x, edge_index, batch = data.x.float(), data.edge_index.long(), data.batch.long()
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.pool_dict[self.pool](x, batch)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x