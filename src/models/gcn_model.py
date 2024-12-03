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
        
        # Second GCN layer
        self.conv2 = GCNConv(hidden_dim, hidden_dim//2)
        
        # Dropout for regularization
        self.dropout = torch.nn.Dropout(0.5)
        
        # Fully connected layer for classification
        self.fc = torch.nn.Linear(hidden_dim//2, output_dim)
    
    def forward(self, data):
        """
        Forward pass of the GCN

        Args:
        data (Data): PyTorch Geometric data object

        Returns:
        torch.Tensor: Class probabilities
        """
        # Ensure input data is in Float type
        data.x = data.x.float()
        data.edge_index = data.edge_index.long()  # edge_index should be in Long type
        if hasattr(data, 'batch'):
            data.batch = data.batch.long()  # batch indices must also be in Long type

        x, edge_index = data.x, data.edge_index

        # First GCN layer with ReLU and dropout
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)

        # Second GCN layer with ReLU and dropout
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)

        # Global mean pooling to get graph-level representation
        x = self.pool_dict[self.pool](x, data.batch)

        # Final classification layer
        x = self.fc(x)

        return x
