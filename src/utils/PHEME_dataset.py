import os
from typing import Optional, Callable
import torch
from torch.utils.data import Dataset,DataLoader
from torch_geometric.data import Dataset, Data

from .graph_utils import GraphConstructor

class PHEMEDataset(Dataset):
    def __init__(
        self,
        root: str="data",
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
        test:bool=True
    ):
        """
        Initialize the PHEME dataset.
        
        Args:
            root (str): Root directory of the dataset
            transform (callable, optional): A function/transform that takes in a Data object and returns a transformed version
            pre_transform (callable, optional): A function/transform to be applied before saving
            pre_filter (callable, optional): A function that takes in a Data object and returns True if the object should be included
        """
        self.graph_constructor = GraphConstructor(test=test)
        self.data_list = None
        super().__init__(root, transform, pre_transform, pre_filter)

    @property
    def raw_file_names(self):
        return ["PHEME"]

    @property
    def processed_file_names(self):
        return ["graphs.pt"]

    def download(self):
        pass

    def process(self):
        """
        Process raw data into PyG Data objects.
        """
        # Construct PyG graphs using graph constructor
        data_list = self.graph_constructor.construct_pyg_graphs()
        self.data_list = data_list

        #Save the data list
        torch.save(data_list, self.processed_paths[0])

    def len(self) -> int:
        return len(self.graph_constructor.data.df)

    def get(self, idx: int) -> Data:
        if not self.data_list:
            data_path = self.processed_paths[0]
            if not os.path.exists(data_path):
                raise ValueError("Processed data not found. Run process() first.")
            self.data_list = torch.load(data_path)
        return self.data_list[idx]
    