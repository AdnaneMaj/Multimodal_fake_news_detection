import networkx as nx
import re
from collections import Counter
import math
from typing import List
import torch
from torch_geometric.utils import from_networkx
from torch_geometric.data import Data
from tqdm import tqdm

from .data_utils import DatasetCreator
from ..Enums import BaseEnum

"""
___TO-DO___
* Add configuration for constant (enums and config)
* Unify the two self.embedd output (the first [1,786] torch and the second (300,) numpy array)
"""

class GraphConstructor:
    def __init__(self, window_size:int=5,padding:bool=True,embedding:str='bert',test:bool=True):
        """
        Initialize PMI Graph Constructor
        
        :param window_size: Size of sliding window for word co-occurrence (default 5)
        """
        self.window_size:int = window_size
        self.total_windows:int = 0
        self.word_window_counts:Counter = Counter()
        self.word_pair_window_counts:Counter = Counter()
        self.corpus_words:List[str] = []
        self.padding = padding
        self.embedding = embedding

        #Load the embedding model and set the embedding function
        self.embedde = self.create_embedde_func()

        #Create a DatasetCreator
        self.data = DatasetCreator(test=test)

    def create_embedde_func(self):
        """
        Create the embeddng function
        """
        if self.embedding=='bert':
            from transformers import AutoTokenizer, AutoModel
            tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
            model = AutoModel.from_pretrained("bert-base-uncased")
            embedde = lambda word: model(**tokenizer(word, return_tensors="pt")).last_hidden_state.mean(dim=1).detach().numpy().flatten()
        else:
            from gensim.models import KeyedVectors
            model = KeyedVectors.load_word2vec_format(BaseEnum.WORD2VEC_MODEL.value, binary=True)
            embedde = lambda word: model[word]

        return embedde

    def preprocess_text(self, text:str)->None:
        """
        Preprocess tweet text
        
        :param text: Input tweet string
        :return: List of cleaned words
        """
        # Convert to lowercase and remove special characters, then get the token (words)
        text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
        words = text.split()

        #Set the corpus words
        self.corpus_words = words
        self.total_windows = len(words)+self.window_size-3

    def compute_corpus_statistics(self, text:str):
        """
        Compute word and word pair window statistics for entire corpus
        
        :param texts: List of text documents
        :param padding: Either to pad the edges of the text ( if yes the padding value is 0 )
        """
        #prerocess the text to get the corpus words
        self.preprocess_text(text)
        words = self.corpus_words
        
        # Compute sliding window statistics
        shift = self.window_size//2-1 if self.padding else 0
        for i in range(shift,self.total_windows-shift):
            window = words[max(0,i-self.window_size+2):i+2]
            
            # Count word occurrences in windows
            for word in set(window):
                self.word_window_counts[word] += 1
            
            # Count word pair occurrences in windows
            for i,word1 in enumerate(list(set(window))):
                    for word2 in list(set(window))[i+1:]:
                        self.word_pair_window_counts[tuple(sorted((word1,word2)))]+=1

    def compute_pmi(self, word1, word2):
        """
        Compute Point-wise Mutual Information for two words
        
        :param word1: First word
        :param word2: Second word
        :return: PMI score
        """
        # Compute probabilities
        p_w1 = self.word_window_counts[word1] / self.total_windows
        p_w2 = self.word_window_counts[word2] / self.total_windows
        
        word_pair = tuple(sorted((word1, word2)))
        p_joint = self.word_pair_window_counts[word_pair] / self.total_windows
        
        # Compute PMI
        try:
            pmi = math.log(p_joint / (p_w1 * p_w2))
            return pmi
        except:
            return 0
        
    def add_pmi_edges(self,graph:nx.Graph):
        """Add edges to a the graph with the pmi score"""
        # Add edges with positive PMI
        for i in range(len(self.corpus_words)):
            for j in range(i+1, len(self.corpus_words)):
                pmi = self.compute_pmi(self.corpus_words[i], self.corpus_words[j])
                if pmi > 0:
                    graph.add_edge(self.corpus_words[i], self.corpus_words[j], weight=pmi)
  
    def construct_graph(self, row):
        """
        Construct graph for a single text/tweet
        
        :param row: A row of the data frame
        :return: NetworkX graph
        """
        self.compute_corpus_statistics(row['text'])
        G = nx.Graph()
        
        # Add nodes
        G.add_nodes_from((word, {"embedding": self.embedde(word)}) for word in self.corpus_words)

        #Add edges
        self.add_pmi_edges(G)

        #Add graph attributes
        G.graph.update({
            'class':row['class'],
            'subject':row['subject']
        })
        
        return G
    
    def construct_nx_graphs(self) -> List[nx.Graph]:
        """
        Create a list of graph
        """
        graphs = []
        for _,row in tqdm(self.data.df.iterrows(),desc='Constructing nx graphs :'):
            graph = self.construct_graph(row)
            if graph.number_of_edges() > 0:
                graphs.append(graph)
        return graphs
    
    def nx_to_pyg(self,graph:nx.Graph):
        # Convert the networkx graph to a PyG Data object
        data = from_networkx(graph,group_node_attrs='embedding',group_edge_attrs="weights")
        data.y = torch.tensor(graph.graph['class'], dtype=torch.long)
        return data
    
    def construct_pyg_graphs(self) -> List[Data]:
        """nx.Graphs to pyg Data"""
        nx_graphs = self.construct_nx_graphs()
        pyg_data_list = [self.nx_to_pyg(graph) for graph in tqdm(nx_graphs,desc='Converting nx graph to pyg graphs')]
        return pyg_data_list


    def visualize_graph(self, graph):
        """
        Visualize the constructed graph
        
        :param graph: NetworkX graph
        """
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 6))
        pos = nx.spring_layout(graph)
        nx.draw_networkx_nodes(graph, pos, node_color='lightblue')
        nx.draw_networkx_edges(graph, pos)
        nx.draw_networkx_labels(graph, pos)
        plt.title("Word Co-occurrence Graph")
        plt.axis('off')
        plt.show()
