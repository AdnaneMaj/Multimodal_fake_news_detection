import numpy as np
import networkx as nx
import re
from collections import Counter
import math
from typing import List

class PMIGraphConstructor:
    def __init__(self, window_size:int=5,padding:bool=True):
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

    def construct_graph(self, text):
        """
        Construct graph for a single text/tweet
        
        :param text: Input text
        :return: NetworkX graph
        """
        self.compute_corpus_statistics(text)
        words = self.corpus_words
        G = nx.Graph()
        
        # Add nodes
        G.add_nodes_from(words)
        
        # Add edges with positive PMI
        for i in range(len(words)):
            for j in range(i+1, len(words)):
                pmi = self.compute_pmi(words[i], words[j])
                if pmi > 0:
                    G.add_edge(words[i], words[j], weight=pmi)
        
        return G

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