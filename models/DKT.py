import numpy as np
import torch
import torch.nn as nn
from utils.utils import NeighborSampler
from models.modules import TimeEncoder, TransformerEncoder, TimeDecayEncoder, TimeDualDecayEncoder, multiParallelEncoder
import seaborn as sns

class DKT(nn.Module):
    def __init__(self, node_raw_features: np.ndarray,
                 edge_raw_features: np.ndarray,
                 time_dim=16,
                 num_neighbors: int = 20,
                 dropout: float = 0.5,
                 device: str = 'cuda:0'):

        super(DKT, self).__init__()
        self.num_neighbors = num_neighbors
        self.num_heads = 2

        self.node_raw_features = torch.from_numpy(node_raw_features.astype(np.float32))  # .to(device)
        self.edge_raw_features = torch.from_numpy(edge_raw_features.astype(np.float32))  # .to(device)

        self.num_skills = int(np.unique(self.node_raw_features[:, 0]).max()) + 1
        self.num_nodes = self.node_raw_features.shape[0]

        self.edge_dim = edge_raw_features.shape[1]  
        self.node_dim = node_raw_features.shape[1]
        self.time_dim = time_dim

        self.projection_layer = nn.ModuleDict({
            'edge': nn.Embedding(2*self.num_nodes+1, self.node_dim),
            'time': nn.Linear(in_features=self.time_dim, out_features=self.node_dim, bias=True),
        })

        self.dropout = dropout
        self.device = device

        self.lstm = nn.LSTM(input_size=self.node_dim, hidden_size=self.node_dim,batch_first=True)

        self.output_layer = nn.Linear(in_features=self.node_dim, out_features=self.node_dim, bias=True)

        self.time_encoder = TimeDecayEncoder(time_dim=self.time_dim)  # TimeEncoder(time_dim=self.time_dim)

    def set_neighbor_sampler(self, neighbor_sampler: NeighborSampler):
        """
        set neighbor sampler to neighbor_sampler and reset the random state (for reproducing the results for uniform and time_interval_aware sampling)
        :param neighbor_sampler: NeighborSampler, neighbor sampler
        :return:
        """
        self.neighbor_sampler = neighbor_sampler

    def compute_src_dst_node_temporal_embeddings(self, src_node_ids: np.ndarray, edge_ids: np.ndarray,
                                                 node_interact_times: np.ndarray, dst_node_ids: np.ndarray):
        src_neighbor_node_ids, src_neighbor_edge_ids, src_neighbor_times = self.neighbor_sampler.get_historical_neighbors(
            src_node_ids, node_interact_times, self.num_neighbors)

        # src_neighbor_node_ids, ndarray, shape (batch_size, num_neighbors + 1)
        src_neighbor_node_ids = np.concatenate((src_neighbor_node_ids, dst_node_ids[:, np.newaxis]), axis=1)
        # src_neighbor_edge_ids, ndarray, shape (batch_size, num_neighbors + 1)
        src_neighbor_edge_ids = np.concatenate(
            (src_neighbor_edge_ids, np.zeros((len(src_node_ids), 1)).astype(np.longlong)), axis=1)
        # src_neighbor_times, ndarray, shape (batch_size, num_neighbors + 1)
        src_neighbor_times = np.concatenate((src_neighbor_times, node_interact_times[:, np.newaxis]), axis=1)

        nodes_edge_raw_features, nodes_neighbor_time_features = self.get_features(
            node_interact_times=node_interact_times, nodes_edge_ids=src_neighbor_edge_ids,
            nodes_neighbor_ids=src_neighbor_node_ids, nodes_neighbor_times=src_neighbor_times)

        node_features = nodes_edge_raw_features + nodes_neighbor_time_features

        src_node_embeddings, _ = self.lstm(node_features)

        src_node_embeddings = self.output_layer(src_node_embeddings[:, -1, :])
        dst_node_embeddings = self.node_raw_features[torch.from_numpy(dst_node_ids).long()].to(self.device)# src_node_embeddings

        return src_node_embeddings, dst_node_embeddings

    def get_features(self, node_interact_times: np.ndarray, nodes_neighbor_ids: np.ndarray, nodes_edge_ids: np.ndarray,
                     nodes_neighbor_times: np.ndarray):
        nodes_neighbor_time_features = self.time_encoder(
            torch.from_numpy(node_interact_times[:, np.newaxis] - nodes_neighbor_times).float().to(self.device))

        nodes_neighbor_time_features = self.projection_layer['time'](nodes_neighbor_time_features)

        nodes_edge_raw_features = self.projection_layer['edge']((2*torch.from_numpy(nodes_neighbor_ids)+self.edge_raw_features[torch.from_numpy(nodes_edge_ids)][:,:,0]).long().to(self.device))

        return nodes_edge_raw_features, nodes_neighbor_time_features

