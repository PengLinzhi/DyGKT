import numpy as np
import torch
import torch.nn as nn
from utils.utils import NeighborSampler
from models.modules import TimeEncoder

class CTNCM(nn.Module):
    def __init__(self, node_raw_features: np.ndarray,
                 edge_raw_features: np.ndarray,
                 time_dim=16,
                 num_neighbors: int = 20,
                 dropout: float = 0.5,
                 device: str = 'cuda:0'):

        super(CTNCM, self).__init__()
        self.num_neighbors = num_neighbors

        self.node_raw_features = torch.from_numpy(node_raw_features.astype(np.float32))  # .to(device)
        self.edge_raw_features = torch.from_numpy(edge_raw_features.astype(np.float32))  # .to(device)

        self.num_skills = int(np.unique(self.node_raw_features[:, 0]).max()) + 1
        self.num_nodes = self.node_raw_features.shape[0]

        self.edge_dim = edge_raw_features.shape[1]
        self.node_dim = node_raw_features.shape[1]
        self.time_dim = time_dim

        self.projection_layer = nn.ModuleDict({
            'edge': nn.Embedding(2*self.num_nodes+1, self.node_dim),
            'feature': nn.Embedding(self.num_skills, self.node_dim),
            'time': nn.Linear(in_features=self.time_dim, out_features=self.node_dim, bias=True),
            'diff': nn.Embedding(self.num_nodes, self.node_dim),
            'disc': nn.Embedding(self.num_nodes, self.node_dim),
        })

        self.dropout = dropout
        self.device = device

        self.src_node_updater = CTNCM_encoder(edge_dim=self.edge_dim, node_dim=self.node_dim)

        self.output_layer = nn.Linear(in_features=self.node_dim, out_features=self.node_dim, bias=True)

        self.time_encoder = TimeEncoder(time_dim=self.time_dim)  # TimeEncoder(time_dim=self.time_dim)

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

        src_nodes_neighbor_node_raw_features, src_nodes_edge_raw_features, src_nodes_neighbor_time_features = self.get_features(
            node_interact_times=node_interact_times, nodes_edge_ids=src_neighbor_edge_ids,
            nodes_neighbor_ids=src_neighbor_node_ids, nodes_neighbor_times=src_neighbor_times)

        src_nodes_features = src_nodes_neighbor_node_raw_features + src_nodes_edge_raw_features + src_nodes_neighbor_time_features
        src_node_embeddings = self.src_node_updater.update(src_nodes_features, src_neighbor_times).float()

        dst_node_embeddings_diff = self.projection_layer['diff'](torch.from_numpy(dst_node_ids).to(self.device))
        dst_node_embeddings_disc = self.projection_layer['disc'](torch.from_numpy(dst_node_ids).to(self.device))

        dst_node_embeddings = (src_node_embeddings - dst_node_embeddings_diff) * dst_node_embeddings_disc #+ self.projection_layer['feature'](self.node_raw_features[torch.from_numpy(dst_node_ids)].to(self.device))

        src_node_embeddings = self.output_layer(src_node_embeddings)
        dst_node_embeddings = self.output_layer(dst_node_embeddings)

        return src_node_embeddings, dst_node_embeddings

    def get_features(self, node_interact_times: np.ndarray, nodes_neighbor_ids: np.ndarray, nodes_edge_ids: np.ndarray,
                     nodes_neighbor_times: np.ndarray):
        # one hot node feature lead to bad performance; reason :
        nodes_neighbor_node_raw_features = self.projection_layer['feature'](
                self.node_raw_features[torch.from_numpy(nodes_neighbor_ids)][:,:,0].long().to(self.device))  # 现在做的题目本身的skill！！

        nodes_neighbor_time_features = self.time_encoder(
            torch.from_numpy(node_interact_times[:, np.newaxis] - nodes_neighbor_times).float().to(self.device))

        nodes_neighbor_time_features = self.projection_layer['time'](nodes_neighbor_time_features)

        nodes_edge_raw_features = self.projection_layer['edge']((2*torch.from_numpy(nodes_neighbor_ids)+self.edge_raw_features[torch.from_numpy(nodes_edge_ids)][:,:,0]).long().to(self.device))

        return nodes_neighbor_node_raw_features, nodes_edge_raw_features, nodes_neighbor_time_features


class CTNCM_encoder(nn.Module):

    def __init__(self, edge_dim: int, node_dim: int):
        super(CTNCM_encoder, self).__init__()
        self.hid_node_updater1 = nn.LSTM(input_size=edge_dim, hidden_size=node_dim, batch_first=True)  # LSTM(768, 256
        self.hid_node_updater2 = nn.LSTM(input_size=edge_dim, hidden_size=node_dim, batch_first=True)
        self.decay_rate_updater = nn.Linear(in_features=node_dim, out_features=1, bias=True)
        self.softplus = nn.Softplus()

        self.learning_func = nn.Sequential(nn.Linear(1, 100),
                                           nn.ReLU(),
                                           nn.Dropout(0.1),
                                           nn.Linear(100, 100),
                                           nn.ReLU(),
                                           nn.Dropout(0.1),
                                           nn.Linear(100, 1))

    def update(self, x: torch.Tensor, t: np.array):
        outputs1, (hidden1, cell1) = self.hid_node_updater1(x)
        outputs2, (hidden2, cell2) = self.hid_node_updater2(x)

        c, c_hat, hidden_states = torch.squeeze(cell1, dim=0), torch.squeeze(cell2, dim=0), torch.squeeze(hidden1,
                                                                                                          dim=0)
        y = self.decay_rate_updater(hidden_states)

        decay_rate = self.softplus(self.decay_rate_updater(hidden_states))
        time_diff = torch.from_numpy(t[:, -1] - t[:, -2]).to(decay_rate.device).unsqueeze(dim=1)

        # c(t) = ¯ci + (ci − c¯i) exp (−δi(t − ti)), t ∈ (ti, ti+1].
        # torch.Size([100, 1]) torch.Size([100, 256])
        time_effect = self.learning_func((-1.0 * decay_rate * time_diff).float())
        # ct = c_hat + (c - c_hat)*(torch.exp(-1.0*decay_rate*time_diff).repeat(1,c.shape[1]))
        ct = c_hat + (c - c_hat) * (time_effect.repeat(1, c.shape[1]))

        return ct