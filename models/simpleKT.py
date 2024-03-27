import numpy as np
import torch
import torch.nn as nn
from utils.utils import NeighborSampler
from models.modules import TimeEncoder, TransformerEncoder, TimeDecayEncoder, TimeDualDecayEncoder, multiParallelEncoder
import seaborn as sns

class SimpleKT(nn.Module):
    def __init__(self, node_raw_features: np.ndarray,
                 edge_raw_features: np.ndarray,
                 time_dim=16,
                 num_neighbors: int = 20,
                 dropout: float = 0.5,
                 device: str = 'cuda:0'):

        super(SimpleKT, self).__init__()
        self.num_neighbors = num_neighbors

        self.node_raw_features = torch.from_numpy(node_raw_features.astype(np.float32))  # .to(device)
        self.edge_raw_features = torch.from_numpy(edge_raw_features.astype(np.float32))  # .to(device)

        self.num_skills = int(np.unique(self.node_raw_features[:, 0]).max()) + 1
        self.num_nodes = self.node_raw_features.shape[0]

        self.edge_dim = edge_raw_features.shape[1]  # 还有label一起在edge_raw_feature中 #对LSTM/RNN的输入的数据长度进行设置，edge_raw_features.shape[1]
        self.node_dim = node_raw_features.shape[1]
        self.time_dim = time_dim

        self.projection_layer = nn.ModuleDict({
            'edge': nn.Linear(in_features=1, out_features=self.node_dim, bias=True),
            'skill': nn.Embedding(self.num_skills, self.node_dim),
            'feature': nn.Embedding(self.num_skills, self.node_dim),
            'time': nn.Linear(in_features=self.time_dim, out_features=self.node_dim, bias=True),
            'diff': nn.Embedding(self.num_nodes, self.node_dim),
        })

        self.dropout = dropout
        self.device = device

        self.transformer1 = TransformerEncoder(attention_dim=self.node_dim, num_heads=2,
                                                     dropout=self.dropout)

        self.transformer2 = TransformerEncoder(attention_dim=self.node_dim, num_heads=2,
                                               dropout=self.dropout)

        self.output_layer = nn.Linear(in_features=self.node_dim, out_features=self.node_dim, bias=True)

        self.time_encoder = TimeEncoder(time_dim=self.time_dim) 

    def set_neighbor_sampler(self, neighbor_sampler: NeighborSampler):
        """
        set neighbor sampler to neighbor_sampler and reset the random state (for reproducing the results for uniform and time_interval_aware sampling)
        :param neighbor_sampler: NeighborSampler, neighbor sampler
        :return:
        """
        self.neighbor_sampler = neighbor_sampler

    def compute_src_dst_node_temporal_embeddings(self, src_node_ids: np.ndarray, edge_ids: np.ndarray,
                                                 node_interact_times: np.ndarray, dst_node_ids: np.ndarray):
        #    def get_historical_neighbors(self, node_ids: np.ndarray, node_interact_times: np.ndarray, num_neighbors: int = 20):
        src_neighbor_node_ids, src_neighbor_edge_ids, src_neighbor_times = self.neighbor_sampler.get_historical_neighbors(
            src_node_ids, node_interact_times, self.num_neighbors)

        # src_neighbor_node_ids, ndarray, shape (batch_size, num_neighbors + 1)
        src_neighbor_node_ids = np.concatenate((src_neighbor_node_ids, dst_node_ids[:, np.newaxis]), axis=1)
        # src_neighbor_edge_ids, ndarray, shape (batch_size, num_neighbors + 1)
        src_neighbor_edge_ids = np.concatenate(
            (src_neighbor_edge_ids, np.zeros((len(src_node_ids), 1)).astype(np.longlong)), axis=1)
        # src_neighbor_times, ndarray, shape (batch_size, num_neighbors + 1)
        src_neighbor_times = np.concatenate((src_neighbor_times, node_interact_times[:, np.newaxis]), axis=1)

        nodes_neighbor_node_raw_features, node_diffs, node_skill_features, nodes_edge_raw_features, nodes_neighbor_time_features = self.get_features(
            node_interact_times=node_interact_times, nodes_edge_ids=src_neighbor_edge_ids,
            nodes_neighbor_ids=src_neighbor_node_ids, nodes_neighbor_times=src_neighbor_times)

        x_features = nodes_neighbor_node_raw_features + node_diffs * node_skill_features + nodes_neighbor_time_features  # + src_nodes_neighbor_struct_features  # torch.cat((src_nodes_neighbor_node_raw_features, src_nodes_edge_raw_features),dim=-1) # 该生做过的题目的题号和作对与否

        y_features = nodes_neighbor_node_raw_features + nodes_edge_raw_features + nodes_neighbor_time_features  # + src_nodes_neighbor_struct_features  # torch.cat((src_nodes_neighbor_node_raw_features, src_nodes_edge_raw_features),dim=-1) # 该生做过的题目的题号和作对与否

        src_node_embeddings = self.transformer1(inputs_query=x_features, inputs_key=x_features,
                                                      inputs_value=y_features)

        src_node_embeddings = self.transformer2(inputs_query=src_node_embeddings, inputs_key=src_node_embeddings,
                                       inputs_value=y_features)

        src_node_embeddings = self.output_layer(src_node_embeddings[:, -1]).squeeze(1)
        dst_node_embeddings = self.output_layer(x_features[:, -1])

        return src_node_embeddings, dst_node_embeddings

    def get_features(self, node_interact_times: np.ndarray, nodes_neighbor_ids: np.ndarray, nodes_edge_ids: np.ndarray,
                     nodes_neighbor_times: np.ndarray):
        # one hot node feature lead to bad performance; reason :
        # nodes_neighbor_node_raw_features = self.projection_layer['node'](torch.from_numpy(nodes_neighbor_ids).unsqueeze(-1).float().to(self.device
        nodes_neighbor_ids = torch.from_numpy(nodes_neighbor_ids)

        nodes_neighbor_node_raw_features = self.projection_layer['feature'](
            self.node_raw_features[nodes_neighbor_ids][:,:,0].long().to(self.device)) # z_ck

        node_skill_features = self.projection_layer['skill'](
            self.node_raw_features[nodes_neighbor_ids][:,:,0].long().to(self.device)) # v_ck

        nodes_neighbor_time_features = self.time_encoder(
            torch.from_numpy(node_interact_times[:, np.newaxis] - nodes_neighbor_times).float().to(self.device))

        nodes_neighbor_time_features = self.projection_layer['time'](nodes_neighbor_time_features) # position emb

        nodes_edge_raw_features = self.projection_layer['edge'](self.edge_raw_features[torch.from_numpy(nodes_edge_ids)].to(self.device)[:,:,0].unsqueeze(-1)) #r_qi

        node_diffs = self.projection_layer['diff'](nodes_neighbor_ids.long().to(self.device)) # m_qi

        return nodes_neighbor_node_raw_features, node_diffs, node_skill_features, nodes_edge_raw_features, nodes_neighbor_time_features