import numpy as np
import torch
import torch.nn as nn
from utils.utils import NeighborSampler
from torch.autograd import Variable

class DIMKT(nn.Module):
    def __init__(self, node_raw_features: np.ndarray,
                 edge_raw_features: np.ndarray,
                 dropout: float = 0.5,
                 dataset_name:str = 'assist17',
                 device: str = 'cuda:0'):

        super(DIMKT, self).__init__()
        OUT_DIFF_FEAT = './processed_data/{}/ml_{}_node_diff.npy'.format(dataset_name, dataset_name)
        OUT_FEATURE_DIFF_FEAT = './processed_data/{}/ml_{}_skill_diff.npy'.format(dataset_name, dataset_name)

        self.node_raw_features = torch.from_numpy(node_raw_features.astype(np.float32))  # .to(device)
        self.edge_raw_features = torch.from_numpy(edge_raw_features.astype(np.float32))  # .to(device)

        self.num_skills = int(np.unique(self.node_raw_features[:, 0]).max()) + 1
        self.num_nodes = self.node_raw_features.shape[0]

        self.edge_dim = edge_raw_features.shape[1]  # 还有label一起在edge_raw_feature中 #对LSTM/RNN的输入的数据长度进行设置，edge_raw_features.shape[1]
        self.node_dim = node_raw_features.shape[1]

        self.node_diff = torch.from_numpy(np.load(OUT_DIFF_FEAT)[:,np.newaxis].astype(np.int32))
        self.skill_diff = torch.from_numpy(np.load(OUT_FEATURE_DIFF_FEAT)[:,np.newaxis].astype(np.int32))

        self.projection_layer = nn.ModuleDict({
            'edge': nn.Embedding(2, self.edge_dim),
            'node': nn.Embedding(self.num_nodes, self.node_dim),
            'skill': nn.Embedding(self.num_skills, self.node_dim),
            'node_diff': nn.Embedding(102, self.node_dim),
            'skill_diff': nn.Embedding(102, self.node_dim)
        })

        self.linear_layer = nn.ModuleDict({
            'input': nn.Linear(4*self.node_dim, self.node_dim),
            'gates_SDF':nn.Linear(self.node_dim, self.node_dim),
            'SDF':nn.Linear(self.node_dim, self.node_dim),
            'gates_PKA':nn.Linear(2*self.node_dim, self.node_dim),
            'PKAt':nn.Linear(2*self.node_dim, self.node_dim),
            'KSU':nn.Linear(4*self.node_dim, self.node_dim)
        })

        self.dropout = nn.Dropout(dropout)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

        self.knowledge = nn.Parameter(nn.init.xavier_uniform_(torch.empty(1, self.node_dim)), requires_grad=True).to(device)

        self.device = device

    def set_neighbor_sampler(self, neighbor_sampler: NeighborSampler):
        """
        set neighbor sampler to neighbor_sampler and reset the random state (for reproducing the results for uniform and time_interval_aware sampling)
        :param neighbor_sampler: NeighborSampler, neighbor sampler
        :return:
        """
        self.neighbor_sampler = neighbor_sampler


    def compute_src_dst_node_temporal_embeddings(self, src_node_ids: np.ndarray,edge_ids:np.ndarray,
                                                 node_interact_times: np.ndarray, dst_node_ids: np.ndarray):
        #    def get_historical_neighbors(self, node_ids: np.ndarray, node_interact_times: np.ndarray, num_neighbors: int = 20):

        src_neighbor_node_ids, src_neighbor_edge_ids, src_neighbor_times = self.neighbor_sampler.get_historical_neighbors(
            src_node_ids, node_interact_times, 1)
        src_neighbor_node_ids  =np.squeeze(src_neighbor_node_ids,1)

        knowledge = self.knowledge.repeat(src_node_ids.shape[0],1)

        nodes_embedding, skill_embedding, nodes_diff_embedding, nodes_skill_diff_embedding = self.get_features(nodes_ids=src_neighbor_node_ids)
        edge_embedding = self.projection_layer['edge'](self.edge_raw_features[torch.from_numpy(src_neighbor_edge_ids.squeeze(1))][:,0].long().to(self.device))
        # print(nodes_embedding.shape,skill_embedding.shape,nodes_diff_embedding.shape,nodes_skill_diff_embedding.shape,edge_embedding.shape)
        src_node_embeddings = torch.cat((nodes_embedding,skill_embedding,nodes_diff_embedding,nodes_skill_diff_embedding),dim=-1)
        src_node_embeddings = self.linear_layer['input'](src_node_embeddings)

        src_node_embeddings = knowledge - src_node_embeddings
        gates_SDF = self.sigmoid(self.linear_layer['gates_SDF'](src_node_embeddings))
        SDFt = self.dropout(self.tanh(self.linear_layer['SDF'](src_node_embeddings)))
        SDFt = gates_SDF*SDFt

        x = torch.cat((SDFt, edge_embedding),-1)
        gates_PKA = self.sigmoid(self.linear_layer['gates_PKA'](x))
        PKAt = self.tanh(self.linear_layer['PKAt'](x))
        PKAt = gates_PKA*PKAt

        ins = torch.cat((knowledge,edge_embedding,nodes_diff_embedding, nodes_skill_diff_embedding), dim=-1)
        gates_KSU = self.sigmoid(self.linear_layer['KSU'](ins))

        knowledge = gates_KSU*knowledge + (1-gates_KSU)*PKAt
        src_node_embeddings = knowledge
        
        dst_node_embeddings = self.linear_layer['input'](torch.cat((self.get_features(nodes_ids=dst_node_ids)),dim=-1))

        # print(src_node_embeddings.shape,'\n',dst_node_embeddings.shape)

        return src_node_embeddings,dst_node_embeddings

    def get_features(self, nodes_ids: np.ndarray):
        #[2000, 128]
        skill_ids = self.node_raw_features[nodes_ids][:,0].long()

        nodes_ids = torch.from_numpy(nodes_ids).long()

        nodes_embedding = self.projection_layer['node'](Variable(nodes_ids.to(self.device)))
        
        skill_embedding = self.projection_layer['skill'](Variable(skill_ids.to(self.device)))

        nodes_diff_embedding = self.projection_layer['node_diff'](
            Variable(self.node_diff[nodes_ids].to(self.device)))
        
        nodes_skill_diff_embedding = self.projection_layer['skill_diff'](
           Variable(self.skill_diff[skill_ids].to(self.device)))

        return nodes_embedding.squeeze(1), skill_embedding.squeeze(1), nodes_diff_embedding.squeeze(1).squeeze(1), nodes_skill_diff_embedding.squeeze(1).squeeze(1)