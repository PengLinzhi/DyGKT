import numpy as np
import torch
import torch.nn as nn
from utils.utils import NeighborSampler
from torch.autograd import Variable
import torch.nn.functional as F
# from modules import multiParallelEncoder
from torch.distributions import Categorical

class IPKT(nn.Module):
    def __init__(self, node_raw_features: np.ndarray,
                 edge_raw_features: np.ndarray,
                 dropout: float = 0.5,
                 num_neighbors:int = 1,
                 device: str = 'cuda:0'):

        super(IPKT, self).__init__()

        
        self.node_raw_features = torch.from_numpy(node_raw_features.astype(np.float32))  # .to(device)
        self.edge_raw_features = torch.from_numpy(edge_raw_features.astype(np.float32))  # .to(device)

        self.num_skills = int(np.unique(self.node_raw_features[:, 0]).max()) + 1
        self.num_nodes = self.node_raw_features.shape[0]

        self.edge_dim = edge_raw_features.shape[1] 
        self.node_dim = node_raw_features.shape[1]

        self.projection_layer = nn.ModuleDict({
            'edge': nn.Embedding(2, self.edge_dim),
            'skill': nn.Linear(self.node_dim, self.node_dim),
            'time':nn.Linear(1,self.node_dim)
        })

        self.linear_layer = nn.ModuleDict({
            'linear1': nn.Linear(3*self.node_dim, self.node_dim),
            'linear2':nn.Linear(4*self.node_dim, self.node_dim,bias=True),
            'linear3':nn.Linear(4*self.node_dim, self.node_dim,bias=True),
            'linear4':nn.Linear(3*self.node_dim, self.node_dim,bias=True),
            'linear5':nn.Linear(4*self.node_dim, self.node_dim,bias=True),
        })

        self.num_neighbors = num_neighbors

        self.dropout = nn.Dropout(dropout)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    
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


        src_neighbor_node_ids, src_neighbor_edge_ids, src_neighbor_times = self.neighbor_sampler.get_historical_neighbors(
            src_node_ids, node_interact_times, 50)
        
        src_neighbor_node_ids = np.concatenate((src_neighbor_node_ids, dst_node_ids[:, np.newaxis]), axis=1)
        src_neighbor_edge_ids = np.concatenate(
            (src_neighbor_edge_ids, np.zeros((len(src_node_ids), 1)).astype(np.longlong)), axis=1)
        src_neighbor_times = np.concatenate((src_neighbor_times, node_interact_times[:, np.newaxis]), axis=1)

        batch_edge_embedding, batch_skill_embedding, batch_interval_embedding = self.get_features(nodes_ids=src_neighbor_node_ids,edge_ids=src_neighbor_edge_ids,node_interact_times=src_neighbor_times)

        batch_src_node_embeddings = torch.cat((batch_skill_embedding, batch_interval_embedding,batch_edge_embedding),dim=-1)

        seq_len = src_neighbor_node_ids.shape[1]
        batch_size= src_neighbor_node_ids.shape[0]
        


        pre_src_node_embeddings = None
        h_pre = nn.init.xavier_uniform_(torch.zeros(batch_size, self.node_dim)).to(self.device)
        h_tilde_pre = None

        for seqi in range(0, seq_len-1):
            src_node_embeddings = batch_src_node_embeddings[:,seqi,:]
            interval_embedding = batch_interval_embedding[:,seqi,:]
            skill_embedding = batch_skill_embedding[:,seqi,:]
            
            # print(pre_src_node_embeddings.shape, src_node_embeddings.shape)
            if pre_src_node_embeddings is None:
                pre_src_node_embeddings = self.linear_layer['linear1'](torch.zeros_like(src_node_embeddings))
            src_node_embeddings = self.linear_layer['linear1'](src_node_embeddings)
            
            if h_tilde_pre is None:
                c_pre = torch.unsqueeze(torch.sum(src_node_embeddings, 1),-1)
                h_tilde_pre = src_node_embeddings*(h_pre).view(batch_size, self.node_dim)/c_pre
            learning_gain = self.tanh(self.linear_layer['linear2'](torch.cat((
                pre_src_node_embeddings, interval_embedding, src_node_embeddings, h_tilde_pre), 1)))
            gamma_l = self.sigmoid(self.linear_layer['linear3'](
                torch.cat((pre_src_node_embeddings, interval_embedding, src_node_embeddings, h_tilde_pre), 1)))
            LG = gamma_l * ((learning_gain + 1) / 2)

            
            LG_tilde = self.dropout(skill_embedding * LG)

            gamma_f = self.sigmoid(self.linear_layer['linear4'](torch.cat((h_pre, LG, interval_embedding), 1)))
            h = LG_tilde + gamma_f * h_pre

            q = self.projection_layer['skill'](self.node_raw_features[dst_node_ids].to(self.device))
            c_tilde = torch.unsqueeze(torch.sum(q, 1),-1)
            h_tilde = q*h / c_tilde
            pre_src_node_embeddings = src_node_embeddings
            h_pre = h
            h_tilde_pre = h_tilde

        dst_node_embeddings = q
        src_node_embeddings = h_tilde

        return h_tilde, dst_node_embeddings

    def get_features(self, nodes_ids: np.ndarray,edge_ids: np.array, node_interact_times: np.array):
        skill_ids = self.node_raw_features[nodes_ids]
        edge_features = self.edge_raw_features[edge_ids][:,:,0].long()
            
        skill_embedding = self.projection_layer['skill'](skill_ids.to(self.device))
        edge_embedding = self.projection_layer['edge']((edge_features.to(self.device)))

        node_interact_times = torch.from_numpy(node_interact_times).to(self.device)
        interval = torch.zeros_like(node_interact_times)
        interval[:,1:] = node_interact_times[:,1:] - node_interact_times[:,:-1]
        interval = interval.unsqueeze(dim=-1)

        interval_embedding = self.projection_layer['time'](interval.float())

        return edge_embedding, skill_embedding, interval_embedding
    


