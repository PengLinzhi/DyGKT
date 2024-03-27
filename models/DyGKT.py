import numpy as np
import torch
import torch.nn as nn
from utils.utils import NeighborSampler
from models.modules import TimeEncoder, TimeDualDecayEncoder
import seaborn as sns
# neighbor sampler strategy: recent
# 这一版的思路：将u-i按时间顺序一个一个喂，借助Neighbor获取直接得到它前面的那几个，如果不够就Pad，如果够就neighbor(num_neighbors max_length)来处理
# X:直接输入edge_feature->skill,q + self.num_q * r但是如何把r编进去呢？：把r直接拼接到q上，将r和q作为edge特征直接输入

class DyGKT(nn.Module):
    def __init__(self,node_raw_features: np.ndarray,
                 edge_raw_features: np.ndarray,
                 time_dim = 16,
                 num_neighbors : int = 50,
                 ablation = '-1',
                 dropout : float = 0.5,
                 device:str='cuda:0'):

        super(DyGKT, self).__init__()
        self.num_neighbors = num_neighbors
        self.ablation = ablation

        self.node_raw_features = torch.from_numpy(node_raw_features.astype(np.float32))#.to(device)
        self.edge_raw_features = torch.from_numpy(edge_raw_features.astype(np.float32))#.to(device)

        self.num_skills = int(np.unique(self.node_raw_features[:, 0]).max()) + 1
        self.num_nodes = self.node_raw_features.shape[0]

        self.edge_dim = 64#edge_raw_features.shape[1] # 还有label一起在edge_raw_feature中 #对LSTM/RNN的输入的数据长度进行设置，edge_raw_features.shape[1]
        self.node_dim = 64
        self.time_dim = time_dim

        self.projection_layer = nn.ModuleDict({
            'feature_Linear':nn.Linear(in_features=self.node_raw_features.shape[-1], out_features=self.node_dim, bias=True),
            # 'feature_Embed':nn.Embedding(self.num_skills, self.node_dim),
            # 'node': nn.Embedding(self.num_nodes, self.node_dim),
            'edge': nn.Linear(in_features=1, out_features=self.node_dim, bias=True),
            'time': nn.Linear(in_features=self.time_dim, out_features=self.node_dim, bias=True),
            'struct': nn.Linear(in_features=1, out_features=self.node_dim, bias=True),
        })

        self.output_layer = nn.Linear(in_features=self.node_dim, out_features=self.node_dim, bias=True)

        self.dropout = dropout
        self.device = device

        
        self.src_node_updater = DyKT_Seq(edge_dim=self.edge_dim, node_dim=self.node_dim)
        self.dst_node_updater = DyKT_Seq(edge_dim=self.edge_dim, node_dim=self.node_dim)
        if self.ablation == 'dual':
            self.time_encoder = TimeEncoder(time_dim=self.time_dim)
        else:
            self.time_encoder = TimeDualDecayEncoder(time_dim=self.time_dim)# TimeEncoder(time_dim=self.time_dim)
        
    def set_neighbor_sampler(self,neighbor_sampler: NeighborSampler):
        self.neighbor_sampler = neighbor_sampler

    def compute_src_dst_node_temporal_embeddings(self,src_node_ids: np.ndarray, edge_ids:np.ndarray,
                                                node_interact_times: np.ndarray,dst_node_ids: np.ndarray):
        #    def get_historical_neighbors(self, node_ids: np.ndarray, node_interact_times: np.ndarray, num_neighbors: int = 20):
        src_neighbor_node_ids, src_neighbor_edge_ids, src_neighbor_times = self.neighbor_sampler.get_historical_neighbors(src_node_ids, node_interact_times, self.num_neighbors)
        dst_neighbor_node_ids, dst_neighbor_edge_ids, dst_neighbor_times = self.neighbor_sampler.get_historical_neighbors(dst_node_ids, node_interact_times, self.num_neighbors)

        # src_neighbor_node_ids, ndarray, shape (batch_size, num_neighbors + 1)
        src_neighbor_node_ids = np.concatenate((src_neighbor_node_ids, src_node_ids[:, np.newaxis]), axis=1)
        # src_neighbor_edge_ids, ndarray, shape (batch_size, num_neighbors + 1)
        src_neighbor_edge_ids = np.concatenate((src_neighbor_edge_ids, np.zeros((len(src_node_ids), 1)).astype(np.longlong)), axis=1)
        # src_neighbor_times, ndarray, shape (batch_size, num_neighbors + 1)
        src_neighbor_times = np.concatenate((src_neighbor_times, node_interact_times[:, np.newaxis]), axis=1)

        # dst_neighbor_node_ids, ndarray, shape (batch_size, num_neighbors + 1)
        dst_neighbor_node_ids = np.concatenate(( dst_neighbor_node_ids, dst_node_ids[:, np.newaxis]), axis=1)
        # dst_neighbor_edge_ids, ndarray, shape (batch_size, num_neighbors + 1)
        dst_neighbor_edge_ids = np.concatenate((dst_neighbor_edge_ids, np.zeros((len(dst_node_ids), 1)).astype(np.longlong)), axis=1)
        # dst_neighbor_times, ndarray, shape (batch_size, num_neighbors + 1)
        dst_neighbor_times = np.concatenate((dst_neighbor_times, node_interact_times[:, np.newaxis]), axis=1)

        src_nodes_neighbor_co_occurrence_features = (torch.from_numpy(src_neighbor_node_ids[:,:-1]) == torch.from_numpy(dst_node_ids).unsqueeze(1).repeat(1, self.num_neighbors)).unsqueeze(-1).float().to(self.device)
        dst_nodes_neighbor_co_occurrence_features = (torch.from_numpy(dst_neighbor_node_ids[:,:-1]) == torch.from_numpy(src_node_ids).unsqueeze(1).repeat(1, self.num_neighbors)).unsqueeze(-1).float().to(self.device)

        src_node_skill = self.node_raw_features[torch.from_numpy(src_neighbor_node_ids)][:, :-1, 0].long().to(self.device)
        dst_node_skill = self.node_raw_features[torch.from_numpy(dst_neighbor_node_ids)][:, -1, 0].long().to(self.device).unsqueeze(1).repeat(1, self.num_neighbors)

        src_nodes_neighbor_skill_features = (src_node_skill == dst_node_skill).unsqueeze(-1).float()
        a = 1
        if self.ablation == 'counter':
            a = 0
        
        src_nodes_neighbor_struct_features = self.projection_layer['struct'](a * src_nodes_neighbor_co_occurrence_features)
        dst_nodes_neighbor_struct_features = self.projection_layer['struct'](a * dst_nodes_neighbor_co_occurrence_features)
        src_nodes_neighbor_skill_struct_features = self.projection_layer['struct'](a * src_nodes_neighbor_skill_features)

        src_nodes_neighbor_node_raw_features, src_nodes_edge_raw_features, src_nodes_neighbor_time_features = self.get_features(
            node_interact_times=node_interact_times, nodes_edge_ids=src_neighbor_edge_ids,
            nodes_neighbor_ids=src_neighbor_node_ids, nodes_neighbor_times=src_neighbor_times)
        dst_nodes_neighbor_node_raw_features, dst_nodes_edge_raw_features, dst_nodes_neighbor_time_features = self.get_features(
            node_interact_times=node_interact_times, nodes_edge_ids=dst_neighbor_edge_ids,
            nodes_neighbor_ids=dst_neighbor_node_ids, nodes_neighbor_times=dst_neighbor_times)
        
        src_nodes_features = src_nodes_neighbor_node_raw_features + src_nodes_edge_raw_features + src_nodes_neighbor_time_features #+ src_nodes_neighbor_struct_features  # torch.cat((src_nodes_neighbor_node_raw_features, src_nodes_edge_raw_features),dim=-1) # 该生做过的题目的题号和作对与否
        dst_nodes_features = dst_nodes_neighbor_node_raw_features + dst_nodes_edge_raw_features + dst_nodes_neighbor_time_features #+ dst_nodes_neighbor_struct_features  # torch.cat((dst_nodes_neighbor_node_raw_features, dst_nodes_edge_raw_features), dim=-1) # 做题学生和做对与否 编码题目？？题号去哪里了

        src_node_embeddings = self.src_node_updater.update(
            src_nodes_features[:, :-1, :] + src_nodes_neighbor_skill_struct_features+ src_nodes_neighbor_struct_features) + (src_nodes_edge_raw_features + src_nodes_neighbor_time_features)[:,-1, :]
        
        if self.ablation in ['q_qid', 'q_kid']:
            dst_node_embeddings = dst_nodes_neighbor_node_raw_features[:,-1]
        else:
            dst_node_embeddings = self.dst_node_updater.update((dst_nodes_edge_raw_features + dst_nodes_neighbor_time_features)[:, :-1, :]+ dst_nodes_neighbor_struct_features) + dst_nodes_features[:,-1,:] 


        

        src_node_embeddings = self.output_layer(src_node_embeddings)
        dst_node_embeddings = self.output_layer(dst_node_embeddings)
        
        return src_node_embeddings, dst_node_embeddings


    def get_features(self, nodes_neighbor_ids: np.ndarray, nodes_edge_ids: np.ndarray,
                     nodes_neighbor_times: np.ndarray,node_interact_times: np.ndarray):
        
        if self.ablation in ['embed', 'q_kid']:
            nodes_neighbor_node_raw_features = self.projection_layer['feature_Embed'](self.node_raw_features[torch.from_numpy(nodes_neighbor_ids)].to(self.device)[:,:,0].long()) # 现在做的题目本身的skill！！
        elif self.ablation == 'q_qid':
            nodes_neighbor_node_raw_features = self.projection_layer['node'](torch.from_numpy(nodes_neighbor_ids).to(self.device))  # 现在做的题目本身的skill！！
        else:
            nodes_neighbor_node_raw_features = self.projection_layer['feature_Linear'](self.node_raw_features[torch.from_numpy(nodes_neighbor_ids)].to(self.device)) # 现在做的题目本身的skill！！
        
        if self.ablation == 'dual':
            nodes_neighbor_time_features = self.time_encoder(torch.from_numpy(node_interact_times[:,np.newaxis]-nodes_neighbor_times).float().to(self.device))
        else:   
            nodes_neighbor_time_features = self.time_encoder(torch.from_numpy(nodes_neighbor_times).float().to(self.device))
        
        nodes_neighbor_time_features = self.projection_layer['time'](nodes_neighbor_time_features)
        nodes_edge_raw_features = self.projection_layer['edge'](self.edge_raw_features[torch.from_numpy(nodes_edge_ids)].to(self.device)[:,:,0].unsqueeze(-1)) #self.projection_layer['edge'](
        
        if self.ablation == 'time':
            nodes_neighbor_time_features *= 0
        elif self.ablation == 'skill':
            nodes_neighbor_node_raw_features *= 0
        
        return nodes_neighbor_node_raw_features, nodes_edge_raw_features, nodes_neighbor_time_features
    
class DyKT_Seq(nn.Module):

    def __init__(self, edge_dim : int,node_dim: int):
        """
        GRU-based memory updater.
        :param memory_bank: MemoryBank
        :param message_dim: int, dimension of node messages
        :param memory_dim: int, dimension of node memories
        """
        super(DyKT_Seq,self).__init__()
        self.patch_enc_layer = nn.Linear(edge_dim, node_dim)

        self.hid_node_updater = nn.GRU(input_size=edge_dim, hidden_size=node_dim,batch_first=True)# LSTM


    def update(self, x):
        outputs, hidden = self.hid_node_updater(x)

        return torch.squeeze(hidden,dim=0)

