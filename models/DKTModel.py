import numpy as np
import torch
import torch.nn as nn
from utils.utils import NeighborSampler
from models.modules import TimeEncoder, TransformerEncoder, TimeDecayEncoder, TimeDualDecayEncoder, multiParallelEncoder
import seaborn as sns
# neighbor sampler strategy: recent
# 这一版的思路：将u-i按时间顺序一个一个喂，借助Neighbor获取直接得到它前面的那几个，如果不够就Pad，如果够就neighbor(num_neighbors max_length)来处理
# X:直接输入edge_feature->skill,q + self.num_q * r但是如何把r编进去呢？：把r直接拼接到q上，将r和q作为edge特征直接输入

class DKTModel(nn.Module):
    def __init__(self,node_raw_features: np.ndarray,
                 edge_raw_features: np.ndarray,
                 time_dim = 16,
                 num_neighbors : int = 20,
                 dropout : float = 0.5,
                 model_name : str = "DKT",
                 # use_node_neighbor=False, 
                 use_node_features=True,use_time_encoder=False,
                 device:str='cuda:0'):

        super(DKTModel, self).__init__()
        self.num_neighbors = num_neighbors
        # self.use_node_neighbor = use_node_neighbor
        self.use_node_features = use_node_features
        self.use_time_encoder = use_time_encoder

        self.node_raw_features = torch.from_numpy(node_raw_features.astype(np.float32))#.to(device)
        self.edge_raw_features = torch.from_numpy(edge_raw_features.astype(np.float32))#.to(device)

        self.edge_dim = edge_raw_features.shape[1] # 还有label一起在edge_raw_feature中 #对LSTM/RNN的输入的数据长度进行设置，edge_raw_features.shape[1]
        self.node_dim = node_raw_features.shape[1]
        self.time_dim = time_dim

        self.projection_layer = nn.ModuleDict({
            'node': nn.Linear(in_features=1, out_features=self.node_dim, bias=True),
            'feature':nn.Embedding(1500, self.node_dim),
            # 'feature':nn.Sequential(nn.Linear(in_features=self.edge_dim, out_features=self.node_dim, bias=True),nn.ReLU(),
            #                          nn.Linear(in_features=self.node_dim, out_features=self.node_dim, bias=True)),
            # 'feature': nn.Linear(in_features=self.node_dim, out_features=self.node_dim, bias=True),
            'edge': nn.Linear(in_features=self.edge_dim, out_features=self.node_dim, bias=True),
            'time': nn.Linear(in_features=self.time_dim, out_features=self.node_dim, bias=True)
        })

        
        self.output_layer = nn.Linear(in_features=self.node_dim, out_features=self.node_dim, bias=True)
        # self.gather = nn.Linear(in_features=self.edge_dim*3, out_features=self.node_dim, bias=True)

        self.dropout = dropout
        self.model_name = model_name
        self.device = device

        
        if self.model_name == 'DKT':
            self.src_node_updater = DKT(edge_dim=self.edge_dim, node_dim=self.node_dim)
            # self.dst_node_updater = DKT(edge_dim=self.edge_dim, node_dim=self.node_dim)
            self.time_encoder = TimeDualDecayEncoder(time_dim=self.time_dim)# TimeEncoder(time_dim=self.time_dim)
        
        if self.model_name == 'AKT':
            self.src_node_updater = AKT(edge_dim=self.edge_dim)
            self.dst_node_updater = DKT(edge_dim=self.edge_dim, node_dim=self.node_dim)
            self.time_encoder = TimeDualDecayEncoder(time_dim=self.time_dim)# TimeEncoder(time_dim=self.time_dim)

        if self.model_name == 'CTNCM' and self.use_time_encoder:
            self.src_node_updater = DKT(edge_dim=self.edge_dim, node_dim=self.node_dim)
            self.dst_node_updater = DKT(edge_dim=self.edge_dim, node_dim=self.node_dim)
            self.time_encoder = TimeDecayEncoder(time_dim=self.time_dim)

        elif self.model_name == 'CTNCM':
            self.src_node_updater = CTNCM(edge_dim=self.edge_dim, node_dim=self.node_dim)
            self.dst_node_updater = DKT(edge_dim=self.edge_dim, node_dim=self.node_dim)
            self.time_encoder = TimeEncoder(time_dim=self.time_dim)
        
    def set_neighbor_sampler(self,neighbor_sampler: NeighborSampler):
        self.neighbor_sampler = neighbor_sampler

    def compute_src_dst_node_temporal_embeddings(self,src_node_ids: np.ndarray, edge_ids:np.ndarray,
                                                node_interact_times: np.ndarray,dst_node_ids: np.ndarray):




        # print(src_node_ids)
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
        # dst_neighbor_node_ids = np.concatenate(( dst_neighbor_node_ids, dst_node_ids[:, np.newaxis]), axis=1)
        # dst_neighbor_edge_ids, ndarray, shape (batch_size, num_neighbors + 1)
        # dst_neighbor_edge_ids = np.concatenate((dst_neighbor_edge_ids, np.zeros((len(dst_node_ids), 1)).astype(np.longlong)), axis=1)
        # dst_neighbor_times, ndarray, shape (batch_size, num_neighbors + 1)
        # dst_neighbor_times = np.concatenate((dst_neighbor_times, node_interact_times[:, np.newaxis]), axis=1)
    
        src_nodes_neighbor_node_raw_features, src_nodes_edge_raw_features, src_nodes_neighbor_time_features = self.get_features(
            node_interact_times=node_interact_times, nodes_edge_ids=src_neighbor_edge_ids, nodes_neighbor_ids=src_neighbor_node_ids, nodes_neighbor_times=src_neighbor_times)
        dst_nodes_neighbor_node_raw_features, dst_nodes_edge_raw_features, dst_nodes_neighbor_time_features = self.get_features(
            node_interact_times=node_interact_times, nodes_edge_ids=dst_neighbor_edge_ids, nodes_neighbor_ids=dst_neighbor_node_ids, nodes_neighbor_times=dst_neighbor_times)
        
        #print(src_nodes_neighbor_node_raw_features.shape, src_nodes_edge_raw_features.shape, src_nodes_neighbor_time_features.shape)
        # TODO： 拼上memory
        src_nodes_features = src_nodes_neighbor_node_raw_features+src_nodes_edge_raw_features+src_nodes_neighbor_time_features  # torch.cat((src_nodes_neighbor_node_raw_features, src_nodes_edge_raw_features),dim=-1) # 该生做过的题目的题号和作对与否
        dst_nodes_features = dst_nodes_neighbor_node_raw_features+dst_nodes_edge_raw_features+dst_nodes_neighbor_time_features  # torch.cat((dst_nodes_neighbor_node_raw_features, dst_nodes_edge_raw_features), dim=-1) # 做题学生和做对与否 编码题目？？题号去哪里了

        
        if self.model_name == 'DKT':
            src_node_embeddings = self.src_node_updater.update(src_nodes_features[:,:-1,:]) + src_nodes_features[:,-1,:]
            dst_node_embeddings = dst_nodes_features[:,-1,:]
            
        elif self.model_name == 'AKT':
            # src_node_embeddings,dst_node_embeddings = self.node_updater.update(src_nodes_features, src_neighbor_node_ids, dst_nodes_features, dst_neighbor_node_ids)
            src_node_embeddings = self.src_node_updater.update(src_nodes_features, src_neighbor_node_ids, dst_nodes_features, dst_neighbor_node_ids)#  + src_nodes_features[:,-1,:]
            dst_node_embeddings = self.dst_node_updater.update(dst_nodes_features[:,:-1,:]) + dst_nodes_features[:,-1,:]

        elif self.model_name == 'CTNCM' and self.use_time_encoder:
            src_node_embeddings = self.src_node_updater.update(src_nodes_features)# src_neighbor_node_ids, dst_nodes_features, dst_neighbor_node_ids)
            dst_node_embeddings = self.dst_node_updater.update(dst_nodes_features)
            
        elif self.model_name == 'CTNCM':
            src_node_embeddings = self.src_node_updater.update(src_nodes_features, src_neighbor_times).float()# src_neighbor_node_ids, dst_nodes_features, dst_neighbor_node_ids)
            dst_node_embeddings = self.dst_node_updater.update(dst_nodes_features)
        

        src_node_embeddings = self.output_layer(src_node_embeddings)
        dst_node_embeddings = self.output_layer(dst_node_embeddings)
        
        return src_node_embeddings, dst_node_embeddings


    def get_features(self, node_interact_times: np.ndarray, nodes_neighbor_ids: np.ndarray, nodes_edge_ids: np.ndarray,
                     nodes_neighbor_times: np.ndarray):
        # one hot node feature lead to bad performance; reason : 
        # nodes_neighbor_node_raw_features = self.projection_layer['node'](torch.from_numpy(nodes_neighbor_ids).unsqueeze(-1).float().to(self.device))
        # if self.use_node_features:
        nodes_neighbor_node_raw_features = self.projection_layer['feature'](self.node_raw_features[torch.from_numpy(nodes_neighbor_ids)][:,:,0].long().to(self.device)) # 现在做的题目本身的skill！！
        # if not self.use_node_features:
        #     nodes_neighbor_node_raw_features = self.projection_layer['feature'](0*self.node_raw_features[torch.from_numpy(nodes_neighbor_ids)][:,:,0].to(self.device))
        
        nodes_neighbor_time_features = self.time_encoder(torch.from_numpy(node_interact_times[:, np.newaxis] - nodes_neighbor_times).float().to(self.device))
        if self.use_time_encoder:
            nodes_neighbor_time_features = self.projection_layer['time'](nodes_neighbor_time_features)
        else:
            nodes_neighbor_time_features = self.projection_layer['time'](0*nodes_neighbor_time_features)

        nodes_edge_raw_features = self.edge_raw_features[torch.from_numpy(nodes_edge_ids)].to(self.device) #self.projection_layer['edge'](
        
        return nodes_neighbor_node_raw_features, nodes_edge_raw_features, nodes_neighbor_time_features
    
class DKT(nn.Module):

    def __init__(self, edge_dim : int,node_dim: int):
        """
        GRU-based memory updater.
        :param memory_bank: MemoryBank
        :param message_dim: int, dimension of node messages
        :param memory_dim: int, dimension of node memories
        """
        super(DKT,self).__init__()
        self.hid_node_updater = nn.LSTM(input_size=edge_dim, hidden_size=node_dim,batch_first=True)# LSTM

    def update(self, x):
        outputs, (hidden, cell) = self.hid_node_updater(x)
        # outputs, hidden = self.hid_node_updater(x)
        return torch.squeeze(hidden,dim=0)

class CTNCM(nn.Module):

    def __init__(self, edge_dim : int,node_dim: int):

        super(CTNCM,self).__init__()
        self.hid_node_updater1 = nn.LSTM(input_size=edge_dim, hidden_size=node_dim,batch_first=True) # LSTM(768, 256
        self.hid_node_updater2 = nn.LSTM(input_size=edge_dim, hidden_size=node_dim,batch_first=True)
        self.decay_rate_updater = nn.Linear(in_features=node_dim, out_features=1, bias=True)
        self.softplus =  nn.Softplus()

        self.learning_func = nn.Sequential(nn.Linear(1, 100),
                                           nn.ReLU(),
                                           nn.Dropout(0.1),
                                           nn.Linear(100,100),
                                           nn.ReLU(),
                                           nn.Dropout(0.1),
                                           nn.Linear(100,1))
        
    def update(self, x:torch.Tensor, t:np.array):
        outputs1, (hidden1, cell1) = self.hid_node_updater1(x)
        outputs2, (hidden2, cell2) = self.hid_node_updater2(x)
        
        c,c_hat,hidden_states = torch.squeeze(cell1,dim=0),torch.squeeze(cell2,dim=0),torch.squeeze(hidden1,dim=0)
        y = self.decay_rate_updater(hidden_states)

        decay_rate = self.softplus(self.decay_rate_updater(hidden_states))
        time_diff = torch.from_numpy(t[:, -1] - t[:, -2]).to(decay_rate.device).unsqueeze(dim=1)

        #c(t) = ¯ci + (ci − c¯i) exp (−δi(t − ti)), t ∈ (ti, ti+1].
        # print(torch.exp(-1.0*decay_rate*time_diff).shape, c.shape)
        # torch.Size([100, 1]) torch.Size([100, 256])
        time_effect = self.learning_func((-1.0*decay_rate*time_diff).float())
        # ct = c_hat + (c - c_hat)*(torch.exp(-1.0*decay_rate*time_diff).repeat(1,c.shape[1]))
        ct = c_hat + (c - c_hat)*(time_effect.repeat(1,c.shape[1]))
        
        return ct


class AKT(nn.Module):

    def __init__(self, edge_dim : int):   
        super(AKT,self).__init__()
        self.num_layers = 1    
        self.num_heads = 1
        self.dropout = 0.1
        self.edge_dim = edge_dim
        # self.transformers = nn.ModuleList([
        #     TransformerEncoder(attention_dim=self.edge_dim, num_heads=self.num_heads, dropout=self.dropout)
        #     for _ in range(self.num_layers)
        # ])
        self.transformers_skill = TransformerEncoder(attention_dim=self.edge_dim, num_heads=self.num_heads, dropout=self.dropout)
        # self.transformers_skill_2 = TransformerEncoder(attention_dim=self.edge_dim, num_heads=self.num_heads, dropout=self.dropout)
        self.transformers_out = TransformerEncoder(attention_dim=self.edge_dim, num_heads=self.num_heads, dropout=self.dropout)

    def update(self,src_node_features, src_neighbor_node_ids, dst_node_features, dst_neighbor_node_ids):
       # for transformer in self.transformers:
            
            # src_node_embeddings = src_node_features = transformer(inputs_query=src_node_features, inputs_key=src_node_features,
            #                                 inputs_value=src_node_features, neighbor_masks=src_neighbor_node_ids)

            # dst_node_embeddings = dst_node_features = transformer(inputs_query=dst_node_features, inputs_key=dst_node_features,
            #                                 inputs_value=dst_node_features, neighbor_masks=dst_neighbor_node_ids)
            
            # src_node_embeddings = transformer(inputs_query=src_node_features, inputs_key=dst_node_features,
            #                                   inputs_value=dst_node_features, neighbor_masks=dst_neighbor_node_ids)

            # dst_node_embeddings = transformer(inputs_query=dst_node_features, inputs_key=src_node_features,
            #                                   inputs_value=src_node_features, neighbor_masks=src_neighbor_node_ids)

        src_node_embeddings =  self.transformers_skill(inputs_query=src_node_features, inputs_key=src_node_features,
                                            inputs_value=src_node_features, neighbor_masks=src_neighbor_node_ids) 
        src_node_embeddings = self.transformers_out(inputs_query=src_node_embeddings, inputs_key=src_node_embeddings,
                                            inputs_value=dst_node_features, neighbor_masks=dst_neighbor_node_ids) 
        return src_node_embeddings[:, -1, :] # , dst_node_embeddings[:, -1, :]

class HawkesKT(nn.Module):

    def __init__(self, edge_dim : int,node_dim: int):
        """
        GRU-based memory updater.
        :param memory_bank: MemoryBank
        :param message_dim: int, dimension of node messages
        :param memory_dim: int, dimension of node memories
        """
        super(HawkesKT,self).__init__()
        self.hid_node_updater = nn.LSTM(input_size=edge_dim, hidden_size=node_dim,batch_first=True)

    def update(self, x):
        outputs, (hidden, cell) = self.hid_node_updater(x)
        return torch.squeeze(hidden,dim=0)
