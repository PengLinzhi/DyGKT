import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.utils import NeighborSampler
from models.modules import multiParallelEncoder
from torch.distributions import Categorical

class IEKT(nn.Module):
    def __init__(self, node_raw_features: np.ndarray,
                 edge_raw_features: np.ndarray,
                 num_neighbors: int = 20,
                 dropout: float = 0.5,
                 device: str = 'cuda:0'):

        super(IEKT, self).__init__()
        self.num_neighbors = num_neighbors
        self.num_heads = 2
        self.device = device

        self.node_raw_features = torch.from_numpy(node_raw_features.astype(np.float32))
        self.edge_raw_features = torch.from_numpy(edge_raw_features.astype(np.float32)) 

        self.num_skills = int(np.unique(self.node_raw_features[:, 0]).max()) + 1
        self.num_nodes = self.node_raw_features.shape[0]

        self.edge_dim = edge_raw_features.shape[
            1]  
        self.node_dim = node_raw_features.shape[1]
        

        self.projection_layer = nn.ModuleDict({
            'edge': nn.Embedding(2, self.edge_dim),
            'node': nn.Embedding(self.num_nodes, self.node_dim),
            'skill': nn.Linear(self.node_dim, self.node_dim),
            'skill_cat':nn.Linear(3*self.node_dim,self.node_dim),
            'case_cat':nn.Linear(6*self.node_dim,self.node_dim),
            'classifier':nn.Linear(4*self.node_dim,1)
        })
        self.linear1 = nn.Linear(3*self.node_dim,self.node_dim)
        self.linear2 = nn.Linear(2*self.node_dim,self.node_dim)
        # self, num_q,num_c,emb_size,lamb=40,n_layer=1,cog_levels=10,acq_levels=10,dropout=0,gamma=0.93,device='cpu'):
        self.model = IEKT_h(num_q=self.num_nodes,num_c=self.num_skills,emb_size=self.node_dim, n_layer=2,device=device)


    def set_neighbor_sampler(self, neighbor_sampler: NeighborSampler):
        self.neighbor_sampler = neighbor_sampler

    def compute_src_dst_node_temporal_embeddings(self, src_node_ids: np.ndarray, edge_ids: np.ndarray,
                                                 node_interact_times: np.ndarray, dst_node_ids: np.ndarray):
        src_neighbor_node_ids, src_neighbor_edge_ids, src_neighbor_times = self.neighbor_sampler.get_historical_neighbors(
            src_node_ids, node_interact_times, self.num_neighbors)

        src_neighbor_node_ids = np.concatenate((src_neighbor_node_ids, dst_node_ids[:, np.newaxis]), axis=1)
        src_neighbor_edge_ids = np.concatenate(
            (src_neighbor_edge_ids, np.zeros((len(src_node_ids), 1)).astype(np.longlong)), axis=1)

        skill_embedding,  node_embedding, edge_embedding = self.get_features(
            src_neighbor_node_ids, src_neighbor_edge_ids)

        data_new = {
            'cc':skill_embedding,
            'cq':node_embedding,
            'cr':edge_embedding
        }
        src, dst = self.model.predict_one_step(data_new)
        src_embedding = self.linear1(src)
        dst_embedding = self.linear2(dst)
        return src_embedding, dst_embedding

    def get_features(self, nodes_ids: np.ndarray,edge_ids: np.array):

        skill_ids = self.node_raw_features[nodes_ids]
        edge_features = self.edge_raw_features[edge_ids][:,:,0].long()

        skill_embedding = self.projection_layer['skill'](skill_ids.to(self.device))
        node_embedding = self.projection_layer['node'](torch.from_numpy(nodes_ids).to(self.device))
        edge_embedding = edge_features.to(self.device)
        return skill_embedding,  node_embedding, edge_embedding 


class IEKT_h(nn.Module):
    def __init__(self, num_q,num_c,emb_size,lamb=40,n_layer=1,cog_levels=10,acq_levels=10,dropout=0,gamma=0.93,device='cpu'):
        
        super(IEKT_h, self).__init__()
        self.model = IEKTNet(num_q=num_q,num_c=num_c,lamb=lamb,emb_size=emb_size,n_layer=n_layer,cog_levels=cog_levels,acq_levels=acq_levels,dropout=dropout,gamma=gamma,device=device)

        self.model = self.model.to(device)
        self.device = device
        

    def predict_one_step(self,data):
        sigmoid_func = torch.nn.Sigmoid()
        data_new = data
        
        data_len = data_new['cc'].shape[0]
        seq_len = data_new['cc'].shape[1]
        h = torch.zeros(data_len, self.model.emb_size).to(self.device)

        rt_x = torch.zeros(data_len, 1, self.model.emb_size * 2).to(self.device)
        for seqi in range(0, seq_len):
            ques_h = torch.cat([
                self.model.get_ques_representation(q=data_new['cq'][:,seqi], c=data_new['cc'][:,seqi]),
                h], dim = 1)#equation4

            flip_prob_emb = self.model.pi_cog_func(ques_h)

            m = Categorical(flip_prob_emb)#equation 5 
            emb_ap = m.sample()#equation 5
            emb_p = self.model.cog_matrix[emb_ap,:]#equation 6

            h_v, v, logits, rt_x,src_embedding,dst_embedding = self.model.obtain_v(q=data_new['cq'][:,seqi], c=data_new['cc'][:,seqi], 
                                                        h=h, x=rt_x, emb=emb_p)#equation 7
            prob = sigmoid_func(logits)#equation 7 sigmoid

            out_operate_groundtruth = data_new['cr'][:,seqi].unsqueeze(-1)
            
            out_x_groundtruth = torch.cat([
                h_v.mul(out_operate_groundtruth.repeat(1, h_v.size()[-1]).float()),
                h_v.mul((1-out_operate_groundtruth).repeat(1, h_v.size()[-1]).float())],
                dim = 1)#equation9

            out_operate_logits = torch.where(prob > 0.5, torch.tensor(1).to(self.device), torch.tensor(0).to(self.device)) 
            out_x_logits = torch.cat([
                h_v.mul(out_operate_logits.repeat(1, h_v.size()[-1]).float()),
                h_v.mul((1-out_operate_logits).repeat(1, h_v.size()[-1]).float())],
                dim = 1)#equation10                
            out_x = torch.cat([out_x_groundtruth, out_x_logits], dim = 1)#equation11
            ground_truth = data_new['cr'][:,seqi]
            flip_prob_emb = self.model.pi_sens_func(out_x)#equation12

            m = Categorical(flip_prob_emb)
            emb_a = m.sample()
            emb = self.model.acq_matrix[emb_a,:]#equation12 s_t
            
            h = self.model.update_state(h, v, emb, ground_truth.unsqueeze(1))#equation13～14
        # torch.Size([2000, 96]) torch.Size([2000, 64])
        # print(src_embedding.shape,dst_embedding.shape)
        return src_embedding,dst_embedding


class IEKTNet(nn.Module): 
    def __init__(self, num_q,num_c,emb_size,lamb=40,n_layer=1,cog_levels=10,acq_levels=10,dropout=0,gamma=0.93,device='cpu'):
        super().__init__()
        self.model_name = "iekt"
        self.emb_size = emb_size
        self.concept_num = num_c
        self.device = device
        self.predictor = funcs(n_layer, emb_size * 5, 1, dropout)
        self.cog_matrix = nn.Parameter(torch.randn(cog_levels, emb_size * 2).to(self.device), requires_grad=True) 
        self.acq_matrix = nn.Parameter(torch.randn(acq_levels, emb_size * 2).to(self.device), requires_grad=True)
        self.select_preemb = funcs(n_layer, emb_size * 3, cog_levels, dropout)#MLP
        self.checker_emb = funcs(n_layer, emb_size * 12, acq_levels, dropout) 
        self.prob_emb = nn.Parameter(torch.randn(num_q, emb_size).to(self.device), requires_grad=True)#题目表征
        self.gamma = gamma
        self.lamb = lamb
        self.gru_h = mygru(0, emb_size * 4, emb_size)
        self.concept_emb = nn.Parameter(torch.randn(self.concept_num, emb_size).to(self.device), requires_grad=True)#知识点表征
        self.sigmoid = nn.Sigmoid()

    def get_ques_representation(self, q, c):
        v = torch.cat([q,c],dim=-1)
        return v


    def pi_cog_func(self, x, softmax_dim = 1):
        return F.softmax(self.select_preemb(x), dim = softmax_dim)
    
    def obtain_v(self, q, c, h, x, emb):
        v = self.get_ques_representation(q,c)
        predict_x = torch.cat([h, v], dim = 1)#equation4
        h_v = torch.cat([h, v], dim = 1)#equation4
        prob = self.predictor(torch.cat([
            predict_x, emb
        ], dim = 1))#equation7
        return h_v, v, prob, x, predict_x, emb

    def update_state(self, h, v, emb, operate):
        #equation 13
        v_cat = torch.cat([
            v.mul(operate.repeat(1, self.emb_size * 2)),
            v.mul((1 - operate).repeat(1, self.emb_size * 2))], dim = 1)
        e_cat = torch.cat([
            emb.mul((1-operate).repeat(1, self.emb_size * 2)),
            emb.mul((operate).repeat(1, self.emb_size * 2))], dim = 1)
        inputs = v_cat + e_cat
        
        h_t_next = self.gru_h(inputs, h)#equation14
        return h_t_next
    

    def pi_sens_func(self, x, softmax_dim = 1):
        return F.softmax(self.checker_emb(x), dim = softmax_dim)


class funcs(nn.Module):
    '''
    classifier decoder implemented with mlp
    '''
    def __init__(self, n_layer, hidden_dim, output_dim, dpo):
        super().__init__()

        self.lins = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim)
            for _ in range(n_layer)
        ])
        self.dropout = nn.Dropout(p = dpo)
        self.out = nn.Linear(hidden_dim, output_dim)
        self.act = torch.nn.Sigmoid()

    def forward(self, x):
        for lin in self.lins:
            x = F.relu(lin(x))
        return self.out(self.dropout(x))

class mygru(nn.Module):
    '''
    classifier decoder implemented with mlp
    '''
    def __init__(self, n_layer, input_dim, hidden_dim):
        super().__init__()
        
        this_layer = n_layer
        self.g_ir = funcsgru(this_layer, input_dim, hidden_dim, 0)
        self.g_iz = funcsgru(this_layer, input_dim, hidden_dim, 0)
        self.g_in = funcsgru(this_layer, input_dim, hidden_dim, 0)
        self.g_hr = funcsgru(this_layer, hidden_dim, hidden_dim, 0)
        self.g_hz = funcsgru(this_layer, hidden_dim, hidden_dim, 0)
        self.g_hn = funcsgru(this_layer, hidden_dim, hidden_dim, 0)
        self.sigmoid = torch.nn.Sigmoid()
        self.tanh = torch.nn.Tanh()

    def forward(self, x, h):
        r_t = self.sigmoid(
            self.g_ir(x) + self.g_hr(h)
        )
        z_t = self.sigmoid(
            self.g_iz(x) + self.g_hz(h)
        )
        n_t = self.tanh(
            self.g_in(x) + self.g_hn(h).mul(r_t)
        )
        h_t = (1 - z_t) * n_t + z_t * h
        return h_t

class funcsgru(nn.Module):
    '''
    classifier decoder implemented with mlp
    '''
    def __init__(self, n_layer, hidden_dim, output_dim, dpo):
        super().__init__()

        self.lins = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim)
            for _ in range(n_layer)
        ])
        self.dropout = nn.Dropout(p = dpo)
        self.out = nn.Linear(hidden_dim, output_dim)
        self.act = torch.nn.Sigmoid()

    def forward(self, x):
        for lin in self.lins:
            x = F.relu(lin(x))
        return self.out(self.dropout(x))