import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.utils import NeighborSampler
from models.modules import multiParallelEncoder

class QIKT(nn.Module):
    def __init__(self, node_raw_features: np.ndarray,
                 edge_raw_features: np.ndarray,
                 num_neighbors: int = 20,
                 dropout: float = 0.5,
                 device: str = 'cuda:0'):

        super(QIKT, self).__init__()
        self.num_neighbors = num_neighbors
        self.num_heads = 2

        self.node_raw_features = torch.from_numpy(node_raw_features.astype(np.float32))
        self.edge_raw_features = torch.from_numpy(edge_raw_features.astype(np.float32)) 

        self.num_skills = int(np.unique(self.node_raw_features[:, 0]).max()) + 1
        self.num_nodes = self.node_raw_features.shape[0]

        self.edge_dim = edge_raw_features.shape[1] 
        self.node_dim = node_raw_features.shape[1]

        self.cog_matrix = nn.Parameter(torch.randn(self.node_dim+2, self.node_dim * 2).to(device), requires_grad=True) 
        self.acq_matrix = nn.Parameter(torch.randn(self.node_dim+2, self.node_dim * 2).to(device), requires_grad=True)
        

        self.projection_layer = nn.ModuleDict({
            'node': nn.Embedding(self.num_nodes, self.node_dim),
            'skill': nn.Linear(self.node_dim, self.node_dim),
            'skill_h':nn.Linear(3*self.node_dim,self.node_dim),
            'question_h':nn.Linear(3*self.node_dim,self.node_dim)
        })

        self.dropout = dropout
        self.device = device

        self.out_question_next = multiParallelEncoder(self.node_dim*3,2)
        self.out_question_all = multiParallelEncoder(self.node_dim,2)
        self.out_concept_next = multiParallelEncoder(self.node_dim*3,2)
        self.out_concept_all = multiParallelEncoder(self.node_dim,2)
        self.que_disc = multiParallelEncoder(self.node_dim*2,2)


        self.que_lstm_layer = nn.LSTM(self.node_dim*4, self.node_dim, batch_first=True)
        self.concept_lstm_layer = nn.LSTM(self.node_dim*2, self.node_dim, batch_first=True)
        self.dropout_layer = nn.Dropout(dropout)


        self.src_output = nn.Linear(in_features=2*self.node_dim, out_features=self.node_dim, bias=True)
        self.dst_output = nn.Linear(in_features=2*self.node_dim, out_features=self.node_dim, bias=True)
        self.sigmoid = nn.Sigmoid()

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
            src_node_ids, node_interact_times, self.num_neighbors)

        src_neighbor_node_ids = np.concatenate((src_neighbor_node_ids, dst_node_ids[:, np.newaxis]), axis=1)
        src_neighbor_edge_ids = np.concatenate(
            (src_neighbor_edge_ids, np.zeros((len(src_node_ids), 1)).astype(np.longlong)), axis=1)

        skill_embedding,  node_embedding, edge_embedding = self.get_features(
            src_neighbor_node_ids, src_neighbor_edge_ids)
        
        embed_qc = torch.cat((skill_embedding, node_embedding),-1)
        embed_qca = self.get_features_cat_r(edge_embedding, embed_qc)

        emb_qc_shift = embed_qc[:,1:,:]
        emb_qca_current = embed_qca[:,:-1,:]
        que_h = self.dropout_layer(self.que_lstm_layer(emb_qca_current)[0])
        que_outputs = self.get_outputs(emb_qc_shift=emb_qc_shift,h=que_h,data=node_embedding[:,1:,],model_type="question")
        outputs = que_outputs

        emb_ca = self.get_features_cat_r(edge_embedding, skill_embedding)        
        emb_ca_current = emb_ca[:,:-1,:]

        concept_h = self.dropout_layer(self.concept_lstm_layer(emb_ca_current)[0])
        concept_outputs = self.get_outputs(emb_qc_shift=emb_qc_shift,h=concept_h,data=skill_embedding[:,1:,:],model_type="concept")

        outputs['y_concept_all'] = concept_outputs['y_concept_all']
        outputs['y_concept_next'] = concept_outputs['y_concept_next']
        
        dst_node_embeddings = outputs['y_concept_all'] + outputs['y_concept_next'] + outputs['y_question_all']
        src_node_embeddings = torch.ones_like(dst_node_embeddings)

        return src_node_embeddings, dst_node_embeddings

    def get_features(self, nodes_ids: np.ndarray,edge_ids: np.array):

        skill_ids = self.node_raw_features[nodes_ids]
        edge_features = self.edge_raw_features[edge_ids][:,:,0].long()

        skill_embedding = self.projection_layer['skill'](skill_ids.to(self.device))
        node_embedding = self.projection_layer['node'](torch.from_numpy(nodes_ids).to(self.device))
        edge_embedding = edge_features.unsqueeze(-1).to(self.device)#self.projection_layer['edge']((edge_features.to(self.device)))
        return skill_embedding,  node_embedding, edge_embedding 
    
    def get_features_cat_r(self, r:torch.Tensor, content:torch.Tensor):
        return torch.cat([
                content.mul(r.repeat(1, 1, content.shape[-1]).float()),
                content.mul((1-r).repeat(1, 1, content.shape[-1]).float())],
                dim = -1)

    def get_outputs(self,emb_qc_shift,h,data,model_type='question'):
        outputs = {}
    
        if model_type == 'question':
            h_next = torch.cat([emb_qc_shift,h],axis=-1)
            y_question_next = torch.sigmoid(self.projection_layer['question_h'](self.out_question_next(h_next)))
            y_question_all = torch.sigmoid(self.out_question_all(h))
            outputs["y_question_next"] = y_question_next[:,-1,:]
            outputs["y_question_all"] = (y_question_all * data)[:,-1,:]
        
        else: 
            h_next = torch.cat([emb_qc_shift,h],axis=-1)
            y_concept_next = torch.sigmoid(self.projection_layer['skill_h'](self.out_concept_next(h_next)))
            y_concept_all = torch.sigmoid(self.out_concept_all(h))
            outputs["y_concept_next"] = y_concept_next[:,-1,:]
            outputs["y_concept_all"] = (y_concept_all * data)[:,-1,:]

        return outputs    

