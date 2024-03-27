import torch
import numpy as np
import torch.nn as nn
from collections import defaultdict

from utils.utils import NeighborSampler
from models.modules import TimeEncoder, MergeLayer, MultiHeadAttention, TimeDualDecayEncoder
from models.DKTModel import DKT, AKT

class KTMemoryModel(torch.nn.Module):

    def __init__(self, node_raw_features: np.ndarray, edge_raw_features: np.ndarray, neighbor_sampler: NeighborSampler,
                 time_feat_dim: int, model_name: str = 'DKTMemory', dropout: float = 0.1,
                 src_node_mean_time_shift: float = 0.0, src_node_std_time_shift: float = 1.0, 
                 dst_node_mean_time_shift_dst: float = 0.0, dst_node_std_time_shift: float = 1.0, 
                 device: str = 'cpu',num_neighbors:int = 50, num_skills:int = 256):
        """
        General framework for memory-based models, support TGN, DyRep and JODIE.
        :param node_raw_features: ndarray, shape (num_nodes + 1, node_feat_dim)
        :param edge_raw_features: ndarray, shape (num_edges + 1, edge_feat_dim)
        :param neighbor_sampler: NeighborSampler, neighbor sampler
        :param time_feat_dim: int, dimension of time features (encodings)
        :param model_name: str, name of memory-based models, could be TGN, DyRep or JODIE
        :param num_layers: int, number of temporal graph convolution layers
        :param num_heads: int, number of attention heads
        :param dropout: float, dropout rate
        :param src_node_mean_time_shift: float, mean of source node time shifts
        :param src_node_std_time_shift: float, standard deviation of source node time shifts
        :param dst_node_mean_time_shift_dst: float, mean of destination node time shifts
        :param dst_node_std_time_shift: float, standard deviation of destination node time shifts
        :param device: str, device
        """
        super(KTMemoryModel, self).__init__()

        self.node_raw_features = torch.from_numpy(node_raw_features.astype(np.float32))# .to(device)
        self.edge_raw_features = torch.from_numpy(edge_raw_features.astype(np.float32))# .to(device)

        # node_num = self.node_raw_features.shape[0]
        self.node_feat_dim = self.node_raw_features.shape[1]
        self.edge_feat_dim = self.edge_raw_features.shape[1]
        self.time_feat_dim = time_feat_dim
        self.dropout = dropout
        self.num_neighbors = num_neighbors
        self.num_skills = int(np.unique(self.node_raw_features[:,0]).max()) + 1# num_skills
        # if self.num_skills == 0:
        #     self.num_skills = 256
        print("==========unique.node_raw_features====",self.num_skills,self.node_raw_features.shape)

        self.device = device
        self.src_node_mean_time_shift = src_node_mean_time_shift
        self.src_node_std_time_shift = src_node_std_time_shift
        self.dst_node_mean_time_shift_dst = dst_node_mean_time_shift_dst
        self.dst_node_std_time_shift = dst_node_std_time_shift

        self.model_name = model_name
        # number of nodes, including the padded node
        self.num_nodes = 1#self.node_raw_features.shape[0]
        self.memory_dim = self.node_feat_dim
        # since models use the identity function for message encoding, message dimension is 2 * memory_dim + time_feat_dim + edge_feat_dim
        # self.message_dim = self.memory_dim + self.memory_dim + self.time_feat_dim + self.edge_feat_dim
        # TODO: node + edge + time + memory
        self.time_encoder = TimeDualDecayEncoder(time_dim=time_feat_dim)

        # message module (models use the identity function for message encoding, hence, we only create MessageAggregator)
        # self.message_aggregator = MessageAggregator()

        self.last_node_id = None

        # memory modules

        self.memory_bank = MemoryBank(1, self.memory_dim, self.edge_feat_dim)
            # self.memory_bank.set_memories(node_ids=np.array(range(self.num_nodes)), updated_node_memories=self.node_raw_features)

        self.pos_label = nn.Parameter(torch.zeros(self.node_raw_features.shape[0], 100, 2), requires_grad=False)
        self.skill_pos_label = nn.Parameter(torch.zeros(self.num_skills, 100, 2), requires_grad=False)

        self.model_name = model_name
        if model_name == 'DKTMemory':
            self.src_embedder = DKT(self.edge_feat_dim, self.node_feat_dim)
            self.dst_embedder = DKT(self.edge_feat_dim, self.node_feat_dim)
        elif model_name == 'AKTMemory':
            self.src_embedder = AKT(edge_dim=self.edge_feat_dim)
            self.dst_embedder = DKT(self.edge_feat_dim, self.node_feat_dim)

        self.projection_layer = nn.ModuleDict({
            'feature': nn.Linear(in_features=self.edge_feat_dim, out_features=self.node_feat_dim, bias=True),
            'memory': nn.Linear(in_features=self.memory_dim, out_features=self.node_feat_dim, bias=True),
            'edge': nn.Linear(in_features=self.edge_feat_dim, out_features=self.node_feat_dim, bias=True),
            'time': nn.Linear(in_features=self.time_feat_dim, out_features=self.node_feat_dim, bias=True),
            'struct': nn.Linear(in_features=2, out_features=self.node_feat_dim,bias=True),
            'skill_struct': nn.Linear(in_features=2, out_features=self.node_feat_dim,bias=True)
        })

        self.output_layer = nn.Sequential(
            nn.Linear(in_features=self.node_feat_dim, out_features=self.node_feat_dim, bias=True),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(in_features=self.node_feat_dim, out_features=self.node_feat_dim, bias=True))

    def compute_src_dst_node_temporal_embeddings(self, src_node_ids: np.ndarray, dst_node_ids: np.ndarray, node_interact_times: np.ndarray,
                                                 edge_ids: np.ndarray, edges_are_positive: bool = True, num_neighbors: int = 20):
        """
        compute source and destination node temporal embeddings
        :param src_node_ids: ndarray, shape (batch_size, )
        :param dst_node_ids:: ndarray, shape (batch_size, )
        :param node_interact_times: ndarray, shape (batch_size, )
        :param edge_ids: ndarray, shape (batch_size, )
        :param edges_are_positive: boolean, whether the edges are positive,
        determine whether to update the memories and raw messages for nodes in src_node_ids and dst_node_ids or not
        :param num_neighbors: int, number of neighbors to sample for each node
        :return:
        """

        src_neighbor_node_ids, src_neighbor_edge_ids, src_neighbor_times = self.neighbor_sampler.get_historical_neighbors(
            src_node_ids, node_interact_times, self.num_neighbors)
        dst_neighbor_node_ids, dst_neighbor_edge_ids, dst_neighbor_times = self.neighbor_sampler.get_historical_neighbors(
            dst_node_ids, node_interact_times, self.num_neighbors)

        # src_neighbor_node_ids, ndarray, shape (batch_size, num_neighbors + 1)
        src_neighbor_node_ids = np.concatenate((src_neighbor_node_ids, src_node_ids[:, np.newaxis]), axis=1)
        # src_neighbor_edge_ids, ndarray, shape (batch_size, num_neighbors + 1)
        src_neighbor_edge_ids = np.concatenate(
            (src_neighbor_edge_ids, np.zeros((len(src_node_ids), 1)).astype(np.longlong)), axis=1)
        # src_neighbor_times, ndarray, shape (batch_size, num_neighbors + 1)
        src_neighbor_times = np.concatenate((src_neighbor_times, node_interact_times[:, np.newaxis]), axis=1)

        # dst_neighbor_node_ids, ndarray, shape (batch_size, num_neighbors + 1)
        dst_neighbor_node_ids = np.concatenate((dst_neighbor_node_ids, dst_node_ids[:, np.newaxis]), axis=1)
        # dst_neighbor_edge_ids, ndarray, shape (batch_size, num_neighbors + 1)
        dst_neighbor_edge_ids = np.concatenate(
            (dst_neighbor_edge_ids, np.zeros((len(dst_node_ids), 1)).astype(np.longlong)), axis=1)
        # dst_neighbor_times, ndarray, shape (batch_size, num_neighbors + 1)
        dst_neighbor_times = np.concatenate((dst_neighbor_times, node_interact_times[:, np.newaxis]), axis=1)

        src_batch = torch.arange(src_neighbor_node_ids.shape[0]).unsqueeze(-1).repeat(1, src_neighbor_node_ids.shape[1])
        dst_batch = torch.arange(dst_neighbor_node_ids.shape[0]).unsqueeze(-1).repeat(1, dst_neighbor_node_ids.shape[1])

        self.pos_label[src_neighbor_node_ids.copy().flatten(), src_batch.flatten(), 0] += 1
        self.pos_label[dst_neighbor_node_ids.copy().flatten(), dst_batch.flatten(), 1] += 1

        src_nodes_neighbor_co_occurrence_features = self.pos_label[src_neighbor_node_ids.copy().flatten(),src_batch.flatten()].reshape(src_neighbor_node_ids.shape[0], src_neighbor_node_ids.shape[1], -1)
        dst_nodes_neighbor_co_occurrence_features = torch.flip(self.pos_label[dst_neighbor_node_ids.copy().flatten(),dst_batch.flatten()].reshape(dst_neighbor_node_ids.shape[0], dst_neighbor_node_ids.shape[1], -1), dims=[-1])

        src_nodes_neighbor_struct_features = self.projection_layer['struct'](src_nodes_neighbor_co_occurrence_features)
        dst_nodes_neighbor_struct_features = self.projection_layer['struct'](dst_nodes_neighbor_co_occurrence_features)

        his_skill = self.node_raw_features[torch.from_numpy(src_neighbor_node_ids)][:, :-1, 0].long().to(self.device).flatten()
        new_skill = self.node_raw_features[torch.from_numpy(dst_neighbor_node_ids)][:, -1, 0].long().to(self.device).flatten()

        self.skill_pos_label[his_skill, src_batch[:, :-1].flatten(), 0] += 1
        self.skill_pos_label[new_skill, dst_batch[:,-1].flatten(), 1] += 1

        src_nodes_neighbor_skill_features = self.skill_pos_label[his_skill, src_batch[:, :-1].flatten()].reshape(src_neighbor_node_ids.shape[0],src_neighbor_node_ids.shape[1]-1, -1)
        dst_nodes_neighbor_skill_features = torch.flip(self.skill_pos_label[new_skill, dst_batch[:,-1].flatten()].reshape(dst_neighbor_node_ids.shape[0],-1), dims=[-1])

        src_nodes_neighbor_skill_struct_features = self.projection_layer['skill_struct'](src_nodes_neighbor_skill_features)
        dst_nodes_neighbor_skill_struct_features = self.projection_layer['skill_struct'](dst_nodes_neighbor_skill_features)

        src_nodes_neighbor_node_raw_features, src_nodes_edge_raw_features, src_nodes_neighbor_time_features = self.get_features(
            node_interact_times=node_interact_times, nodes_neighbor_edge_ids=src_neighbor_edge_ids,
            nodes_neighbor_ids=src_neighbor_node_ids, nodes_neighbor_times=src_neighbor_times, dst=False)
        dst_nodes_neighbor_node_raw_features, dst_nodes_edge_raw_features, dst_nodes_neighbor_time_features = self.get_features(
            node_interact_times=node_interact_times, nodes_neighbor_edge_ids=dst_neighbor_edge_ids,
            nodes_neighbor_ids=dst_neighbor_node_ids, nodes_neighbor_times=dst_neighbor_times, dst=True)

        src_nodes_features = src_nodes_neighbor_node_raw_features + src_nodes_edge_raw_features + src_nodes_neighbor_time_features + src_nodes_neighbor_struct_features  # torch.cat((src_nodes_neighbor_node_raw_features, src_nodes_edge_raw_features),dim=-1) # 该生做过的题目的题号和作对与否
        dst_nodes_features = dst_nodes_neighbor_node_raw_features + dst_nodes_edge_raw_features + dst_nodes_neighbor_time_features + dst_nodes_neighbor_struct_features # torch.cat((dst_nodes_neighbor_node_raw_features, dst_nodes_edge_raw_features), dim=-1) # 做题学生和做对与否 编码题目？？题号去哪里了

        if self.model_name == 'DKTMemory':
            src_node_embeddings = self.src_embedder.update(src_nodes_features[:, :-1, :]+src_nodes_neighbor_skill_struct_features) + src_nodes_features[:, -1, :]
            dst_node_embeddings = self.dst_embedder.update(dst_nodes_features[:, :-1, :]) + dst_nodes_features[:, -1, :] + dst_nodes_neighbor_skill_struct_features
            # src_node_embeddings = src_nodes_features.view(src_nodes_features.shape[0], -1)
            # dst_node_embeddings = dst_nodes_features.view(src_nodes_features.shape[0], -1)
            # src_node_embeddings = self.projection_layer['his'](src_node_embeddings)
            # dst_node_embeddings = self.projection_layer['his'](dst_node_embeddings)
        elif self.model_name == 'AKTMemory':
            # src_node_embeddings,dst_node_embeddings = self.embedder.update(src_nodes_features, src_neighbor_node_ids, dst_nodes_features, dst_neighbor_node_ids)
            src_node_embeddings = self.src_embedder.update(src_nodes_features, src_neighbor_node_ids,
                                                           dst_nodes_features, dst_neighbor_node_ids)
            dst_node_embeddings = self.dst_embedder.update(dst_nodes_features[:, :-1, :]) + dst_nodes_features[:, -1, :]

            # src_node_embeddings = src_nodes_features.view(src_nodes_features.shape[0], -1)
            # dst_node_embeddings = dst_nodes_features.view(src_nodes_features.shape[0], -1)
            # src_node_embeddings = self.projection_layer['his'](src_node_embeddings)
            # dst_node_embeddings = self.projection_layer['his'](dst_node_embeddings)
        src_node_embeddings = self.output_layer(src_node_embeddings)
        dst_node_embeddings = self.output_layer(dst_node_embeddings)

        # if edges_are_positive:
        #     with torch.no_grad():
        #         node_message = torch.cat([src_node_embeddings+dst_node_embeddings, torch.from_numpy(np.array(self.edge_raw_features[edge_ids]).astype(np.float32)).to(self.device)], dim=-1)
        #
        #         self.memory_bank.set_memories(src_node_ids, node_message)
        #         self.memory_bank.set_memories(dst_node_ids, node_message)
        #         self.memory_bank.set_memories((np.array(self.node_raw_features[torch.from_numpy(dst_node_ids)][:,0]) + self.num_nodes).astype(np.int64), node_message)

                # self.memory_bank.set_memories(src_neighbor_node_ids[:, :-1].reshape(src_node_ids.shape[0], -1), node_message.unsqueeze(1).tile((1,self.num_neighbors,1)))
                # self.memory_bank.set_memories(dst_neighbor_node_ids[:, :-1].reshape(src_node_ids.shape[0], -1), node_message.unsqueeze(1).tile((1,self.num_neighbors,1)))
        self.pos_label.data.zero_()
        self.skill_pos_label.data.zero_()

        return src_node_embeddings, dst_node_embeddings

    def get_features(self, node_interact_times: np.ndarray, nodes_neighbor_ids: np.ndarray,
                     nodes_neighbor_edge_ids: np.ndarray,
                     nodes_neighbor_times: np.ndarray, dst: False):
        #
        # use memory as node feature:
        # print(self.node_raw_features[torch.from_numpy(nodes_neighbor_ids)][:,:,0].long())
        # skill_memories = self.memory_bank.get_memories_skill(
        #     self.num_nodes + self.node_raw_features[torch.from_numpy(nodes_neighbor_ids)][:, :, 0].long().to(
        #         self.device))
        # #
        # nodes_neighbor_node_memory_features = self.projection_layer['feature'](
        #     skill_memories)  # 现在做的题目本身的skill！！self.node_num+nodes_neighbor_ids
        # # if not dst: nodes_neighbor_node_memory_features = torch.zeros_like(nodes_neighbor_node_memory_features)
        #
        # if dst:
        #     nodes_neighbor_node_memory_features[:, :-1, :] = torch.zeros_like(
        #         nodes_neighbor_node_memory_features[:, :-1, :])
        #     node_memories_neighbor = self.memory_bank.get_memories_student(
        #         torch.from_numpy(nodes_neighbor_ids[:, :-1]).to(self.device))
        #     node_memories_end = self.memory_bank.get_memories_question(
        #         torch.from_numpy(nodes_neighbor_ids[:, -1]).to(self.device).unsqueeze(1))
        # else:
        #     nodes_neighbor_node_memory_features[:, -1, :] = torch.zeros_like(
        #         nodes_neighbor_node_memory_features[:, -1, :])
        #     node_memories_neighbor = self.memory_bank.get_memories_question(
        #         torch.from_numpy(nodes_neighbor_ids[:, :-1]).to(self.device))
        #     node_memories_end = self.memory_bank.get_memories_student(
        #         torch.from_numpy(nodes_neighbor_ids[:, -1]).to(self.device).unsqueeze(1))
        #
        # node_memories = torch.cat([node_memories_neighbor, node_memories_end], dim=1)

        nodes_neighbor_node_raw_features = 0#nodes_neighbor_node_memory_features + self.projection_layer['memory'](node_memories)
        # torch.zeros_like(nodes_neighbor_node_memory_features) #nodes_neighbor_node_memory_features + self.projection_layer['memory'](node_memories[torch.from_numpy(nodes_neighbor_ids).to(self.device)])

        nodes_neighbor_time_intervals = self.time_encoder(
            torch.from_numpy(node_interact_times[:, np.newaxis]).float().to(self.device) - torch.from_numpy(nodes_neighbor_times).float().to(self.device))
        nodes_neighbor_time_features = self.projection_layer['time'](nodes_neighbor_time_intervals)

        nodes_edge_raw_features = self.projection_layer['edge'](
            self.edge_raw_features[torch.from_numpy(nodes_neighbor_edge_ids)].to(
                self.device))  # self.projection_layer['edge'](
        # nodes_neighbor_time_features = torch.zeros_like(nodes_edge_raw_features)
        return nodes_neighbor_node_raw_features, nodes_edge_raw_features, nodes_neighbor_time_features


    def set_neighbor_sampler(self, neighbor_sampler: NeighborSampler):
        """
        set neighbor sampler to neighbor_sampler and reset the random state (for reproducing the results for uniform and time_interval_aware sampling)
        :param neighbor_sampler: NeighborSampler, neighbor sampler
        :return:
        """
        assert self.model_name in ['AKTMemory','DKTMemory','TGN', 'DyRep'], f'Neighbor sampler is not defined in model {self.model_name}!'
        self.neighbor_sampler = neighbor_sampler
        if self.neighbor_sampler.sample_neighbor_strategy in ['uniform', 'time_interval_aware']:
            assert self.neighbor_sampler.seed is not None
            self.neighbor_sampler.reset_random_state()

class MemoryBank(nn.Module):

    def __init__(self, num_nodes: int, node_feat_dim: int, edge_feat_dim : int):
        """
        Memory bank, store node memories, node last updated times and node raw messages.
        :param num_nodes: int, number of nodes
        :param memory_dim: int, dimension of node memories
        """
        super(MemoryBank, self).__init__()
        self.num_nodes = num_nodes
        self.node_feat_dim = node_feat_dim
        self.edge_feat_dim = edge_feat_dim

        # Parameter, treat memory as parameters so that it is saved and loaded together with the model, shape (num_nodes, memory_dim)
        self.node_memories = nn.Parameter(torch.zeros((self.num_nodes, 1, self.edge_feat_dim + self.node_feat_dim)), requires_grad=False)

        self.node_embedding = nn.Embedding(self.num_nodes, self.edge_feat_dim + self.node_feat_dim)

        self.question_mlp = nn.Sequential(nn.Linear(self.edge_feat_dim + self.node_feat_dim, self.node_feat_dim), nn.Sigmoid())
        self.student_mlp = nn.Sequential(nn.Linear(self.edge_feat_dim + self.node_feat_dim, self.node_feat_dim), nn.Sigmoid())
        self.skill_mlp = nn.Sequential(nn.Linear(self.edge_feat_dim + self.node_feat_dim, self.node_feat_dim), nn.Sigmoid())

        self.__init_memory_bank__()

    def __init_memory_bank__(self):
        """
        initialize all the memories and node_last_updated_times to zero vectors, reset the node_raw_messages, which should be called at the start of each epoch
        :return:
        """
        self.node_memories.data.zero_()

    def get_memories_question(self, node_ids):
        """
        get memories for nodes in node_ids
        :param node_ids: ndarray, shape (batch_size, )
        :return:
        """
        return self.question_mlp(torch.sum(self.node_memories[node_ids.to(torch.long)], dim=2)+self.node_embedding(node_ids.to(torch.long)))

    def get_memories_student(self, node_ids):
        """
        get memories for nodes in node_ids
        :param node_ids: ndarray, shape (batch_size, )
        :return:
        """
        return self.student_mlp(torch.sum(self.node_memories[node_ids.to(torch.long)], dim=2)+self.node_embedding(node_ids.to(torch.long)))

    def get_memories_skill(self, node_ids):
        """
        get memories for nodes in node_ids
        :param node_ids: ndarray, shape (batch_size, )
        :return:
        """
        return self.skill_mlp(torch.sum(self.node_memories[node_ids.to(torch.long)], dim=2)+self.node_embedding(node_ids.to(torch.long)))

    def set_memories(self, node_ids: np.ndarray, updated_node_memories: torch.Tensor):
        """
        set memories for nodes in node_ids to updated_node_memories
        :param node_ids: ndarray, shape (batch_size, )
        :param updated_node_memories: Tensor, shape (num_unique_node_ids, memory_dim)
        :return:
        """
        self.node_memories[torch.from_numpy(node_ids).to(torch.long), :-1] = self.node_memories[torch.from_numpy(node_ids).to(torch.long),1:]
        self.node_memories[torch.from_numpy(node_ids).to(torch.long), 0] = updated_node_memories

    def backup_memory_bank(self):
        """
        backup the memory bank, get the copy of current memories, node_last_updated_times and node_raw_messages
        :return:
        """
        return self.node_memories.data.clone()

    def reload_memory_bank(self, backup_memory_bank: tuple):
        """
        reload the memory bank based on backup_memory_bank
        :param backup_memory_bank: tuple (node_memories, node_last_updated_times, node_raw_messages)
        :return:
        """
        self.node_memories.data = backup_memory_bank.clone()

    def detach_memory_bank(self):
        """
        detach the gradients of node memories and node raw messages
        :return:
        """
        self.node_memories.detach_()

    def extra_repr(self):
        """
        set the extra representation of the module, print customized extra information
        :return:
        """
        return 'num_nodes={}, memory_dim={}'.format(self.node_memories.shape[0], self.node_memories.shape[1])

# Memory-related Modules
class MemoryBank2(nn.Module):

    def __init__(self, num_nodes: int, memory_dim: int):
        """
        Memory bank, store node memories, node last updated times and node raw messages.
        :param num_nodes: int, number of nodes
        :param memory_dim: int, dimension of node memories
        """
        super(MemoryBank2, self).__init__()
        self.num_nodes = num_nodes
        self.memory_dim = memory_dim

        # Parameter, treat memory as parameters so that it is saved and loaded together with the model, shape (num_nodes, memory_dim)
        self.node_memories = nn.Parameter(torch.zeros((self.num_nodes, self.memory_dim)), requires_grad=False)
        # Parameter, last updated time of nodes, shape (num_nodes, )
        self.node_last_updated_times = nn.Parameter(torch.zeros(self.num_nodes), requires_grad=False)
        # dictionary of list, {node_id: list of tuples}, each tuple is (message, time) with type (Tensor shape (message_dim, ), a scalar)
        self.node_raw_messages = defaultdict(list)

        self.__init_memory_bank__()

    def __init_memory_bank__(self):
        """
        initialize all the memories and node_last_updated_times to zero vectors, reset the node_raw_messages, which should be called at the start of each epoch
        :return:
        """
        self.node_memories.data.zero_()
        self.node_last_updated_times.data.zero_()
        self.node_raw_messages = defaultdict(list)

    def get_memories(self, node_ids: np.ndarray):
        """
        get memories for nodes in node_ids
        :param node_ids: ndarray, shape (batch_size, )
        :return:
        """
        return self.node_memories[torch.from_numpy(node_ids)]

    def set_memories(self, node_ids: np.ndarray, updated_node_memories: torch.Tensor):
        """
        set memories for nodes in node_ids to updated_node_memories
        :param node_ids: ndarray, shape (batch_size, )
        :param updated_node_memories: Tensor, shape (num_unique_node_ids, memory_dim)
        :return:
        """
        self.node_memories[torch.from_numpy(node_ids)] = updated_node_memories

    def backup_memory_bank(self):
        """
        backup the memory bank, get the copy of current memories, node_last_updated_times and node_raw_messages
        :return:
        """
        cloned_node_raw_messages = {}
        for node_id, node_raw_messages in self.node_raw_messages.items():
            cloned_node_raw_messages[node_id] = [(node_raw_message[0].clone(), node_raw_message[1].copy()) for node_raw_message in node_raw_messages]

        return self.node_memories.data.clone(), self.node_last_updated_times.data.clone(), cloned_node_raw_messages

    def reload_memory_bank(self, backup_memory_bank: tuple):
        """
        reload the memory bank based on backup_memory_bank
        :param backup_memory_bank: tuple (node_memories, node_last_updated_times, node_raw_messages)
        :return:
        """
        self.node_memories.data, self.node_last_updated_times.data = backup_memory_bank[0].clone(), backup_memory_bank[1].clone()

        self.node_raw_messages = defaultdict(list)
        for node_id, node_raw_messages in backup_memory_bank[2].items():
            self.node_raw_messages[node_id] = [(node_raw_message[0].clone(), node_raw_message[1].copy()) for node_raw_message in node_raw_messages]

    def detach_memory_bank(self):
        """
        detach the gradients of node memories and node raw messages
        :return:
        """
        self.node_memories.detach_()

        # Detach all stored messages
        for node_id, node_raw_messages in self.node_raw_messages.items():
            new_node_raw_messages = []
            for node_raw_message in node_raw_messages:
                new_node_raw_messages.append((node_raw_message[0].detach(), node_raw_message[1]))

            self.node_raw_messages[node_id] = new_node_raw_messages

    def store_node_raw_messages(self, node_ids: np.ndarray, new_node_raw_messages: dict):
        """
        store raw messages for nodes in node_ids
        :param node_ids: ndarray, shape (batch_size, )
        :param new_node_raw_messages: dict, dictionary of list, {node_id: list of tuples},
        each tuple is (message, time) with type (Tensor shape (message_dim, ), a scalar)
        :return:
        """
        for node_id in node_ids:
            self.node_raw_messages[node_id].extend(new_node_raw_messages[node_id])


    def clear_node_raw_messages(self, node_ids: np.ndarray):
        """
        clear raw messages for nodes in node_ids
        :param node_ids: ndarray, shape (batch_size, )
        :return:
        """
        for node_id in node_ids:
            self.node_raw_messages[node_id] = []

    def get_node_last_updated_times(self, unique_node_ids: np.ndarray):
        """
        get last updated times for nodes in unique_node_ids
        :param unique_node_ids: ndarray, (num_unique_node_ids, )
        :return:
        """
        return self.node_last_updated_times[torch.from_numpy(unique_node_ids)]

    def extra_repr(self):
        """
        set the extra representation of the module, print customized extra information
        :return:
        """
        return 'num_nodes={}, memory_dim={}'.format(self.node_memories.shape[0], self.node_memories.shape[1])

# Message-related Modules
class MessageAggregator(nn.Module):

    def __init__(self):
        """
        Message aggregator. Given a batch of node ids and corresponding messages, aggregate messages with the same node id.
        """
        super(MessageAggregator, self).__init__()

    def aggregate_messages(self, node_ids: np.ndarray, node_raw_messages: dict):
        """
        given a list of node ids, and a list of messages of the same length,
        aggregate different messages with the same node id (only keep the last message for each node)
        :param node_ids: ndarray, shape (batch_size, )
        :param node_raw_messages: dict, dictionary of list, {node_id: list of tuples},
        each tuple is (message, time) with type (Tensor shape (message_dim, ), a scalar)
        :return:
        """
        unique_node_ids = np.unique(node_ids)
        unique_node_messages = []
        unique_node_timestamps, to_update_node_ids = [], []

        for node_id in unique_node_ids:
        #     if len(node_raw_messages[node_id]) > 0:
        #         to_update_node_ids.append(node_id)
        #         unique_node_messages.append(node_raw_messages[node_id][-1][0])
        #         unique_node_timestamps.append(node_raw_messages[node_id][-1][1])
            if len(node_raw_messages[node_id]) > 0:
                to_update_node_ids.append(node_id)
                unique_node_messages.append(torch.mean(torch.stack([m[0] for m in node_raw_messages[node_id]]), dim=0))
                unique_node_timestamps.append(node_raw_messages[node_id][-1][1])

        # ndarray, shape (num_unique_node_ids, ), array of unique node ids
        to_update_node_ids = np.array(to_update_node_ids)
        # Tensor, shape (num_unique_node_ids, message_dim), aggregated messages for unique nodes
        unique_node_messages = torch.stack(unique_node_messages, dim=0) if len(unique_node_messages) > 0 else torch.Tensor([])
        # ndarray, shape (num_unique_node_ids, ), timestamps for unique nodes
        unique_node_timestamps = np.array(unique_node_timestamps)

        return to_update_node_ids, unique_node_messages, unique_node_timestamps