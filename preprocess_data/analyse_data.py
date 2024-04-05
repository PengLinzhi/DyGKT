from tqdm import tqdm
import numpy as np
import warnings
import shutil
import json
import torch
import torch.nn as nn
import sys
import seaborn as sns
from sklearn.metrics import pairwise_distances, jaccard_score
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
sys.path.append("/home/plz/my/DyGLib-master")
from utils.utils import get_neighbor_sampler
from utils.DataLoader import get_idx_data_loader, get_link_classification_data


def similarity_func(u, v):
    intersection = np.intersect1d(u,v)
    union = np.union1d(u,v)
    # print("union.shape",u,v,union.shape,intersection.shape,intersection)
    similarity = intersection.shape[0] / union.shape[0] 
    return similarity


dataset_name = 'EdnetKT1' # 'mooc' #'assisst12' # 'mooc'#'assist17'#'Slepemapy'
node_raw_features, edge_raw_features, full_data, train_data, val_data, test_data, new_node_val_data, new_node_test_data = \
    get_link_classification_data(dataset_name=dataset_name, val_ratio=0.1, test_ratio=0.1)
full_neighbor_sampler = get_neighbor_sampler(data=full_data, sample_neighbor_strategy='recent',
                                                time_scaling_factor=1e-6, seed=1)

src_nodes = np.unique(full_data.src_node_ids)
interact_times = np.full_like(src_nodes,full_data.node_interact_times.max())
src_nodes_neighbor_ids_list, src_nodes_edge_ids_list, src_nodes_neighbor_times_list = \
    full_neighbor_sampler.get_all_first_hop_neighbors(node_ids=src_nodes, node_interact_times=interact_times,
                                                      num=None,node_raw_features=node_raw_features)

# src_nodes_neighbor_ids_list = node_raw_features[src_nodes_neighbor_ids_list][:,0]
lengths = list(map(len, src_nodes_neighbor_ids_list))
max_length = np.max(lengths)
print(max_length, len(src_nodes_neighbor_ids_list))
src_nodes_neighbor_ids_list = [np.pad(arr, (0, max_length - len(arr)), mode='constant') for arr in src_nodes_neighbor_ids_list]
src_nodes_neighbor_ids_list = np.stack(src_nodes_neighbor_ids_list)
# 相似度矩阵
num_arrays = len(src_nodes_neighbor_ids_list)
real_num = 500
random_list = random.sample(range(num_arrays), real_num)
# random_list = [num_arrays-1-i for i in random_list]

similarities = np.zeros((real_num,real_num))
data = np.array([])
# print(src_nodes_neighbor_ids_list[0].shape,np.array(src_nodes_neighbor_ids_list).shape)
# 计算相似度矩阵
# expanded_a = src_nodes_neighbor_ids_list[:,np.newaxis,:]  # 扩展维度以便广播
# expanded_a = np.tile(expanded_a,(1,expanded_a.shape[0],1))
# expanded_b = src_nodes_neighbor_ids_list[np.newaxis,:,:]  # 扩展维度以便广播
# expanded_b = np.tile(expanded_b,(expanded_b.shape[1],1,1))
# combined_array = np.stack((expanded_a, expanded_b), axis=0)
# print(combined_array.shape,expanded_a,expanded_b)
# similarities = np.apply_along_axis(lambda x: similarity_func(x[0], x[1]), axis=-1, arr=combined_array)
# similarities = np.apply_along_axis(lambda x,y: similarity_func(x, y), (expanded_a,expanded_b), axis=0)
# similarities = similarity_func(expanded_a, expanded_b)
# print(expanded_a.shape, expanded_b.shape,similarities)
for i in tqdm(range(real_num)):
    for j in range(real_num):
        # print(node_raw_features[src_nodes_neighbor_ids_list[random_list[i]]][:,0])
        s = similarity_func(node_raw_features[src_nodes_neighbor_ids_list[random_list[i]]][:,0]%256, node_raw_features[src_nodes_neighbor_ids_list[random_list[j]]][:,0]%256)
        # if i == j:print(s)
        similarities[i, j] = s
        data = np.append(data, float(s))
# results = [similarity_func(src_nodes_neighbor_ids_list[i] ,src_nodes_neighbor_ids_list[j]) for i in range(num_arrays) for j in range(i+1, num_arrays)]
# 输出相似度矩阵
print(similarities.shape,similarities)
data = similarities.flatten()
counts, bins = np.histogram(data, bins=np.arange(0, 1.0, 0.05))
# print(similarities)
plt.bar(bins[:-1], counts, width=0.1, align='edge')
plt.xlabel('0-1')
plt.ylabel('num')
plt.title((dataset_name+'each 0.1 num'))
plt.savefig('../processed_data/{}/skill_jac_similarity_histogram{}_{}.png'.format(dataset_name, dataset_name,max_length))
np.save('../processed_data/{}/skill_jac_similarity_{}_{}.npy'.format(dataset_name, dataset_name,max_length),similarities)

