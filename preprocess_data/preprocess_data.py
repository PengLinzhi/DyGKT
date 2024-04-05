import numpy as np
import pandas as pd
from pathlib import Path
import argparse
from pandas.testing import assert_frame_equal
from distutils.dir_util import copy_tree
import time
from datetime import datetime
from scipy import stats
from tqdm import tqdm
import glob
import re
# timestr = '2019-01-14 15:22:18'
# datetime_obj = datetime.strptime(timestr, "%Y-%m-%d %H:%M:%S")
# ret_stamp = int(time.mktime(datetime_obj.timetuple()))
# print(ret_stamp)
# y = pd.read_csv("E:\Programs\Graph0\my\DyGLib-master\DG_data\Slepemapy\\answer.csv",nrows=30)
# m = [p==q for (p,q) in zip(y['place_asked'],y['place_answered'])]
# print(y)
def reindexTime(time_list:list, mode="%Y-%m-%d %H:%M:%S"):
    time_mode = [datetime.strptime(ts[:19], mode) for ts in time_list]
    second_time = [int(time.mktime(datetime_obj.timetuple())) for datetime_obj in time_mode]
    return second_time

def reindexColumn(column:list):
    uni_column = set(column)
    uni_id = set(range(0,len(uni_column)))
    dict_column = dict(zip(uni_column,uni_id))
    return [dict_column[id] for id in column]
# reindexColumnSkill(merged_df['question_id'],problem_dict)
def reindexColumnSkill(column:list, skill:dict):
    uni_column = set(column)
    uni_id = set(range(0,len(uni_column)))
    dict_column = dict(zip(uni_column,uni_id))
    return [dict_column[id] for id in column], {dict_column[v]:skill[v]['tags'] for v in uni_column}
def preprocess(dataset_name: str):
    """
    read the original data file and return the DataFrame that has columns ['u', 'i', 'ts', 'label', 'idx']
    :param dataset_name: str, dataset name
    :return:
    """
    u_list, i_list, ts_list, label_list = [], [], [], []
    feat_l = []
    idx_list = []

    with open(dataset_name) as f:
        # skip the first line
        s = next(f)
        previous_time = -1
        for idx, line in enumerate(f):
            e = line.strip().split(',')
            # user_id
            u = int(e[0])
            # item_id
            i = int(e[1])

            # timestamp
            ts = float(e[2])
            # check whether time in ascending order
            assert ts >= previous_time
            previous_time = ts
            # state_label
            if float(e[3]) < 0.5: label = 0
            else : label = 1
            #llabel = float(0 if e[3] < 0.5 else 1)

            # edge features
            feat = [label, 0]#np.array([float(x) for x in e[3:]])

            u_list.append(u)
            i_list.append(i)
            ts_list.append(ts)
            label_list.append(label)
            # edge index
            idx_list.append(idx)

            feat_l.append(feat)
    print("总共做对多少题目",sum(label_list),sum(label_list)/len(label_list)*100 ,"%")
    return pd.DataFrame({'u': u_list,
                         'i': i_list,
                         'ts': ts_list,
                         'label': label_list,
                         'idx': idx_list}), np.array(feat_l)

def preprocessKTDiff(dataset_name: str,PATH:str):
    """
    read the original data file and return the DataFrame that has columns ['u', 'i', 'ts', 'label', 'idx']
    :param dataset_name: str, dataset name
    :return:
    """
    u_list, i_list, ts_list, label_list = [], [], [], []
    feat_l = []
    reindex_time = False
    if dataset_name in[ "assisst12","assisst12x"]:
        cols = ['user_id','problem_id','start_time','correct', 'skill']
        reindex_time = True
        f = pd.read_csv(PATH,usecols=cols, index_col=None).fillna("0")
        
    if dataset_name in['assist17']:
        # f = pd.read_csv(PATH,index_col=None,sep='\t').fillna("0")
        # print(f.head(1))
        cols = ['user_id','problem_id','timestamp','correct','skill_id']
        f = pd.read_csv(PATH,usecols=cols, index_col=None,sep='\t').fillna("0")

    if dataset_name in['Slepemapy']:
        cols = ['user','place_asked','inserted','place_answered']
        f = pd.read_csv(PATH,usecols=cols, index_col=None,sep=';').fillna("0")
        f[cols[-1]] = f.apply(lambda row: 1 if row[cols[1]] == row[cols[-1]] else 0, axis=1)
        cols.append('skill')
        f[cols[-1]] = f[cols[1]].copy()
        reindex_time = True

    print(f.shape)
    # f = f.filter(lambda x: len(x) > threshold)
    # 使用 .filter 和 lambda 表达式来筛选数据
    f = f[f[cols[3]].apply(lambda x: x in [0, 1])]
    # new_f = pd.DataFrame(columns = cols)
    threshold = 5
    f = f.groupby(cols[1]).filter(lambda x: len(x) > threshold)
    
    f = f.groupby(cols[0]).filter(lambda x: len(x) > threshold)

    f = f.groupby(cols[1]).filter(lambda x: len(x) > threshold)
    print("m1",f.shape) 

    f = f.groupby(cols[0]).filter(lambda x: len(x) > threshold)
    print("m2",f.shape)

    u_list = reindexColumn(f[cols[0]])
    i_list = reindexColumn(f[cols[1]])
    feat_l = reindexColumn(f[cols[4]])

    u_len = len(set(u_list)) + 1
    i_len = len(set(i_list))
    skill_len = len(set(feat_l))+1

    i_diff = np.zeros(u_len + i_len)
    skill_diff = np.zeros(skill_len)

    i = 0
    progress_bar = tqdm(f.groupby(cols[1]))
    for _, item_df in progress_bar:
        diff = sum(item_df[cols[3]])/len(item_df[cols[3]])
        if diff < 0.01 or len(item_df[cols[3]]) < 30:
            diff = 1
        else:
            diff = int(diff * 100)
        i_diff[u_len + i] = diff
        i+=1   

    i = 1    
    progress_bar = tqdm(f.groupby(cols[4]))
    for _, item_df in progress_bar:
        diff = sum(item_df[cols[3]])/len(item_df[cols[3]])
        if diff < 0.01 or len(item_df[cols[3]]) < 30:
            diff = 1
        else:
            diff = int(diff * 100)
        skill_diff[i] = diff
        i+=1

    OUT_DIFF_FEAT = '../processed_data/{}/ml_{}_node_diff.npy'.format(dataset_name, dataset_name)
    OUT_FEATURE_DIFF_FEAT = '../processed_data/{}/ml_{}_skill_diff.npy'.format(dataset_name, dataset_name)
    print(skill_diff, "\n",i_diff)
    np.save(OUT_DIFF_FEAT, i_diff)
    np.save(OUT_FEATURE_DIFF_FEAT, skill_diff)


def preprocessEdnetKT1():
    problem_df = pd.read_csv('/home/plz/my/data/questions.csv',sep=',')
    # 将 question_id 列设置为索引列
    problem_df.set_index('question_id', inplace=True)

    # 将 tag 列按 ';' 分割成列表形式，并将其作为新列添加到 DataFrame 中, 删除 tag 列
    problem_df['tags'] = problem_df['tags'].str.split(';')
    # problem_df.drop('tag', axis=1, inplace=True)
    # 将 DataFrame 转换为字典
    problem_dict = problem_df.to_dict(orient='index') 

    # 遍历所有 u+数字+.csv 文件
    merged_df = pd.DataFrame()
    i = 0 
    pdlist =  [[] for _ in range(100)]

    for filename in tqdm(glob.glob('/home/plz/my/data/KT1/u*.csv')):
        # 读取文件为 pandas DataFrame
        # if i > 5:
        #     break
        i += 1
        df = pd.read_csv(filename, usecols=['question_id', 'timestamp', 'user_answer'],index_col=None,sep=',')

        # 将正确答案列添加到 DataFrame 中
        length = len(df)
        number = re.findall(r'\d+', filename)[1]
        ids = [number] * length
        df['u'] = ids
        pdlist[i//10000].append(df)
        # merged_df = pd.concat([merged_df, df])
    for pdx in pdlist:
        merged_df = pd.concat([merged_df, pd.concat(pdx)])
    merged_df = merged_df[merged_df['question_id'].isin(problem_dict)]
    merged_df['correct_answer'] = merged_df['question_id'].map(lambda q_id: problem_dict[q_id]['correct_answer'])
    merged_df['r'] = (merged_df['user_answer'] == merged_df['correct_answer']).astype(int) 
    merged_df.drop(['correct_answer','user_answer'], axis=1, inplace=True)

    merged_df.sort_values(by='timestamp')
    # print(merged_df)
    u_list = reindexColumn(merged_df['u'])
    i_list, feat_skill = reindexColumnSkill(merged_df['question_id'],problem_dict)
    ts_list = merged_df['timestamp']
    label_list = merged_df['r']

    # 填充 numpy 数组
    item_i = u_len = len(set(u_list)) + 1 #因为边要从0开始编码，所以对于第0条要空出来且对u编码的时候要往后面挪一格
    i_len = len(set(i_list))
    feat_n = np.zeros((u_len + i_len, 20), dtype=int)

    for tags in tqdm(feat_skill.values()):
        feat_n[item_i, :len(tags)] = tags
        item_i += 1

    
    p = pd.DataFrame({'u': u_list,
                         'i': i_list,
                         'ts': ts_list,
                         'label': label_list,
                         'idx': range(len(u_list))})
    
    print(sum(label_list)/len(label_list))

    return p, label_list[:,np.newaxis],feat_n

    
def preprocessKT(dataset_name: str,PATH:str):
    """
    read the original data file and return the DataFrame that has columns ['u', 'i', 'ts', 'label', 'idx']
    :param dataset_name: str, dataset name
    :return:
    """
    u_list, i_list, ts_list, label_list = [], [], [], []
    feat_l = []
    reindex_time = False
    if dataset_name == 'EdnetKT1':
        return preprocessEdnetKT1()
    if dataset_name in[ "assisst12","assisst12x"]:
        cols = ['user_id','problem_id','start_time','correct', 'skill']
        reindex_time = True
        f = pd.read_csv(PATH,usecols=cols, index_col=None).fillna("0")
        
    if dataset_name in['assist17']:
        cols = ['user','problem_id','timestamp','correct','skill_id']
        f = pd.read_csv(PATH,usecols=cols, index_col=None,sep='\t').fillna("0")

    if dataset_name in['Slepemapy']:
        cols = ['user','place_asked','inserted','place_answered']
        f = pd.read_csv(PATH,usecols=cols, index_col=None,sep=';').fillna("0")
        f[cols[-1]] = f.apply(lambda row: 1 if row[cols[1]] == row[cols[-1]] else 0, axis=1)
        cols.append('skill')
        f[cols[-1]] = f[cols[1]].copy()
        reindex_time = True

    print(f.shape)
    # f = f.filter(lambda x: len(x) > threshold)
    # 使用 .filter 和 lambda 表达式来筛选数据
    f = f[f[cols[3]].apply(lambda x: x in [0, 1])]
    # new_f = pd.DataFrame(columns = cols)
    threshold = 5
    f = f.groupby(cols[1]).filter(lambda x: len(x) > threshold)
    
    f = f.groupby(cols[0]).filter(lambda x: len(x) > threshold)

    f = f.groupby(cols[1]).filter(lambda x: len(x) > threshold)
    print("m1",f.shape) 

    f = f.groupby(cols[0]).filter(lambda x: len(x) > threshold)
    print("m2",f.shape)
    


    if reindex_time: f[cols[2]] = reindexTime(f[cols[2]])
    f = (f.reindex(columns=cols)).sort_values(by=cols[2])
    f[cols[0]] = u_list = reindexColumn(f[cols[0]])
    f[cols[1]] = i_list = reindexColumn(f[cols[1]])
    ts_list = f[cols[2]]
    f[cols[3]] = label_list = f[cols[3]]#.apply(lambda x: 0 if x < 0.5 else 1)
    f[cols[4]] = reindexColumn(f[cols[4]])

    # print(f.head(2))
    
    item_i = u_len = len(set(u_list)) + 1 #因为边要从0开始编码，所以对于第0条要空出来且对u编码的时候要往后面挪一格
    i_len = len(set(i_list))
    feat_n = np.zeros(u_len + i_len)
    
    skills = []
    progress_bar = tqdm(f.groupby(cols[1]))
    for _, item_df in progress_bar:
        skill = float(item_df[cols[4]].iloc[0])
        if skill not in skills:
            skills.append(skill)
        feat_n[item_i] = skill
        item_i += 1

    print("u_len,i_len,skill_len",u_len,i_len,len(skills))
    return pd.DataFrame({'u': u_list,
                         'i': i_list,
                         'ts': ts_list,
                         'label': label_list,
                         'idx': range(len(u_list))}), label_list[:,np.newaxis],feat_n[:,np.newaxis]# np.array(label_list) 而不是 feat_l 了！



def reindex(df: pd.DataFrame, bipartite: bool = True):
    """
    reindex the ids of nodes and edges
    :param df: DataFrame
    :param bipartite: boolean, whether the graph is bipartite or not
    :return:
    """
    new_df = df.copy()
    if bipartite:
        # check the ids of users and items
        assert (df.u.max() - df.u.min() + 1 == len(df.u.unique()))
        assert (df.i.max() - df.i.min() + 1 == len(df.i.unique()))
        assert df.u.min() == df.i.min() == 0

        # if bipartite, discriminate the source and target node by unique ids (target node id is counted based on source node id)
        upper_u = df.u.max() + 1
        new_i = df.i + upper_u

        new_df.i = new_i

    # make the id start from 1
    new_df.u += 1
    new_df.i += 1
    new_df.idx += 1

    return new_df


def preprocess_data(dataset_name: str, bipartite: bool = True, node_feat_dim: int = 32):
    """
    preprocess the data
    :param dataset_name: str, dataset name
    :param bipartite: boolean, whether the graph is bipartite or not
    :param node_feat_dim: int, dimension of node features
    :return:
    """
    Path("../processed_data/{}/".format(dataset_name)).mkdir(parents=True, exist_ok=True)
    PATH = '../DG_data/{}/{}.csv'.format(dataset_name, dataset_name)
    OUT_DF = '../processed_data/{}/ml_{}.csv'.format(dataset_name, dataset_name)
    OUT_FEAT = '../processed_data/{}/ml_{}.npy'.format(dataset_name, dataset_name)
    OUT_NODE_FEAT = '../processed_data/{}/ml_{}_node.npy'.format(dataset_name, dataset_name)
    OUT_USER_DF = '../processed_data/{}/KT_{}.csv'.format(dataset_name, dataset_name)
    OUT_INFO = '../processed_data/{}/KT_info_{}.csv'.format(dataset_name, dataset_name)

    if dataset_name in ['assist17','assisst12', 'assisst12x','Slepemapy','EdnetKT1']:
        # preprocessKTDiff(dataset_name, PATH)
        # return 
        df, edge_feats,node_feats = preprocessKT(dataset_name, PATH)
        new_df = reindex(df, bipartite)

    else:
        df, edge_feats = preprocess(PATH)
        new_df = reindex(df, bipartite)
        max_idx = max(new_df.u.max(), new_df.i.max())
        node_feats = np.zeros((max_idx + 1, node_feat_dim)) #因为是从1开始对节点编号所以第0个节点就是0000，其实不存在

    print("new_df",new_df.shape,edge_feats.shape)

    # edge feature for zero index, which is not used (since edge id starts from 1)
    empty = np.zeros(edge_feats.shape[1])[np.newaxis, :]
    # Stack arrays in sequence vertically(row wise),
    edge_feats = np.vstack([empty, edge_feats])

    # node features with one additional feature for zero index (since node id starts from 1)
    

    print('number of nodes ', node_feats.shape[0] - 1)
    print('number of node features ', node_feats.shape[1])
    print('number of edges ', edge_feats.shape[0] - 1)
    print('number of edge features ', edge_feats.shape[1])

    new_df.to_csv(OUT_DF)  # edge-list
    np.save(OUT_FEAT, edge_feats)  # edge features
    np.save(OUT_NODE_FEAT, node_feats)  # node features


def check_data(dataset_name: str):
    """
    check whether the processed datasets are identical to the given processed datasets
    :param dataset_name: str, dataset name
    :return:
    """
    # original data paths
    origin_OUT_DF = '../DG_data/{}/ml_{}.csv'.format(dataset_name, dataset_name)
    origin_OUT_FEAT = '../DG_data/{}/ml_{}.npy'.format(dataset_name, dataset_name)
    origin_OUT_NODE_FEAT = '../DG_data/{}/ml_{}_node.npy'.format(dataset_name, dataset_name)

    # processed data paths
    OUT_DF = '../processed_data/{}/ml_{}.csv'.format(dataset_name, dataset_name)
    OUT_FEAT = '../processed_data/{}/ml_{}.npy'.format(dataset_name, dataset_name)
    OUT_NODE_FEAT = '../processed_data/{}/ml_{}_node.npy'.format(dataset_name, dataset_name)

    # Load original data
    origin_g_df = pd.read_csv(origin_OUT_DF)
    origin_e_feat = np.load(origin_OUT_FEAT)
    origin_n_feat = np.load(origin_OUT_NODE_FEAT)

    # Load processed data
    g_df = pd.read_csv(OUT_DF)
    e_feat = np.load(OUT_FEAT)
    n_feat = np.load(OUT_NODE_FEAT)

    assert_frame_equal(origin_g_df, g_df)
    # check numbers of edges and edge features
    assert origin_e_feat.shape == e_feat.shape and origin_e_feat.max() == e_feat.max() and origin_e_feat.min() == e_feat.min()
    # check numbers of nodes and node features
    assert origin_n_feat.shape == n_feat.shape and origin_n_feat.max() == n_feat.max() and origin_n_feat.min() == n_feat.min()


parser = argparse.ArgumentParser('Interface for preprocessing datasets')
parser.add_argument('--dataset_name', type=str,
                    choices=['EdnetKT1','Slepemapy','assist17','assisst12x','assisst12',
                             'wikipedia', 'reddit', 'mooc', 'lastfm', 'myket', 'enron', 'SocialEvo', 'uci','Flights', 'CanParl', 'USLegis', 'UNtrade', 'UNvote', 'Contacts'
                             ],
                    help='Dataset name', default='wikipedia')
parser.add_argument('--node_feat_dim', type=int, default=32, help='Number of node raw features') ##defalut=172
parser.add_argument('--max_step', type=int, default=50, help='Number of steps of student\' records')

args = parser.parse_args()

print(f'preprocess dataset {args.dataset_name}...')
if args.dataset_name in ['enron', 'SocialEvo', 'uci']:
    Path("../processed_data/{}/".format(args.dataset_name)).mkdir(parents=True, exist_ok=True)
    copy_tree("../DG_data/{}/".format(args.dataset_name), "../processed_data/{}/".format(args.dataset_name))
    print(f'the original dataset of {args.dataset_name} is unavailable, directly use the processed dataset by previous works.')
else:
    # bipartite dataset
    if args.dataset_name in ['wikipedia', 'reddit', 'mooc', 'lastfm', 'myket','assisst12x','assisst12','assist17','Slepemapy','EdnetKT1']:
        print("bip")
        preprocess_data(dataset_name=args.dataset_name, bipartite=True, node_feat_dim=args.node_feat_dim)
    else:
        preprocess_data(dataset_name=args.dataset_name, bipartite=False, node_feat_dim=args.node_feat_dim)
    print(f'{args.dataset_name} is processed successfully.')

    if args.dataset_name not in ['myket']:
        check_data(args.dataset_name)
    print(f'{args.dataset_name} passes the checks successfully.')
