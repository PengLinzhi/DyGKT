import argparse
import sys
import torch


def get_link_classification_args(is_evaluation: bool = False):
    """
    get the args for the link prediction task
    :param is_evaluation: boolean, whether in the evaluation process
    :return:
    """
    # arguments
    parser = argparse.ArgumentParser('Interface for the link prediction task')
    parser.add_argument('--test', action='store_true', default=False)
    parser.add_argument('--dataset_name', type=str, help='dataset to be used', default='assist17',
                        choices=['junyi','EdnetKT1','Slepemapy', 'assist17','assist12'])
    parser.add_argument('--batch_size', type=int, default=2000, help='batch size')
    parser.add_argument('--model_name', type=str, default='DyGKT', help='name of the model',
                        choices=['DyGKT','QIKT','IEKT','IPKT','DIMKT','simpleKT','AKTMemory','DKTMemory','CTNCM','AKT','DKT', 'TGAT', 'TGN', 'DyGFormer'])
    parser.add_argument('--gpu', type=int, default=0, help='number of gpu to use')

    parser.add_argument('--ablation', type=str, default='-1', choices=['-1','q_qid','q_kid','counter', 'dual', 'embed','skill','time'], help='how to sample historical neighbors')

    parser.add_argument('--predict_loss', type=float, default=0.0, help='loss function to use')

    parser.add_argument('--state_dim', type=int, default=256, help='dimension of the student state embedding')
    
    parser.add_argument('--num_neighbors', type=int, default=50, help='number of neighbors to sample for each node')
    
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout rate')
    parser.add_argument('--num_epochs', type=int, default=100, help='number of epochs')


    parser.add_argument('--sample_neighbor_strategy', type=str, default='recent', choices=['uniform', 'recent', 'time_interval_aware'], help='how to sample historical neighbors')
    parser.add_argument('--time_scaling_factor', default=1e-6, type=float, help='the hyperparameter that controls the sampling preference with time interval, '
                        'a large time_scaling_factor tends to sample more on recent links, 0.0 corresponds to uniform sampling, '
                        'it works when sample_neighbor_strategy == time_interval_aware')
    parser.add_argument('--num_walk_heads', type=int, default=8, help='number of heads used for the attention in walk encoder')
    parser.add_argument('--num_heads', type=int, default=2, help='number of heads used in attention layer')
    parser.add_argument('--num_layers', type=int, default=2, help='number of model layers')
    parser.add_argument('--walk_length', type=int, default=1, help='length of each random walk')
    parser.add_argument('--time_gap', type=int, default=2000, help='time gap for neighbors to compute node features')
    parser.add_argument('--time_feat_dim', type=int, default=16, help='dimension of the time embedding')
    parser.add_argument('--position_feat_dim', type=int, default=172, help='dimension of the position embedding')
    parser.add_argument('--time_window_mode', type=str, default='fixed_proportion', help='how to select the time window size for time window memory',
                        choices=['fixed_proportion', 'repeat_interval'])
    parser.add_argument('--patch_size', type=int, default=1, help='patch size')
    parser.add_argument('--channel_embedding_dim', type=int, default=50, help='dimension of each channel embedding')
    parser.add_argument('--max_input_sequence_length', type=int, default=20, help='maximal length of the input sequence of each node')
    parser.add_argument('--learning_rate', type=float, default=0.0005, help='learning rate')
    
    parser.add_argument('--optimizer', type=str, default='Adam', choices=['SGD', 'Adam', 'RMSprop'], help='name of optimizer')
    parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay')
    parser.add_argument('--patience', type=int, default=10, help='patience for early stopping')
    parser.add_argument('--val_ratio', type=float, default=0.1, help='ratio of validation set')
    parser.add_argument('--test_ratio', type=float, default=0.1, help='ratio of test set')
    parser.add_argument('--num_runs', type=int, default=3, help='number of runs')
    parser.add_argument('--test_interval_epochs', type=int, default=10, help='how many epochs to perform testing once')
    parser.add_argument('--negative_sample_strategy', type=str, default='random', choices=['random', 'historical', 'inductive'],
                        help='strategy for the negative edge sampling')

    try:
        args = parser.parse_args()
        args.device = f'cuda:{args.gpu}' if torch.cuda.is_available() and args.gpu >= 0 else 'cpu'
    except:
        parser.print_help()
        sys.exit()


    return args
