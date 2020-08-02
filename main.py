import argparse
from asteroid.losses.sdr import SingleSrcNegSDR
loss_func = SingleSrcNegSDR(
    sdr_type="sisdr", zero_mean=False, reduction='mean')

parser = argparse.ArgumentParser(
    description='Argument Parser for SERIL project.')
parser.add_argument('--name', required=True,
                    help='Name of current experiment.')
parser.add_argument('--noisy_trainset', nargs='+', required=True,
                    help='Noisy utterances for training in different environments.')
parser.add_argument('--clean_trainset', nargs='+', required=True,
                    help='Clean utterances correspond to the training noisy utterances.')
parser.add_argument('--noisy_validset', nargs='+', required=True,
                    help='Noisy utterances for testing in different environments.')
parser.add_argument('--clean_validset', nargs='+', required=True,
                    help='Clean utterances correspond to the testing noisy utterances.')
parser.add_argument('--n_jobs', default=1, type=int)

# Options
parser.add_argument('--config', default='config/downstream.yaml',
                    help='Path to downstream experiment config.', required=False)
parser.add_argument('--expdir', default='result',
                    help='Path to store experiment result, if empty then default is used.', required=False)
parser.add_argument('--seed', default=1337, type=int,
                    help='Random seed for reproducable results.', required=False)
parser.add_argument('--gpu', default='0', action='store_true',
                    help='Assigning GPU id.')

# parse
args = parser.parse_args()
config = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)

# return args, config
