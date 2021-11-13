import sys
import json
from theconf import Config as C, ConfigArgumentParser
import os
import copy
import torch
import shutil
from pystopwatch2 import PyStopwatch
from taa.common import get_logger, add_filehandler
from taa.search import get_path, search_policy, train_model_parallel
import logging

w = PyStopwatch()
logging.basicConfig(level=logging.INFO)
logger = get_logger('Text AutoAugment')

parser = ConfigArgumentParser(conflict_handler='resolve')
parser.add_argument('--until', type=int, default=5)
parser.add_argument('--num-op', type=int, default=2)
parser.add_argument('--num-policy', type=int, default=4)
parser.add_argument('--num-search', type=int, default=200)
parser.add_argument('--redis', type=str, default='localhost:6379')
parser.add_argument('--smoke-test', action='store_true')  # default to False
parser.add_argument('--abspath', type=str, default='/home/renshuhuai/text-autoaugment')
parser.add_argument('--n-aug', type=int, default=16,
                    help='magnification of augmentation. synthesize n-aug for each given sample')
parser.add_argument('--train-npc', type=int, default=40, help='train example num per class')
parser.add_argument('--valid-npc', type=int, default=30, help='valid example num per class')
parser.add_argument('--test-npc', type=int, default=30, help='test example num per class when policy opt')
parser.add_argument('--ir', type=float, default=1, help='imbalance rate')
parser.add_argument('--topN', type=int, default=3, help='top N')
parser.add_argument('--trail', type=int, default=1, help='trail')
parser.add_argument('--seed', type=int, default=59, help='random seed')
parser.add_argument('--method', type=str, default='taa', help='augmentation method', choices=['taa', 'random_taa'])
parser.add_argument('--magnitude', type=float, default=0.8)

args = parser.parse_args()

dataset_type = C.get()['dataset']['name']
model_type = C.get()['model']['type']
abspath = args.abspath

add_filehandler(logger, os.path.join('taa/models', '%s_%s_op%d_policy%d_n-aug%d_ir%.2f_%s.log' %
                                     (dataset_type, model_type, args.num_op, args.num_policy, args.n_aug,
                                      args.ir, C.get()['method'])))
logger.info('Configuration:')
logger.info(json.dumps(C.get().conf, sort_keys=True, indent=4))

copied_c = copy.deepcopy(C.get().conf)

# Train without Augmentations
logger.info('-' * 59)
logger.info('----- Train without Augmentations seed=%3d train-npc=%d -----' % (args.seed, args.train_npc))
logger.info('-' * 59)
torch.cuda.empty_cache()
w.start(tag='train_no_aug')
model_path = get_path(dataset_type, model_type, 'pretrained_trail%d_train-npc%d_seed%d' %
                       (args.trail, args.train_npc, args.seed))
logger.info('Model path: {}'.format(model_path))

pretrain_results = train_model_parallel('pretrained_trail%d_train-npc%d_n-aug%d' %
                                        (args.trail, args.train_npc, args.n_aug), copy.deepcopy(copied_c),
                                        augment=None, save_path=model_path,
                                        only_eval=True if os.path.isfile(model_path) else False)
logger.info('Getting results:')
for train_mode in ['pretrained']:
    avg = 0.
    r_model, r_dict = pretrain_results
    log = ' '.join(['%s=%.4f;' % (key, r_dict[key]) for key in r_dict.keys()])
    logger.info('[%s] ' % train_mode + log)
w.pause('train_no_aug')
logger.info('Processed in %.4f secs' % w.get_elapsed('train_no_aug'))
shutil.rmtree(model_path)
if args.until == 1:
    sys.exit(0)

# Search Test-Time Augmentation Policies
logger.info('-' * 51)
logger.info('----- Search Test-Time Augmentation Policies -----')
logger.info('-' * 51)

w.start(tag='search')
final_policy = search_policy(dataset_type, abspath, num_search=args.num_search, num_policy=args.num_policy,
                             num_op=args.num_op)
w.pause(tag='search')

# Train with Augmentations
logger.info('-' * 94)
logger.info('----- Train with Augmentations model=%s dataset=%s aug=%s -----' %
            (model_type, dataset_type, C.get()['aug']))
logger.info('-' * 94)
torch.cuda.empty_cache()
w.start(tag='train_aug')

augment_path = get_path(dataset_type, model_type, 'augment_trail%d_train-npc%d_n-aug%d_%s_seed%d_ir%.2f' %
                         (args.trail, args.train_npc, args.n_aug, args.method, args.seed, args.ir))
logger.info('Getting results:')
final_results = train_model_parallel('augment_trail%d_train-npc%d_n-aug%d' %
                                     (args.trail, args.train_npc, args.n_aug), copy.deepcopy(copied_c),
                                     augment=final_policy, save_path=augment_path, only_eval=False)

for train_mode in ['augment']:
    avg = 0.
    r_model, r_dict = final_results
    log = ' '.join(['%s=%.4f;' % (key, r_dict[key]) for key in r_dict.keys()])
    logger.info('[%s] ' % train_mode + log)
w.pause('train_aug')
logger.info('Processed in %.4f secs' % w.get_elapsed('train_aug'))

logger.info(w)
shutil.rmtree(augment_path)
