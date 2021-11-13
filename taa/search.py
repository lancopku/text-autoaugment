import sys
import warnings
import os
import copy
import time
from collections import OrderedDict, defaultdict
import torch
import ray
import gorilla
from ray.tune.trial import Trial
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.suggest.hyperopt import HyperOptSearch
from ray.tune import register_trainable, run_experiments
from ray.tune.suggest import ConcurrencyLimiter
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray import tune
from ray import tune
from tqdm import tqdm
from datetime import datetime
from .archive import policy_decoder, remove_deplicates
from .augmentation import augment_list
from .common import get_logger, add_filehandler
from .utils.train_tfidf import train_tfidf
from .train import train_and_eval
from theconf import Config as C, ConfigArgumentParser
import joblib
import random
import logging
import json

logging.basicConfig(level=logging.INFO)


def step_w_log(self):
    original = gorilla.get_original_attribute(ray.tune.trial_runner.TrialRunner, 'step')

    # log
    cnts = OrderedDict()
    for status in [Trial.RUNNING, Trial.TERMINATED, Trial.PENDING, Trial.PAUSED, Trial.ERROR]:
        cnt = len(list(filter(lambda x: x.status == status, self._trials)))
        cnts[status] = cnt
    best_opt_object = 0.
    for trial in filter(lambda x: x.status == Trial.TERMINATED, self._trials):
        if not trial.last_result:
            continue
        best_opt_object = max(best_opt_object, trial.last_result['opt_object'])
    print('Iter', self._iteration, 'opt_object=%.3f' % best_opt_object, cnts, end='\r')
    return original(self)


patch = gorilla.Patch(ray.tune.trial_runner.TrialRunner, 'step', step_w_log, settings=gorilla.Settings(allow_hit=True))
gorilla.apply(patch)

logger = get_logger('Text AutoAugment')


def get_path(dataset, model, tag=None):
    if tag:
        return os.path.join(os.path.dirname(os.path.realpath(__file__)), 'models/%s_%s_%s' % (dataset, model, tag))
    else:
        return os.path.join(os.path.dirname(os.path.realpath(__file__)), 'models/%s_%s' % (dataset, model))


def train_model_parallel(tag, config, augment, save_path=None, only_eval=False):
    C.get()  # useless unless C._instance=None
    C.get().conf = config
    C.get()['aug'] = augment
    result = train_and_eval(tag, policy_opt=False, save_path=save_path, only_eval=only_eval)
    return C.get()['model']['type'], result


def objective(config, checkpoint_dir=None):
    '''evaluate one searched policy'''
    C.get()
    C.get().conf = config

    # setup - provided augmentation rules
    C.get()['aug'] = policy_decoder(config, config['num_policy'], config['num_op'])

    start_t = time.time()

    tag = ''.join([str(i) for i in sum(sum(config['aug'], []), ()) if type(i) is float])[:15]
    save_path = get_path('', '', tag=tag)

    result = train_and_eval(config['tag'], policy_opt=True, save_path=save_path, only_eval=False)

    gpu_secs = (time.time() - start_t) * torch.cuda.device_count()
    tune.report(eval_loss=result['eval_loss'], eval_accuracy=result['eval_accuracy'],
                eval_micro_f1=result['eval_micro_f1'], eval_macro_f1=result['eval_macro_f1'],
                eval_weighted_f1=result['eval_weighted_f1'], n_dist=result['n_dist'], opt_object=result['opt_object'],
                elapsed_time=gpu_secs, done=True)


def search_policy(dataset, abspath, configfile=None, num_search=200, num_policy=4, num_op=2):
    '''search for a customized policy with trained parameters for text augmentation'''
    logger.info('----- Search Test-Time Augmentation Policies -----')
    logger.info('-' * 51)

    logger.info('Loading configuration...')
    if configfile is not None:
        _ = C(configfile)
    C.get()['dataset']['name'] = dataset
    C.get()['num_search'] = num_search
    C.get()['num_policy'] = num_policy
    C.get()['num_op'] = num_op
    C.get()['abspath'] = abspath
    dataset_type = C.get()['dataset']['name']
    model_type = C.get()['model']['type']
    n_aug = C.get()['n_aug']
    num_op = C.get()['num_op']
    num_policy = C.get()['num_policy']
    method = C.get()['method']
    topN = C.get()['topN']
    ir = C.get()['ir']
    seed = C.get()['seed']
    trail = C.get()['trail']
    train_npc = C.get()['train']['npc']
    valid_npc = C.get()['valid']['npc']
    test_npc = C.get()['test']['npc']
    num_search = C.get()['num_search']
    copied_c = copy.deepcopy(C.get().conf)
    num_gpus = C.get()['num_gpus']
    num_cpus = C.get()['num_cpus']

    logger.info('Initialize ray...')
    # ray.init(num_cpus=num_cpus, local_mode=True)  # used for debug
    ray.init(num_gpus=num_gpus, num_cpus=num_cpus)

    train_tfidf(dataset_type)  # calculate tf-idf score for TS and TI operations

    if method == 'taa':
        pkl_path = os.path.join(abspath, 'final_policy')
        if not os.path.exists(pkl_path):
            os.makedirs(pkl_path)
        policy_dir = os.path.join(pkl_path, '%s_%s_seed%d_train-npc%d_n-aug%d_ir%.2f_%s.pkl' %
                                  (dataset_type, model_type, seed, train_npc, n_aug, ir, method))
        total_computation = 0
        if os.path.isfile(policy_dir):  # have been searched
            logger.info('Use existing policy from %s' % policy_dir)
            final_policy = joblib.load(policy_dir)[:topN]
        else:
            ops = augment_list()  # get all possible operations
            logger.info(ops)
            space = {}
            for i in range(num_policy):
                for j in range(num_op):
                    space['policy_%d_%d' % (i, j)] = tune.choice(list(range(0, len(ops))))
                    space['prob_%d_%d' % (i, j)] = tune.uniform(0.0, 1.0)
                    space['level_%d_%d' % (i, j)] = tune.uniform(0.0, 1.0)

            logger.info('Size of search space: %d' % len(space))  # 5*2*3=30
            final_policy = []
            reward_attr = 'opt_object'
            name = "search_%s_%s_seed%d_trail%d" % (dataset_type, model_type, seed, trail)
            logger.info('name: {}'.format(name))
            tune_kwargs = {
                'name': name,
                'num_samples': num_search,
                'resources_per_trial': {'gpu': 1},
                'config': {
                    'tag': 'seed%d_trail%d_train-npc%d_n-aug%d' % (seed, trail, train_npc, n_aug),
                    'num_op': num_op, 'num_policy': num_policy,
                },
                'local_dir': os.path.join(abspath, "ray_results"),
            }
            tune_kwargs['config'].update(space)
            tune_kwargs['config'].update(copied_c)
            algo = HyperOptSearch()
            scheduler = AsyncHyperBandScheduler()
            register_trainable(name, objective)
            analysis = tune.run(objective, search_alg=algo, scheduler=scheduler, metric=reward_attr, mode="max",
                                **tune_kwargs)
            results = [x for x in analysis.results.values()]
            logger.info("num_samples = %d" % (len(results)))
            logger.info("select top = %d" % topN)
            results = sorted(results, key=lambda x: x[reward_attr], reverse=True)

            # calculate computation usage
            for result in results:
                total_computation += result['elapsed_time']

            all_policy = []
            for result in results:  # print policy in #arg.num_search trails
                policy = policy_decoder(result['config'], num_policy, num_op)
                logger.info('opt_object=%.4f; eval_accuracy=%.4f; n_dist=%.4f; %s' %
                            (result['opt_object'], result['eval_accuracy'],
                             result['n_dist'], policy))
                all_policy.extend(policy)

            for result in results[:topN]:  # get top #args.topN in #arg.num_search trails
                policy = policy_decoder(result['config'], num_policy, num_op)
                policy = remove_deplicates(policy)
                final_policy.extend(policy)

            logger.info('Save searched policy to %s' % policy_dir)
            logger.info(json.dumps(final_policy))
            joblib.dump(final_policy, policy_dir)
    elif method == 'random_taa':
        total_computation = 0
        final_policy = []
        for i in range(num_policy):  # 1
            sub_policy = []
            for j in range(num_op):  # 2
                op, _, _ = random.choice(augment_list())
                sub_policy.append((op.__name__, random.random(), random.random()))
            final_policy.append(sub_policy)
    else:
        total_computation = 0
        final_policy = method
    logger.info('Total computation for policy search is %.2f' % total_computation)
    return final_policy
