import sys
import logging
import os
import random
import torch
from torch.utils.data import SubsetRandomSampler, Sampler, Subset, ConcatDataset
import .transforms as text_transforms
from sklearn.model_selection import StratifiedShuffleSplit, KFold
from theconf import Config as C
import numpy as np
from .custom_dataset import general_dataset
from datasets import load_dataset 
from .augmentation import get_augment, apply_augment, random_augment
from .common import get_logger
import pandas as pd
from .utils.raw_data_utils import get_processor, general_subsample_by_classes, get_examples, general_split
from transformers import BertTokenizer, BertTokenizerFast
from .text_networks import num_class
import math
import copy
from .archive import policy_map, huggingface_dataset
import multiprocessing
from functools import partial
import time
from .utils.get_data import download_data
from .utils.metrics import n_dist
import joblib


logger = get_logger('Text AutoAugment')
logger.setLevel(logging.INFO)


def get_datasets(dataset, dataroot, policy_opt):
    # do data augmentation
    transform_train = text_transforms.Compose([])
    aug = C.get()['aug']
    data_config = C.get()['data_config']
    logger.info('aug: {}'.format(aug))
    if isinstance(aug, list) or aug in ['taa', 'random_taa']:  # using sampled policies
        transform_train.transforms.insert(0, Augmentation(aug))
    elif aug in list(policy_map.keys()):  # use pre-searched policy
        transform_train.transforms.insert(0, Augmentation(policy_map[aug]))
    else:
        transform_train.transforms.insert(0, Augmentation(huggingface_dataset()))


    # load dataset
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

    train_dataset = load_dataset(data_config, dataset, split='train')
    valid_dataset = load_dataset(data_config, dataset, split='validation')
    test_dataset = load_dataset(data_config, dataset, split='test')
    
    
    label_names = train_dataset.features['label'].names  
    class_num = train_dataset.features['label'].num_classes
    label_mapping = {}
    for i in range(class_num):
        label_mapping[label_names[i]] = i
    all_train_examples = get_examples(train_dataset)

    if C.get()['train']['npc'] == C.get()['valid']['npc'] == C.get()['test']['npc'] == -1:
        assert policy_opt is False
        all_train_examples_num = len(all_train_examples)
        train_examples, valid_examples = general_split(all_train_examples, int(0.2 * all_train_examples_num),
                                                         int(0.8 * all_train_examples_num))
    else:
        train_examples, _ = general_split(all_train_examples, class_num * C.get()['valid']['npc'],
                                            class_num * C.get()['train']['npc'])
        train_examples = general_subsample_by_classes(train_examples, label_names, label_mapping, 'train')
        all_train_examples = list(set(all_train_examples) - set(train_examples))
        valid_examples, test_examples = general_split(all_train_examples, class_num * C.get()['test']['npc'],
                                                        class_num * C.get()['valid']['npc'])

    if not policy_opt:
        test_examples = get_examples(test_dataset)

    train_dataset = general_dataset(train_examples, tokenizer, text_transform=transform_train)
    val_dataset = general_dataset(valid_examples, tokenizer, text_transform=None)
    test_dataset = general_dataset(test_examples, tokenizer, text_transform=None)
    logger.info('len(trainset) = %d; len(validset) = %d; len(testset) = %d' %
                (len(train_dataset), len(val_dataset), len(test_dataset)))
    return train_dataset, val_dataset, test_dataset


class Augmentation(object):
    def __init__(self, policy):
        self.policy = policy

    def __call__(self, texts, labels):
        texts = np.array(texts)
        labels = np.array(labels)
        # generate multiple augmented data if necessary
        labels = labels.repeat(C.get()['n_aug'])
        texts = texts.repeat(C.get()['n_aug'])            
        partial_apply_augment = partial(apply_augment, policy=self.policy, config=copy.deepcopy(C.get().conf))
        with multiprocessing.Pool(processes=8) as pool:
                aug_texts = pool.map(partial_apply_augment, texts)
        n_dist_value = n_dist(aug_texts, 2)  # ngram=2
        labels = [int(i) for i in list(labels)]
        return aug_texts, labels, n_dist_value


class SubsetSampler(Sampler):
    r"""Samples elements from a given list of indices, without replacement.

    Arguments:
        indices (sequence): a sequence of indices
    """

    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return (i for i in self.indices)

    def __len__(self):
        return len(self.indices)




def augment(dataset, policy_path, n_aug, configfile=None):
    # get searched augmentation policy
#   _ = C('confs/%s' % configfile)
    C.get()['n_aug'] = n_aug

    policy = joblib.load(policy_path)
    transform_train = text_transforms.Compose([])
    transform_train.transforms.insert(0, Augmentation(policy))    

    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

    examples = get_examples(dataset)
    augmented_dataset = general_dataset(examples, tokenizer, text_transform=transform_train)
    
    return augmented_dataset 





if __name__ == '__main__':
    from theconf import Config as C, ConfigArgumentParser

    parser = ConfigArgumentParser(conflict_handler='resolve')
    parser.add_argument('--dataroot', type=str, default='/data/private/pretrainedmodels',
                        help='torchvision data folder')
    parser.add_argument('--until', type=int, default=5)
    parser.add_argument('--num-op', type=int, default=2)
    parser.add_argument('--num-policy', type=int, default=5)
    parser.add_argument('--num-search', type=int, default=200)
    parser.add_argument('--cv-ratio', type=float, default=0.4)
    parser.add_argument('--redis', type=str, default='9.134.78.50:6379')
    parser.add_argument('--per-class', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--smoke-test', action='store_true')
    args = parser.parse_args()
