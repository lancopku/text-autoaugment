import sys
import logging
import os
import random
import torch
from torch.utils.data import SubsetRandomSampler, Sampler, Subset, ConcatDataset
import transforms as text_transforms
from sklearn.model_selection import StratifiedShuffleSplit, KFold
from theconf import Config as C
import numpy as np
from datasets import get_dataset
from augmentation import get_augment, apply_augment, random_augment
from common import get_logger
import pandas as pd
from utils.raw_data_utils import get_processor, subsample_by_classes
from transformers import BertTokenizer, BertTokenizerFast
from text_networks import num_class
import math
import copy
from archive import policy_map
import multiprocessing
from functools import partial
import time
from utils.get_data import download_data
from utils.metrics import n_dist


logger = get_logger('Text AutoAugment')
logger.setLevel(logging.INFO)


def get_datasets(dataset, dataroot, policy_opt):
    # do data augmentation
    transform_train = text_transforms.Compose([])
    aug = C.get()['aug']
    logger.info('aug: {}'.format(aug))

    if isinstance(aug, list) or aug in ['taa', 'random_taa']:  # using sampled policies
        transform_train.transforms.insert(0, Augmentation(aug))
    elif aug in list(policy_map.keys()):  # use pre-searched policy
        transform_train.transforms.insert(0, Augmentation(policy_map[aug]))

    # load dataset
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    download_data(dataset)
    processor = get_processor(dataset)

    class_num = num_class(C.get()['dataset'])
    all_train_examples = processor.get_train_examples(dataroot)
    if C.get()['train']['npc'] == C.get()['valid']['npc'] == C.get()['test']['npc'] == -1:
        assert policy_opt is False
        all_train_examples_num = len(all_train_examples)
        train_examples, valid_examples = processor.split(all_train_examples, int(0.2 * all_train_examples_num),
                                                         int(0.8 * all_train_examples_num))
    else:
        train_examples, _ = processor.split(all_train_examples, class_num * C.get()['valid']['npc'],
                                            class_num * C.get()['train']['npc'])
        train_examples = subsample_by_classes(train_examples, processor.get_labels(), 'train')
        all_train_examples = list(set(all_train_examples) - set(train_examples))
        valid_examples, test_examples = processor.split(all_train_examples, class_num * C.get()['test']['npc'],
                                                        class_num * C.get()['valid']['npc'])

    if not policy_opt:
        test_examples = processor.get_test_examples(dataroot)

    train_dataset = get_dataset(dataset, train_examples, tokenizer, text_transform=transform_train)
    val_dataset = get_dataset(dataset, valid_examples, tokenizer, text_transform=None)
    test_dataset = get_dataset(dataset, test_examples, tokenizer, text_transform=None)
    logger.info('len(trainset) = %d; len(validset) = %d; len(testset) = %d' %
                (len(train_dataset), len(val_dataset), len(test_dataset)))
    return train_dataset, val_dataset, test_dataset


class Augmentation(object):
    def __init__(self, policy):
        self.policy = policy

    def __call__(self, texts, labels):
        texts = np.array(texts)
        labels = np.array(labels)
        if C.get()['ir'] < 1 and C.get()['method'] != 'bt':
            # rebalanced data
            ir_index = np.where(labels == 0)
            texts = np.append(texts, texts[ir_index].repeat(int(1 / C.get()['ir']) - 1))
            labels = np.append(labels, labels[ir_index].repeat(int(1 / C.get()['ir']) - 1))
        # generate multiple augmented data if necessary
        labels = labels.repeat(C.get()['n']['aug'])
        if C.get()['method'] == 'taa' or C.get()['method'] == 'random_taa':
            texts = texts.repeat(C.get()['n']['aug'])
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
