import sys
import nlpaug.augmenter.word as naw
from theconf import Config as C
import os
import ray
import torch
import random
import numpy as np


def random_word_swap(text, m):
    '''
    Randomly swap adjacent words
    :param m: magnitude of operation, here means the proportion of modified words
    :return:
    '''
    max_seq_length = C.get()['max_seq_length']
    aug = naw.RandomWordAug(action='swap', aug_max=max_seq_length, aug_p=m)
    return aug.augment(text)


def random_word_delete(text, m):
    max_seq_length = C.get()['max_seq_length']
    aug = naw.RandomWordAug(action='delete', aug_max=max_seq_length, aug_p=m)
    return aug.augment(text)


def tfidf_word_insert(text, m):  # default top_k=5
    dataset_type = C.get()['dataset']['name']
    abspath = C.get()['abspath']
    max_seq_length = C.get()['max_seq_length']
    model_path = os.path.join(abspath, 'models/tfidf/%s' % dataset_type)
    aug = naw.TfIdfAug(model_path=model_path, action='insert', aug_max=max_seq_length, aug_p=m, top_k=10)
    return aug.augment(text)


def tfidf_word_substitute(text, m):  # default top_k=5
    dataset_type = C.get()['dataset']['name']
    abspath = C.get()['abspath']
    max_seq_length = C.get()['max_seq_length']
    model_path = os.path.join(abspath, 'models/tfidf/%s' % dataset_type)
    assert os.path.exists(model_path)
    aug = naw.TfIdfAug(model_path=model_path, action='substitute', aug_max=max_seq_length, aug_p=m, top_k=10)
    return aug.augment(text)


def andonym_word_substitute(text, m):
    max_seq_length = C.get()['max_seq_length']
    aug = naw.AntonymAug(aug_max=max_seq_length, aug_p=m)
    return aug.augment(text)


def synonym_word_substitute(text, m):
    max_seq_length = C.get()['max_seq_length']
    aug = naw.SynonymAug(aug_max=max_seq_length, aug_p=m)
    return aug.augment(text)


def augment_list():  # 16 operations and their ranges
    l = [
        (random_word_swap, 0.0, 0.5),  # 0
        (random_word_delete, 0.0, 0.5),  # 1
        (tfidf_word_insert, 0.0, 0.5),  # 2
        (tfidf_word_substitute, 0.0, 0.5),  # 3
        (synonym_word_substitute, 0.0, 0.5),  # 5
    ]
    return l


augment_dict = {fn.__name__: (fn, v1, v2) for fn, v1, v2 in augment_list()}


def get_augment(name):
    return augment_dict[name]


# @ray.remote
# def apply_augment(text, sub_policy, config):
def apply_augment(text, policy, config):
    sub_policy = random.choice(policy)
    C.get()
    C.get().conf = config
    for name, pr, level in sub_policy:
        if random.random() > pr:
            continue
        augment_fn, low, high = get_augment(name)
        text = augment_fn(text, level * (high - low) + low)
    return text


# @ray.remote(num_cpus=1)
def random_augment(text, config):
    C.get()
    C.get().conf = config
    for _ in range(C.get()['num']['op']):
        pr = random.random()
        if random.random() > pr:
            continue
        augment_fn, low, high = random.choice(augment_list())
        level = random.random()
        text = augment_fn(text, level * (high - low) + low)  # TODO
    return text

