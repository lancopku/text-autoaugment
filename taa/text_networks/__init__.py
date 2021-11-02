import sys
import torch
import os
from torch import nn
from torch.nn import DataParallel
import torch.backends.cudnn as cudnn
from transformers import BertTokenizer, BertForSequenceClassification
from datasets import load_dataset
from theconf import Config as C


def get_model(conf, num_class=10, data_parallel=True):
    name = conf['model']['type']

    if 'BertForSequenceClassification' in name:
        model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=num_class)
    else:
        raise NameError('no model named, %s' % name)
    
    if torch.cuda.is_available():
        if data_parallel:
            model = model.cuda()
            # model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
            model = DataParallel(model)
        else:
            import horovod.torch as hvd
            device = torch.device('cuda', hvd.local_rank())
            model = model.to(device)
        cudnn.benchmark = True
    return model


def num_class(dataset):
    return {
        'wiki_qa': 2,
        'toy': 2,
        'imdb': 2,
        'sts': 1,
        'mrpc': 2,
        'sst2': 2,
        'sst5': 5,
        'trec': 6,
        'yelp2': 2,
        'yelp5': 5,
        'amazon2': 2
    }[dataset]


def get_num_class(dataset):
    path = C.get()['dataset']['path']
    if path is None:
        dataset = load_dataset(path=dataset, split='train')
    else:
        dataset = load_dataset(path=path, name=dataset, split='train')
    return dataset.features['label'].num_classes

