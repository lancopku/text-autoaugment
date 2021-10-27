import os
import transformers
import pkg_resources
from .data import augment
from .search import search_policy
from datasets import load_dataset
from theconf import Config as C
from .text_networks import get_model, num_class, get_num_class
from transformers import BertForSequenceClassification, Trainer, TrainingArguments, BertTokenizerFast
import joblib
from .archive import policy_map


def search_and_augment(dataset, abspath, n_aug):
    """search optimal policy and then use it to augment texts"""
    resource_package = __name__
    resource_path = '/'.join(('confs','bert_huggingface.yaml'))
    configfile = pkg_resources.resource_filename(resource_package, resource_path)

    _ = C(configfile)

    search_policy(dataset=dataset, abspath=abspath)

    train_dataset = load_dataset('glue', dataset, split='train')

    policy_path = os.path.join(abspath,'final_policy/%s_Bert_seed59_train-npc50_n-aug%d_ir1.00_taa.pkl' % (dataset, n_aug))
    policy = joblib.load(policy_path)
    augmented_train_dataset = augment(dataset=train_dataset, policy=policy, n_aug=n_aug)
    
    return augmented_train_dataset


def augment_with_presearched_policy(dataset, abspath, n_aug):
    """use pre-searched policy to augment texts"""
    resource_package = __name__
    resource_path = '/'.join(('confs', 'bert_huggingface.yaml'))
    configfile = pkg_resources.resource_filename(resource_package, resource_path)

    _ = C(configfile)

    train_dataset = load_dataset('glue', dataset, split='train')

    assert dataset in list(policy_map.keys())
    policy = policy_map[dataset]
    augmented_train_dataset = augment(dataset=train_dataset, policy=policy, n_aug=n_aug)

    return augmented_train_dataset
