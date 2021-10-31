import pkg_resources
from .data import augment
from .search import search_policy
from datasets import load_dataset
from theconf import Config as C
from .archive import policy_map


def search_and_augment(dataset, abspath, n_aug, configfile=None):
    """search optimal policy and then use it to augment texts"""
    if configfile is None:
        resource_package = __name__
        resource_path = '/'.join(('confs', 'bert_huggingface.yaml'))
        configfile = pkg_resources.resource_filename(resource_package, resource_path)

    _ = C(configfile)

    policy = search_policy(dataset=dataset, abspath=abspath)

    train_dataset = load_dataset('glue', dataset, split='train')

    augmented_train_dataset = augment(dataset=train_dataset, policy=policy, n_aug=n_aug)

    return augmented_train_dataset


def augment_with_presearched_policy(dataset, n_aug, configfile=None):
    """use pre-searched policy to augment texts"""
    if configfile is None:
        resource_package = __name__
        resource_path = '/'.join(('confs', 'bert_huggingface.yaml'))
        configfile = pkg_resources.resource_filename(resource_package, resource_path)

    _ = C(configfile)

    train_dataset = load_dataset('glue', dataset, split='train')

    assert dataset in list(policy_map.keys())
    policy = policy_map[dataset]
    augmented_train_dataset = augment(dataset=train_dataset, policy=policy, n_aug=n_aug)

    return augmented_train_dataset
