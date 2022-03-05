import pkg_resources
from .data import augment
from .search import search_policy
from datasets import load_dataset
from theconf import Config as C
from .archive import policy_map
from .utils.train_tfidf import train_tfidf


def search_and_augment(configfile=None):
    """search optimal policy and then use it to augment texts"""
    if configfile is None:
        resource_package = __name__
        resource_path = '/'.join(('confs', 'bert_sst2_example.yaml'))
        configfile = pkg_resources.resource_filename(resource_package, resource_path)

    _ = C(configfile)

    name = C.get()['dataset']['name']
    path = C.get()['dataset']['path']
    data_dir = C.get()['dataset']['data_dir']
    data_files = C.get()['dataset']['data_files']
    abspath = C.get()['abspath']

    policy = search_policy(dataset=name, abspath=abspath)

    if path is None:
        train_dataset = load_dataset(path=name, data_dir=data_dir, data_files=data_files, split='train')
    else:
        train_dataset = load_dataset(path=path, name=name, data_dir=data_dir, data_files=data_files, split='train')

    n_aug = C.get()['n_aug']
    augmented_train_dataset = augment(dataset=train_dataset, policy=policy, n_aug=n_aug)

    return augmented_train_dataset


def augment_with_presearched_policy(configfile=None):
    """use pre-searched policy to augment texts"""
    if configfile is None:
        resource_package = __name__
        resource_path = '/'.join(('confs', 'bert_imdb_example.yaml'))
        configfile = pkg_resources.resource_filename(resource_package, resource_path)

    _ = C(configfile)

    name = C.get()['dataset']['name']
    path = C.get()['dataset']['path']
    data_dir = C.get()['dataset']['data_dir']
    data_files = C.get()['dataset']['data_files']

    if path is None:
        train_dataset = load_dataset(path=name, data_dir=data_dir, data_files=data_files, split='train')
    else:
        train_dataset = load_dataset(path=path, name=name, data_dir=data_dir, data_files=data_files, split='train')

    train_tfidf(name)  # calculate tf-idf score for TS and TI operations

    assert name in list(policy_map.keys()), "No policy was found for this dataset."
    policy = policy_map[name]
    n_aug = C.get()['n_aug']
    augmented_train_dataset = augment(dataset=train_dataset, policy=policy, n_aug=n_aug)

    return augmented_train_dataset
