from .imdb import IMDB
from .sst5 import SST5
from .sst2 import SST2
from .trec import TREC
from .yelp2 import YELP2
from .yelp5 import YELP5

__all__ = ('IMDB', 'SST2', 'SST5', 'TREC', 'YELP2', 'YELP5')


def get_dataset(dataset_name, examples, tokenizer, text_transform=None):
    dataset_name = dataset_name.lower()
    datasets = {
        'imdb': IMDB,
        'sst2': SST2,
        'sst5': SST5,
        'trec': TREC,
        'yelp2': YELP2,
        'yelp5': YELP5,
    }
    dataset = datasets[dataset_name](examples, tokenizer, text_transform)
    return dataset
