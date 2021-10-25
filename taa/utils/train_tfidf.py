import re
import argparse
import nlpaug.model.word_stats as nmw
from .raw_data_utils import get_examples
import os
from theconf import Config as C
from datasets import load_dataset 


def _tokenizer(text, token_pattern=r"(?u)\b\w\w+\b"):
    token_pattern = re.compile(token_pattern)
    return token_pattern.findall(text)


def train_tfidf(dataset, data_path=None):
    model_path = 'models/tfidf/{}'.format(dataset)
    data_config = C.get()['data_config']
    if not os.path.exists(model_path):
        print('make model directory')
        os.mkdir(model_path)

        dataset = load_dataset(data_config, dataset, split='train')
        examples = get_examples(dataset)
        texts = [d.text_a if d.text_b is None else d.text_a + ' ' + d.text_b for d in examples]

        # Tokenize input
        train_x_tokens = [_tokenizer(x) for x in texts]  # List[List[str]]

        # Train TF-IDF models
        tfidf_model = nmw.TfIdf()
        tfidf_model.train(train_x_tokens)
        tfidf_model.save(model_path)
