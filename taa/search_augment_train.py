import os
import math
import torch
import shutil
import logging
import transformers
from .data import augment
from .train import compute_metrics
from .search import search_policy
from datasets import load_dataset
import pandas as pd
import numpy as np
from theconf import Config as C
from .utils.raw_data_utils import get_examples
from .custom_dataset import general_dataset
from .text_networks import get_model, num_class, get_num_class
from transformers import BertForSequenceClassification, Trainer, TrainingArguments, BertTokenizerFast
from .common import get_logger, add_filehandler
from .utils.train_tfidf import train_tfidf
import joblib


def train_bert(tag, augmented_train_dataset, valid_dataset, test_dataset, policy_opt, save_path=None, only_eval=False):
    transformers.logging.set_verbosity_info()
    logger = get_logger('Text AutoAugment')
    logger.setLevel(logging.INFO)

    config = C.get()
    dataset_type = config['dataset']['name']
    model_type = config['model']['type']
    C.get()['tag'] = tag
    text_key = C.get()['dataset']['text_key']

    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

    valid_examples = get_examples(valid_dataset, text_key)
    test_examples = get_examples(test_dataset, text_key)

    train_dataset = augmented_train_dataset
    val_dataset = general_dataset(valid_examples, tokenizer, text_transform=None)
    test_dataset = general_dataset(test_examples, tokenizer, text_transform=None)

    do_train = True
    logging_dir = os.path.join('logs/%s_%s/%s' % (dataset_type, model_type, tag))

    if save_path and os.path.exists(save_path):
        model_name_or_path = save_path
    else:
        model_name_or_path = 'bert-base-uncased'
        only_eval = False
        logger.info('"%s" file not found. skip to pretrain weights...' % save_path)

    if only_eval:
        do_train = False

    if policy_opt:
        logging_dir = None
        per_device_train_batch_size = config['per_device_train_batch_size']
    else:
        per_device_train_batch_size = int(config[
                                              'per_device_train_batch_size'] / torch.cuda.device_count())  # To make batch size equal to that in policy opt phase

    print('per_device_train_batch_size: ', per_device_train_batch_size)
    warmup_steps = math.ceil(len(train_dataset) / config['per_device_train_batch_size'] * config[
        'epoch'] * 0.1)  # number of warmup steps for learning rate scheduler
    print('warmup_steps: ', warmup_steps)
    # create model
    training_args = TrainingArguments(
        output_dir=save_path,  # output directory
        do_train=do_train,
        do_eval=True,
        evaluation_strategy="epoch",
        num_train_epochs=config['epoch'],  # total number of training epochs
        per_device_train_batch_size=per_device_train_batch_size,  # batch size per device during training
        per_device_eval_batch_size=config['per_device_eval_batch_size'],  # batch size for evaluation
        warmup_steps=warmup_steps,
        learning_rate=float(config['lr']),
        logging_dir=logging_dir,  # directory for storing logs
        load_best_model_at_end=True,
        metric_for_best_model="eval_accuracy",
        seed=config['seed']
    )
    model = BertForSequenceClassification.from_pretrained(model_name_or_path, num_labels=get_num_class(dataset_type))

    trainer = Trainer(
        model=model,  # the instantiated 🤗 Transformers model to be trained
        args=training_args,  # training arguments, defined above
        train_dataset=train_dataset,  # training dataset
        eval_dataset=val_dataset,  # evaluation dataset
        compute_metrics=compute_metrics,
    )

    # logger.info("evaluating on train set")
    # result = trainer.evaluate(eval_dataset=train_dataset)

    if do_train:
        trainer.train()
        if not policy_opt:
            trainer.save_model()
            dirs = os.listdir(save_path)
            for file in dirs:
                if file.startswith("checkpoint-"):
                    shutil.rmtree(os.path.join(save_path, file))

    # logger.info("evaluating on test set")
    # note that when policy_opt, the test_dataset is only a subset of true test_dataset, used for evaluating policy
    # result = trainer.evaluate(eval_dataset=test_dataset)

    # result['n_dist'] = train_dataset.aug_n_dist
    # result['opt_object'] = result['eval_accuracy']
    logger.info("Predicting on test set")
    result = trainer.predict(test_dataset)
    logits = result.predictions
    predictions = np.argmax(logits, axis=1)

    predict_df = pd.read_csv('%s.tsv' % dataset_type, sep='\t')
    predict_df['prediction'] = predictions
    predict_df.to_csv('%s.tsv' % dataset_type, sep='\t', index=False)

    if policy_opt:
        shutil.rmtree(save_path)
    return result


if __name__ == '__main__':
    _ = C('confs/bert_sst2_example.yaml')

    # search augmentation policy for specific dataset
    # search_policy(dataset='sst2', configfile='bert_sst2_example.yaml', abspath='/home/renshuhuai/text-autoaugment' )

    train_dataset = load_dataset('glue', 'sst2', split='train')
    valid_dataset = load_dataset('glue', 'sst2', split='validation')
    test_dataset = load_dataset('glue', 'sst2', split='test')

    # generate augmented dataset
    configfile = 'bert_sst2_example.yaml'
    policy_path = '/home/renshuhuai/text-autoaugment/final_policy/sst2_Bert_seed59_train-npc50_n-aug8_ir1.00_taa.pkl'
    policy = joblib.load(policy_path)
    augmented_train_dataset = augment(dataset=train_dataset, policy=policy, n_aug=8, configfile=configfile)

    # training
    dataset_type = C.get()['dataset']['name']
    model_type = C.get()['model']['type']

    train_tfidf(dataset_type)  # calculate tf-idf score for TS and TI operations

    tag = '%s_%s_with_found_policy' % (dataset_type, model_type)
    save_path = os.path.join('models', tag)

    result = train_bert(tag, augmented_train_dataset, valid_dataset, test_dataset, policy_opt=False,
                        save_path=save_path, only_eval=True)
    # for k,v in result.items():
    #     print('%s:%s' % (k,v))
