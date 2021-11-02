import sys
import itertools
import json
import logging
import math
import os
from collections import OrderedDict
import time
import shutil
import numpy as np
import torch
from theconf import Config as C, ConfigArgumentParser
from .common import get_logger, add_filehandler
from .data import get_datasets
from .text_networks import get_model, num_class, get_num_class
import transformers
from transformers import BertForSequenceClassification, Trainer, TrainingArguments
from .utils.metrics import accuracy, f1, accuracy_score
from .utils.train_tfidf import train_tfidf

transformers.logging.set_verbosity_info()
logger = get_logger('Text AutoAugment')
logger.setLevel(logging.INFO)


def update_result(result, rs):
    for key, setname in itertools.product(['loss', 'top1'], ['train', 'valid', 'test']):
        result['%s_%s' % (key, setname)] = rs[setname][key]
    return result


def compute_metrics(p):
    preds_list, out_label_list = p.predictions, p.label_ids
    preds_list = np.argmax(preds_list, axis=-1)
    return {
        "accuracy": accuracy_score(preds_list, out_label_list) * 100,
        "micro_f1": f1(preds_list, out_label_list, 'micro') * 100,
        "macro_f1": f1(preds_list, out_label_list, 'macro') * 100,
        "weighted_f1": f1(preds_list, out_label_list, 'weighted') * 100,
    }


def train_and_eval(tag, policy_opt, save_path=None, only_eval=False):
    config = C.get()
    dataset_type = config['dataset']['name']
    model_type = config['model']['type']
    C.get()['tag'] = tag
    
    train_dataset, val_dataset, test_dataset = get_datasets(dataset_type, policy_opt=policy_opt)

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
        per_device_train_batch_size = int(config['per_device_train_batch_size'] / torch.cuda.device_count())  # To make batch size equal to that in policy opt phase
    
    print('per_device_train_batch_size: ', per_device_train_batch_size)
    warmup_steps = math.ceil(len(train_dataset) / config['per_device_train_batch_size'] * config['epoch'] * 0.1)  # number of warmup steps for learning rate scheduler
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

    logger.info("Evaluating on test set")
    # note that when policy_opt, the test_dataset is only a subset of true test_dataset, used for evaluating policy
    result = trainer.evaluate(eval_dataset=test_dataset)

    result['n_dist'] = train_dataset.aug_n_dist
    result['opt_object'] = result['eval_accuracy']

    if policy_opt:
        shutil.rmtree(save_path)
    return result


if __name__ == '__main__':
    parser = ConfigArgumentParser(conflict_handler='resolve')
    parser.add_argument('--tag', type=str, default='')
    parser.add_argument('--save', type=str, default='')
    parser.add_argument('--num-op', type=int, default=2)
    parser.add_argument('--num-policy', type=int, default=4)
    parser.add_argument('--only-eval', action='store_true')
    parser.add_argument('--abspath', type=str, default='/home/renshuhuai/text-autoaugment')
    parser.add_argument('--n-aug', type=int, default=16,
                        help='magnification of augmentation. synthesize n-aug for each given sample')
    parser.add_argument('--train-npc', type=int, default=40, help='train example num per class')
    parser.add_argument('--valid-npc', type=int, default=30, help='valid example num per class')
    parser.add_argument('--test-npc', type=int, default=30, help='test example num per class when policy opt')
    parser.add_argument('--ir', type=float, default=1, help='imbalance rate')
    parser.add_argument('--trail', type=int, default=1, help='trail')
    parser.add_argument('--seed', type=int, default=59, help='random seed')
    parser.add_argument('--method', type=str, default='taa', help='augmentation method', choices=['taa', 'random_taa'])
    args = parser.parse_args()

    assert (args.only_eval and args.save) or not args.only_eval, 'checkpoint path not provided in evaluation mode.'

    import time

    dataset_type = C.get()['dataset']['name']
    model_type = C.get()['model']['type']
    train_tfidf(dataset_type)  # calculate tf-idf score for TS and TI operations

    total_computation = 0

    t = time.time()
    tag = '%s_%s_with_found_policy' % (dataset_type, model_type) if args.tag == '' else args.tag
    save_path = os.path.join('models', tag) if args.save == '' else args.save
    result = train_and_eval(tag, policy_opt=False, save_path=save_path, only_eval=args.only_eval)
    elapsed = time.time() - t

    config = C.get()
    logger.info('Done.')
    logger.info('Model: %s' % config['model'])
    logger.info('Augmentation: %s' % config['aug'])
    logger.info('\n' + json.dumps(result, indent=4))
    logger.info('Elapsed time: %.3f Hours' % (elapsed / 3600.))
    logger.info(' '.join(['%s=%.4f;' % (key, result[key]) for key in result.keys()]))
    logger.info(args.save)
