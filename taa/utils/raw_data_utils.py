from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from sklearn.model_selection import StratifiedShuffleSplit, KFold
import torchtext
import csv
import os
import numpy as np
import random
import pandas as pd
import re
from theconf import Config as C
from ..common import get_logger
import logging

logger = get_logger('Text AutoAugment')
logger.setLevel(logging.INFO)


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
          guid: Unique id for the example.
          text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
          text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
          label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, raw_data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, raw_data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    def get_train_size(self):
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None, delimiter="\t"):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding='utf-8') as f:
            reader = csv.reader(f, delimiter=delimiter, quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines


def clean_web_text(st):
    """clean text."""
    st = st.replace("<br />", " ")
    st = st.replace("&quot;", "\"")
    st = st.replace("<p>", " ")
    if "<a href=" in st:
        # print("before:\n", st)
        while "<a href=" in st:
            start_pos = st.find("<a href=")
            end_pos = st.find(">", start_pos)
            if end_pos != -1:
                st = st[:start_pos] + st[end_pos + 1:]
            else:
                print("incomplete href")
                print("before", st)
                st = st[:start_pos] + st[start_pos + len("<a href=")]
                print("after", st)

        st = st.replace("</a>", "")
        # print("after\n", st)
        # print("")
    st = st.replace("\\n", " ")
    st = st.replace("\\", " ")
    # while "  " in st:
    #   st = st.replace("  ", " ")
    return st


def subsample_by_classes(all_examples, labels, tag):
    if C.get()['ir'] == 1:
        return all_examples

    num_per_class = {label: sum([e.label == label for e in all_examples]) for label in labels}
    logger.info("{}_num_per_class before: {}".format(tag, num_per_class))
    num_per_class[labels[0]] = round(num_per_class[labels[0]] * C.get()['ir'])
    logger.info("{}_num_per_class after:  {}".format(tag, num_per_class))

    examples = {label: [] for label in labels}
    for example in all_examples:
        examples[example.label].append(example)

    selected_examples = []
    for label in labels:
        random.seed(C.get()['seed'])
        random.shuffle(examples[label])

        num_in_class = num_per_class[label]
        selected_examples = selected_examples + examples[label][:num_in_class]

    return selected_examples


class IMDbProcessor(DataProcessor):
    """Processor for the CoLA data set (GLUE version)."""

    def get_train_examples(self, raw_data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(raw_data_dir, "train.csv"), quotechar='"'), "train")

    def get_test_examples(self, raw_data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(raw_data_dir, "test.csv"), quotechar='"'), "test")

    def get_unsup_examples(self, raw_data_dir, unsup_set):
        """See base class."""
        if unsup_set == "unsup_ext":
            return self._create_examples(self._read_tsv(os.path.join(raw_data_dir, "unsup_ext.csv"), quotechar='"'),
                                         "unsup_ext", skip_unsup=False)
        elif unsup_set == "unsup_in":
            return self._create_examples(self._read_tsv(os.path.join(raw_data_dir, "train.csv"), quotechar='"'),
                                         "unsup_in", skip_unsup=False)

    def get_labels(self):
        """See base class."""
        return ["pos", "neg"]

    def _create_examples(self, lines, set_type, skip_unsup=True):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            if skip_unsup and line[1] == "unsup":
                continue
            if line[1] == "unsup" and len(line[0]) < 500:
                # tf.logging.info("skipping short samples:{:s}".format(line[0]))
                continue
            guid = "%s-%s" % (set_type, line[2])
            text_a = line[0]
            label = line[1]
            text_a = clean_web_text(text_a)
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples

    def get_train_size(self):
        return 25000

    def get_test_size(self):
        return 25000

    def split(self, examples, test_size, train_size, n_splits=2, split_idx=0):
        label_map = {"pos": 0, "neg": 1}
        labels = [label_map[e.label] for e in examples]
        kf = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size, train_size=train_size,
                                    random_state=C.get()['seed'])
        kf = kf.split(list(range(len(examples))), labels)
        for _ in range(split_idx + 1):  # split_idx equal to cv_fold. this loop is used to get i-th fold
            train_idx, valid_idx = next(kf)
        train_dev_set = np.array(examples)
        return list(train_dev_set[train_idx]), list(train_dev_set[valid_idx])


class STSProcessor(DataProcessor):
    def get_train_examples(self, raw_data_dir):
        return self._create_examples(
            pd.read_csv(os.path.join(raw_data_dir, 'sts-train-dev.csv'), header=None, sep='\t', quoting=csv.QUOTE_NONE,
                        encoding='utf-8', usecols=[3, 4, 5, 6]), "train")

    def get_test_examples(self, raw_data_dir):
        return self._create_examples(
            pd.read_csv(os.path.join(raw_data_dir, 'sts-test.csv'), header=None, sep='\t', quoting=csv.QUOTE_NONE,
                        encoding='utf-8', usecols=[3, 4, 5, 6]), "test")

    def _create_examples(self, lines, set_type, skip_unsup=True):
        examples = []
        for (i, line) in lines.iterrows():
            guid = "%s-%s" % (set_type, line[3])
            text_a = line[5]
            text_b = line[6]
            label = line[4]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

    def get_labels(self):
        return []  # from 0.0 to 5.0

    def get_train_size(self):
        return 7249

    def get_test_size(self):
        return 1379

    def split(self, examples, n_splits=None, split=None, split_idx=None):  # TODO control size
        kf = KFold(n_splits=n_splits, random_state=C.get()['seed'])
        kf = kf.split(list(range(len(examples))))
        for _ in range(split_idx + 1):  # split_idx equal to cv_fold. this loop is used to get i-th fold
            train_idx, valid_idx = next(kf)
        train_dev_set = np.array(examples)
        return list(train_dev_set[train_idx]), list(train_dev_set[valid_idx])


class MRPCProcessor(DataProcessor):
    def get_train_examples(self, raw_data_dir):
        return self._create_examples(self._read_tsv(os.path.join(raw_data_dir, 'msr_paraphrase_train.txt')), "train")

    def get_test_examples(self, raw_data_dir):
        return self._create_examples(self._read_tsv(os.path.join(raw_data_dir, 'msr_paraphrase_test.txt')), "test")

    def get_labels(self):
        return [0, 1]

    def _create_examples(self, lines, set_type):
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = line[3]
            text_b = line[4]
            label = int(line[0])
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

    def get_train_size(self):
        return 4076  # label_0:1323; label_1:2753

    def get_test_size(self):
        return 1725  # label_0:578; label_1:1147

    def split(self, examples, test_size, train_size, n_splits=2, split_idx=0):
        labels = [e.label for e in examples]
        kf = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size, train_size=train_size,
                                    random_state=C.get()['seed'])
        kf = kf.split(list(range(len(examples))), labels)
        for _ in range(split_idx + 1):  # split_idx equal to cv_fold. this loop is used to get i-th fold
            train_idx, valid_idx = next(kf)
        train_dev_set = np.array(examples)
        return list(train_dev_set[train_idx]), list(train_dev_set[valid_idx])


class SST2Processor(DataProcessor):
    """Processor for the SST-2 data set (GLUE version)."""

    def get_train_examples(self, raw_data_dir):
        train_e = self._create_examples(os.path.join(raw_data_dir, 'stsa.binary.train'), "train")
        dev_e = self._create_examples(os.path.join(raw_data_dir, 'stsa.binary.dev'), "dev")
        train_e.extend(dev_e)
        return train_e

    def get_test_examples(self, raw_data_dir):
        return self._create_examples(os.path.join(raw_data_dir, 'stsa.binary.test'), "test")

    def get_labels(self):
        """See base class."""
        return [0, 1]

    def _create_examples(self, dataroot, set_type):
        examples = []
        with open(dataroot, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                guid = "%s-%s" % (set_type, i)
                parts = line.strip().split()
                label = int(parts[0])
                text_a = ' '.join(parts[1:])
                text_a = self.clean_sst_text(text_a)
                examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples

    def clean_sst_text(self, text):
        """Cleans tokens in the SST data, which has already been tokenized.
        """
        text = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", text)
        text = re.sub(r"\s{2,}", " ", text)
        return text.strip().lower()

    def get_train_size(self):
        return 7791  # 6919+872

    def get_test_size(self):
        return 1821

    def split(self, examples, test_size, train_size, n_splits=2, split_idx=0):
        labels = [e.label for e in examples]
        kf = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size, train_size=train_size,
                                    random_state=C.get()['seed'])
        kf = kf.split(list(range(len(examples))), labels)
        for _ in range(split_idx + 1):  # split_idx equal to cv_fold. this loop is used to get i-th fold
            train_idx, valid_idx = next(kf)
        train_dev_set = np.array(examples)
        return list(train_dev_set[train_idx]), list(train_dev_set[valid_idx])


class SST5Processor(DataProcessor):
    def __init__(self):
        self.TEXT = torchtext.data.Field()
        self.LABEL = torchtext.data.Field(sequential=False)

    def get_train_examples(self, raw_data_dir):
        train_e = torchtext.datasets.SST(os.path.join(raw_data_dir, 'train.txt'), self.TEXT, self.LABEL,
                                         fine_grained=True).examples
        dev_e = torchtext.datasets.SST(os.path.join(raw_data_dir, 'dev.txt'), self.TEXT, self.LABEL,
                                       fine_grained=True).examples
        train_e.extend(dev_e)
        return self._create_examples(train_e, "train")

    def get_test_examples(self, raw_data_dir):
        test_e = torchtext.datasets.SST(os.path.join(raw_data_dir, 'test.txt'), self.TEXT, self.LABEL,
                                        fine_grained=True).examples
        return self._create_examples(test_e, "test")

    def get_labels(self):
        return ['negative', 'very positive', 'neutral', 'positive', 'very negative']

    def _create_examples(self, lines, set_type, skip_unsup=True):
        examples = []
        for i, line in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            examples.append(InputExample(guid=guid, text_a=' '.join(line.text), text_b=None, label=line.label))
        return examples

    def get_train_size(self):
        return 9643  # 8542+1101

    def get_test_size(self):
        return 2210

    def split(self, examples, test_size, train_size, n_splits=2, split_idx=0):
        label_map = {"negative": 0, "very positive": 1, 'neutral': 2, 'positive': 3, 'very negative': 4}
        labels = [label_map[e.label] for e in examples]
        kf = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size, train_size=train_size,
                                    random_state=C.get()['seed'])
        kf = kf.split(list(range(len(examples))), labels)
        for _ in range(split_idx + 1):  # split_idx equal to cv_fold. this loop is used to get i-th fold
            train_idx, valid_idx = next(kf)
        train_dev_set = np.array(examples)
        return list(train_dev_set[train_idx]), list(train_dev_set[valid_idx])


class TRECProcessor(DataProcessor):
    def __init__(self):
        self.TEXT = torchtext.data.Field()
        self.LABEL = torchtext.data.Field(sequential=False)

    def get_train_examples(self, raw_data_dir):
        train_e = torchtext.datasets.TREC(os.path.join(raw_data_dir, 'train_5500.label'), self.TEXT, self.LABEL,
                                          fine_grained=False).examples
        return self._create_examples(train_e, "train")

    def get_test_examples(self, raw_data_dir):
        test_e = torchtext.datasets.TREC(os.path.join(raw_data_dir, 'TREC_10.label'), self.TEXT, self.LABEL,
                                         fine_grained=False).examples
        return self._create_examples(test_e, "test")

    def get_labels(self):
        return ['ENTY', 'DESC', 'LOC', 'ABBR', 'NUM', 'HUM']

    def _create_examples(self, lines, set_type, skip_unsup=True):
        examples = []
        for i, line in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            examples.append(InputExample(guid=guid, text_a=' '.join(line.text), text_b=None, label=line.label))
        return examples

    def get_train_size(self):
        return 5452

    def get_test_size(self):
        return 500

    def split(self, examples, test_size, train_size, n_splits=2, split_idx=0):
        label_map = {"ENTY": 0, "DESC": 1, 'LOC': 2, 'ABBR': 3, 'NUM': 4, 'HUM': 5}
        labels = [label_map[e.label] for e in examples]
        kf = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size, train_size=train_size,
                                    random_state=C.get()['seed'])
        kf = kf.split(list(range(len(examples))), labels)
        for _ in range(split_idx + 1):  # split_idx equal to cv_fold. this loop is used to get i-th fold
            train_idx, valid_idx = next(kf)
        train_dev_set = np.array(examples)
        return list(train_dev_set[train_idx]), list(train_dev_set[valid_idx])


class TextClassProcessor(DataProcessor):

    def get_train_examples(self, raw_data_dir):
        """See base class."""
        examples = self._create_examples(
            self._read_tsv(os.path.join(raw_data_dir, "train.csv"), quotechar="\"", delimiter=","), "train")
        assert len(examples) == self.get_train_size()
        return examples

    def get_test_examples(self, raw_data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(raw_data_dir, "test.csv"), quotechar="\"", delimiter=","), "test")

    def get_unsup_examples(self, raw_data_dir, unsup_set):
        """See base class."""
        if unsup_set == "unsup_in":
            return self._create_examples(
                self._read_tsv(os.path.join(raw_data_dir, "train.csv"), quotechar="\"", delimiter=","), "unsup_in",
                skip_unsup=False)
        else:
            return self._create_examples(
                self._read_tsv(os.path.join(raw_data_dir, "{:s}.csv".format(unsup_set)), quotechar="\"", delimiter=","),
                unsup_set, skip_unsup=False)

    def _create_examples(self, lines, set_type, skip_unsup=True, only_unsup=False):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if skip_unsup and line[0] == "unsup":
                continue
            if only_unsup and line[0] != "unsup":
                continue
            guid = "%s-%d" % (set_type, i)
            if self.has_title:
                text_a = line[2]
                text_b = line[1]
            else:
                text_a = line[1]
                text_b = None
            label = int(line[0]) - 1  # TODO right for all datasets??
            text_a = clean_web_text(text_a)
            if text_b is not None:
                text_b = clean_web_text(text_b)
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

    def split(self, examples, test_size, train_size, n_splits=2, split_idx=0):
        labels = [e.label for e in examples]
        kf = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size, train_size=train_size,
                                    random_state=C.get()['seed'])
        kf = kf.split(list(range(len(examples))), labels)
        for _ in range(split_idx + 1):  # split_idx equal to cv_fold. this loop is used to get i-th fold
            train_idx, valid_idx = next(kf)
        train_dev_set = np.array(examples)
        return list(train_dev_set[train_idx]), list(train_dev_set[valid_idx])


class YELP2Processor(TextClassProcessor):

    def __init__(self):
        self.has_title = False

    def get_labels(self):
        """See base class."""
        return [0, 1]

    def get_train_size(self):
        return 560000

    def get_dev_size(self):
        return 38000


class YELP5Processor(TextClassProcessor):

    def __init__(self):
        self.has_title = False

    def get_labels(self):
        """See base class."""
        return [i for i in range(0, 5)]

    def get_train_size(self):
        return 650000

    def get_dev_size(self):
        return 50000


class AMAZON2Processor(TextClassProcessor):

    def __init__(self):
        self.has_title = True

    def get_labels(self):
        """See base class."""
        return [0, 1]

    def get_train_size(self):
        return 3600000

    def get_dev_size(self):
        return 400000

    def get_unsup_examples(self, raw_data_dir, unsup_set):
        """See base class."""
        if unsup_set == "unsup_in":
            return self._create_examples(
                self._read_tsv(
                    os.path.join(raw_data_dir, "train.csv"),
                    quotechar="\"",
                    delimiter=","),
                "unsup_in", skip_unsup=False)
        else:
            dir_cell = raw_data_dir[5:7]
            unsup_dir = None  # update this path if you use unsupervised data
            return self._create_examples(
                self._read_tsv(
                    os.path.join(unsup_dir, "{:s}.csv".format(unsup_set)),
                    quotechar="\"",
                    delimiter=","),
                unsup_set, skip_unsup=False)


class AMAZON5Processor(TextClassProcessor):
    def __init__(self):
        self.has_title = True

    def get_labels(self):
        """See base class."""
        return [str(i) for i in range(1, 6)]  # TODO why range(0,5)?

    def get_unsup_examples(self, raw_data_dir, unsup_set):
        """See base class."""
        if unsup_set == "unsup_in":
            return self._create_examples(
                self._read_tsv(os.path.join(raw_data_dir, "train.csv"), quotechar="\"", delimiter=","), "unsup_in",
                skip_unsup=False)
        else:
            dir_cell = raw_data_dir[5:7]
            unsup_dir = None  # update this path if you use unsupervised data
            return self._create_examples(
                self._read_tsv(os.path.join(unsup_dir, "{:s}.csv".format(unsup_set)), quotechar="\"", delimiter=","),
                unsup_set, skip_unsup=False)

    def get_train_size(self):
        return 3000000

    def get_dev_size(self):
        return 650000


class DBPediaProcessor(TextClassProcessor):

    def __init__(self):
        self.has_title = True

    def get_labels(self):
        """See base class."""
        return [str(i) for i in range(1, 15)]

    def get_train_size(self):
        return 560000

    def get_dev_size(self):
        return 70000


def get_processor(task_name):
    """get processor."""
    task_name = task_name.lower()
    processors = {
        "imdb": IMDbProcessor,
        "dbpedia": DBPediaProcessor,
        "yelp2": YELP2Processor,
        "yelp5": YELP5Processor,
        "amazon2": AMAZON2Processor,
        "amazon5": AMAZON5Processor,
        'sts': STSProcessor,
        'mrpc': MRPCProcessor,
        'sst2': SST2Processor,
        'sst5': SST5Processor,
        'trec': TRECProcessor
    }
    processor = processors[task_name]()
    return processor


def get_examples(dataset, text_key='text'):
    """get dataset examples"""
    examples = []
    for i in range(dataset.num_rows):
        guid = i
        text_a = dataset[i][text_key]
        label = dataset[i]['label']
        text_a = clean_web_text(text_a)
        examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
    return examples


def general_split(examples, test_size, train_size, n_splits=2, split_idx=0):
    """used for datasets on huggingface"""
    labels = [e.label for e in examples]
    kf = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size, train_size=train_size,
                                    random_state=C.get()['seed'])
    kf = kf.split(list(range(len(examples))), labels)
    for _ in range(split_idx + 1):  # split_idx equal to cv_fold. this loop is used to get i-th fold
        train_idx, valid_idx = next(kf)
    train_dev_set = np.array(examples)
    return list(train_dev_set[train_idx]), list(train_dev_set[valid_idx])


def general_subsample_by_classes(all_examples, labels, label_mapping, tag):
    if C.get()['ir'] == 1:
        return all_examples

    num_per_class = {label: sum([e.label == label_mapping[label] for e in all_examples]) for label in labels}
    logger.info("{}_num_per_class before: {}".format(tag, num_per_class))
    num_per_class[labels[0]] = round(num_per_class[labels[0]] * C.get()['ir'])
    logger.info("{}_num_per_class after:  {}".format(tag, num_per_class))

    examples = {label: [] for label in labels}
    for example in all_examples:
        for label in labels:
            if label_mapping[label] == example.label:
                examples[label].append(example)

    selected_examples = []
    for label in labels:
        random.seed(C.get()['seed'])
        random.shuffle(examples[label])

        num_in_class = num_per_class[label]
        selected_examples = selected_examples + examples[label][:num_in_class]

    return selected_examples


if __name__ == '__main__':
    pc = get_processor('yelp5')
    pc.get_train_examples('/home/renshuhuai/text-autoaugment/data/yelp_review_full_csv')
