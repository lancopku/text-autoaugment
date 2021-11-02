import urllib.request
import zipfile
import tarfile
import os
import csv
import sys
import re
dataroot = os.path.join(os.path.abspath(os.path.dirname(os.path.dirname(__file__))), 'data')


def progress(block_num, block_size, total_size):
    sys.stdout.write('\r>> Downloading %s %.1f%%' % (dataroot, float(block_num * block_size) / float(total_size) * 100.0))
    sys.stdout.flush()


def download_imdb():
    imdb_root = os.path.join(dataroot, "aclImdb")

    def _dump_raw_data(contents, file_path):
        with open(file_path, "w", encoding='utf-8') as ouf:
            writer = csv.writer(ouf, delimiter="\t", quotechar="\"")
            for line in contents:
                writer.writerow(line)

    def _load_data_by_id(sub_set, id_path):
        with open(id_path, encoding='utf-8') as inf:
            id_list = inf.readlines()
        contents = []
        for example_id in id_list:
            example_id = example_id.strip()
            label = example_id.split("_")[0]
            file_path = os.path.join(imdb_root, sub_set, label, example_id[len(label) + 1:])
            with open(file_path, encoding='utf-8') as inf:
                st_list = inf.readlines()
                assert len(st_list) == 1
                # st = clean_web_text(st_list[0].strip())
                st = st_list[0]
                contents += [(st, label, example_id)]
        return contents

    def _load_all_data(sub_set):
        contents = []
        for label in ["pos", "neg", "unsup"]:
            data_path = os.path.join(imdb_root, sub_set, label)
            if not os.path.exists(data_path):
                continue
            for filename in os.listdir(data_path):
                file_path = os.path.join(data_path, filename)
                with open(file_path, encoding='utf-8') as inf:
                    st_list = inf.readlines()
                    assert len(st_list) == 1
                    # st = clean_web_text(st_list[0].strip())
                    st = st_list[0]
                    example_id = "{}_{}".format(label, filename)
                    contents += [(st, label, example_id)]
        return contents

    # download data
    if not os.path.exists(os.path.join(imdb_root, 'train_id_list.txt')):
        filename = os.path.join(dataroot, "aclImdb_v1.tar.gz")
        urllib.request.urlretrieve("http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz", filename, progress)
        with tarfile.open(filename, "r") as tar_ref:
            tar_ref.extractall(dataroot)
        os.remove(filename)
        urllib.request.urlretrieve(
            "https://raw.githubusercontent.com/google-research/uda/master/text/data/IMDB_raw/train_id_list.txt",
            os.path.join(imdb_root, 'train_id_list.txt'))
    if not os.path.exists(os.path.join(imdb_root, "train.csv")):
        # load train
        header = ["content", "label", "id"]
        contents = _load_data_by_id("train", os.path.join(imdb_root, 'train_id_list.txt'))
        _dump_raw_data([header] + contents, os.path.join(imdb_root, "train.csv"))
        # load test
        contents = _load_all_data("test")
        _dump_raw_data([header] + contents, os.path.join(imdb_root, "test.csv"))


def download_sst2():
    '''reference to https://github.com/tanyuqian/learning-data-manipulation/blob/master/data_utils/download_sst2.py.
    '''
    def _clean_sst_text(text):
        """Cleans tokens in the SST data, which has already been tokenized.
        """
        text = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", text)
        text = re.sub(r"\s{2,}", " ", text)
        return text.strip().lower()

    def _transform_raw_sst(data_path, raw_fn, new_fn):
        """Transforms the raw data format to a new format.
        """
        fout_x_name = os.path.join(data_path, new_fn + '.sentences.txt')
        fout_x = open(fout_x_name, 'w', encoding='utf-8')
        fout_y_name = os.path.join(data_path, new_fn + '.labels.txt')
        fout_y = open(fout_y_name, 'w', encoding='utf-8')

        fin_name = os.path.join(data_path, raw_fn)
        with open(fin_name, 'r', encoding='utf-8') as fin:
            for line in fin:
                parts = line.strip().split()
                label = parts[0]
                sent = ' '.join(parts[1:])
                sent = _clean_sst_text(sent)
                fout_x.write(sent + '\n')
                fout_y.write(label + '\n')

        return fout_x_name, fout_y_name

    sst2_root = os.path.join(dataroot, 'sst2')
    if not os.path.exists(sst2_root):
        os.makedirs(sst2_root)
        url = ('https://raw.githubusercontent.com/ZhitingHu/'
               'logicnn/master/data/raw/')
        files = ['stsa.binary.train', 'stsa.binary.dev', 'stsa.binary.test']
        for fn in files:
            urllib.request.urlretrieve(url + fn, os.path.join(sst2_root, fn), progress)

        _transform_raw_sst(sst2_root, 'stsa.binary.train', 'sst2.train')
        _transform_raw_sst(sst2_root, 'stsa.binary.dev', 'sst2.dev')
        _transform_raw_sst(sst2_root, 'stsa.binary.test', 'sst2.test')


def download_sst5():
    sst5_root = os.path.join(dataroot, "trees")
    # download data
    if not os.path.exists(sst5_root):
        os.makedirs(sst5_root)
        filename = os.path.join(dataroot, "trainDevTestTrees_PTB.zip")
        urllib.request.urlretrieve("http://nlp.stanford.edu/sentiment/trainDevTestTrees_PTB.zip", filename, progress)
        with zipfile.ZipFile(filename, "r") as tar_ref:
            tar_ref.extractall(dataroot)
        os.remove(filename)


def download_trec():
    trec_root = os.path.join(dataroot, 'trec')
    if not os.path.exists(trec_root):
        os.makedirs(trec_root)
        urllib.request.urlretrieve('http://cogcomp.org/Data/QA/QC/train_5500.label', os.path.join(trec_root, 'train_5500.label'), progress)
        urllib.request.urlretrieve('http://cogcomp.org/Data/QA/QC/TREC_10.label', os.path.join(trec_root, 'TREC_10.label'), progress)


def download_yelp2():
    yelp2_root = os.path.join(dataroot, "yelp_review_polarity_csv")
    # download data
    if not os.path.exists(yelp2_root):
        filename = os.path.join(dataroot, "yelp_review_polarity_csv.tar.gz")
        urllib.request.urlretrieve("https://s3.amazonaws.com/fast-ai-nlp/yelp_review_polarity_csv.tgz", filename, progress)
        with tarfile.open(filename, "r") as tar_ref:
            tar_ref.extractall(dataroot)
        os.remove(filename)


def download_yelp5():
    yelp5_root = os.path.join(dataroot, "yelp_review_full_csv")
    # download data
    if not os.path.exists(yelp5_root):
        filename = os.path.join(dataroot, "yelp_review_full_csv.tar.gz")
        urllib.request.urlretrieve("https://s3.amazonaws.com/fast-ai-nlp/yelp_review_full_csv.tgz", filename, progress)
        with tarfile.open(filename, "r") as tar_ref:
            tar_ref.extractall(dataroot)
        os.remove(filename)


download_function = {'imdb': download_imdb,
                     'sst2': download_sst2,
                     'sst5': download_sst5,
                     'trec': download_trec,
                     'yelp2': download_yelp2,
                     'yelp5': download_yelp5
                     }


def download_data(dataset):
    download_function[dataset]()
    print("All datasets downloaded and extracted")


if __name__ == '__main__':
    download_data('sst5')
