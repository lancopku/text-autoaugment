# Text-AutoAugment (TAA)
This repository contains the code for our paper [Text AutoAugment: Learning Compositional Augmentation Policy for Text Classification](https://arxiv.org/abs/2109.00523) (EMNLP 2021 main conference).

![Overview of IAIS](figures/taa.png)

**************************** Updates ****************************
- 21.10.27: We make taa installable as a package and adapt to [huggingface/transformers](https://github.com/huggingface/transformers). 
Now you can search augmentation policy for your custom dataset with **TWO** lines of code.

## Quick Links
* [Overview](#overview)
* [Getting Started](#getting-started)
* [Prepare environment](#prepare-environment)
* [Use TAA with Huggingface](#use-taa-with-huggingface)
  - [Get augmented training dataset with TAA policy](#get-augmented-training-dataset-with-taa-policy)
    * [Search for the optimal policy](#search-for-the-optimal-policy)
    * [Use our pre-searched policy](#use-our-pre-searched-policy)
  - [Fine-tune a new model on the augmented training dataset](#fine-tune-a-new-model-on-the-augmented-training-dataset)
* [Reproduce results in the paper](#reproduce-results-in-the-paper)
* [Contact](#contact)
* [Acknowledgments](#acknowledgments)
* [Citation](#citation)
* [License](#license)

## Overview
1. We  present  a  learnable  and  compositional framework for data augmentation.  Our proposed algorithm automatically searches for the optimal compositional policy, which improves the diversity and quality of augmented samples.

2. In low-resource and class-imbalanced regimes of six benchmark datasets, TAA significantly improves the generalization ability of deep neural networks like  BERT and effectively boosts text classification performance.

## Getting Started

### Prepare environment

Install pytorch, torchvision and other small additional dependencies. Then, install this repo as a python package. Note that `cudatoolkit=10.0` should match the CUDA version on your machine.

```bash
conda install pytorch torchvision cudatoolkit=10.0 -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch
pip install git+https://github.com/wbaek/theconf
pip install git+https://github.com/ildoonet/pystopwatch2.git
pip install git+https://github.com/lancopku/text-autoaugment.git
python -c "import nltk; nltk.download('wordnet'); nltk.download('averaged_perceptron_tagger')"
```

### Use TAA with Huggingface

#### Get augmented training dataset with TAA policy

##### Search for the optimal policy

You can search for the optimal policy on classification datasets supported by [huggingface/datasets](https://huggingface.co/datasets):
```bash
from taa.search_and_augment import search_and_augment

# return the augmented train dataset in the form of torch.utils.data.Dataset
augmented_train_dataset = search_and_augment(configfile="/path/to/your/config.yaml")
```

The `configfile` contains some preset arguments, including:

- `model`:
  - `type`: *backbone model*
- `dataset`:
  - `path`: *Path or name of the dataset*
  - `name`: *Defining the name of the dataset configuration*
  - `data_dir`: *Defining the data_dir of the dataset configuration*
  - `data_files`: *Path(s) to source data file(s)*
  
  All the augments above are used for the `load_dataset()` function in [huggingface/datasets](https://huggingface.co/datasets). Please refer to [link](https://huggingface.co/docs/datasets/v1.12.1/package_reference/loading_methods.html#datasets.load_dataset) for details. 
  - `text_key`: *Used to get text from a data instance (`dict` form in huggingface/datasets. See this [IMDB example](https://huggingface.co/datasets/imdb#data-instances).)*
- `abspath`: *Your working directory*
- `aug`: *Pre-searched policy*. Now we support imdb, sst5, trec, yelp2 and yelp5. See [archive.py](taa/archive.py).
- `per_device_train_batch_size`: *Batch size per device for training*
- `per_device_eval_batch_size`: *Batch size per device for evaluation*
- `epoch`: *Training epoch*
- `lr`: *Learning rate*
- `max_seq_length`
- `n_aug`: *Augment each text sample n_aug times*
- `num_op`: *Number of operations per sub-policy*
- `num_policy`: *Number of sub-policy per policy*
- `method`: *Search method (taa)*
- `topN`: *Ensemble topN sub-policy to get final policy*
- `ir`: *Imbalance rate*
- `seed`: *Random seed*
- `trail`: *Trail under current random seed*
- `train`:
  - `npc`: *Number of examples per class in the training dataset*
- `valid`:
  - `npc`: *Number of examples per class in the val dataset*
- `test`:
  - `npc`: *Number of examples per class in the test dataset*
- `num_search`: *Number of optimization iteration*
- `num_gpus`: *Number of GPUs used in RAY*
- `num_cpus`: *Number of CPUs used in RAY*

**ATTENTION**: [bert_sst2_example.yaml](taa/confs/bert_sst2_example.yaml) is a config example for BERT model and [SST2](https://huggingface.co/datasets/glue#sst2) dataset. You can follow this example to create your own config file. For instance, if you only want to change the dataset from `sst2` to `imdb`, just delete the `sst2` in the `'path'` argument, modify the `'name'` to `imdb` and modity the `'text_key'` to `text`.

**WARNING**: The policy optimization framework is based on [ray](https://github.com/ray-project/ray). By default we use 4 GPUs and 40 CPUs for policy optimization. Make sure your computing resources meet this condition, or you will need to create a new configuration file. And please specify the gpus, e.g., `CUDA_VISIBLE_DEVICES=0,1,2,3` before using the above code. TPU does not seem to be supported now.   

##### Use our pre-searched policy

To train a model on the datasets augmented by our pre-searched policy, please use (Take [IMDB](https://huggingface.co/datasets/imdb) as an example):
```bash
from taa.search_and_augment import augment_with_presearched_policy

# return the augmented train dataset in the form of torch.utils.data.Dataset
augmented_train_dataset = augment_with_presearched_policy(configfile="/path/to/your/config.yaml")
```

Now we support imdb, sst5, trec, yelp2 and yelp5. See [archive.py](taa/archive.py) for details. 

This table lists the test accuracy (%) of pre-searched TAA policy on **full** datasets:

| Dataset |  IMDB | SST-5 |  TREC | YELP-2 | YELP-5 |
|---------|:-----:|:-----:|:-----:|:------:|:------:|
| No Aug  | 88.77 | 52.29 | 96.40 |  95.85 |  65.55 |
| TAA     | 89.37 | 52.55 | 97.07 |  96.04 |  65.73 |
| n_aug   |   4   |   4   |   4   |    2   |    2   |

More pre-searched policies and their performance will be coming soon. 

#### Fine-tune a new model on the augmented training dataset

After getting `augmented_train_dataset`, you can load it to the huggingface trainer directly. Please refer to [search_augment_train.py](taa/search_augment_train.py) for details. 

### Reproduce results in the paper

Please run [huggingface_lowresource.sh](taa/script/huggingface_lowresource.sh) or [huggingface_imbalanced.sh](taa/script/huggingface_imbalanced.sh).

## Contact

If you have any questions related to the code or the paper, feel free to open an issue or email Shuhuai (renshuhuai007 [AT] gmail [DOT] com).

## Acknowledgments
Code refers to: [fast-autoaugment](https://github.com/kakaobrain/fast-autoaugment).

## Citation

If you find this code useful for your research, please consider citing:
```
@inproceedings{ren2021taa,
  title={Text AutoAugment: Learning Compositional Augmentation Policy for Text Classification},
  author={Shuhuai Ren, Jinchao Zhang, Lei Li, Xu Sun, Jie Zhou},
  booktitle={EMNLP},
  year={2021}
}
```

## License

MIT
