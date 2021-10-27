# Text-AutoAugment (TAA)
This repository contains the code for our paper [Text AutoAugment: Learning Compositional Augmentation Policy for Text Classification](https://arxiv.org/abs/2109.00523) (EMNLP 2021 main conference).

![Overview of IAIS](figures/taa.png)

**************************** Updates ****************************
- 21.10.27: We make taa installable as a package and adapt to [huggingface/transformers](https://github.com/huggingface/transformers). 
Now you can search augmentation policy for your custom dataset with a few lines of code.

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

#### Get training dataset augmented by TAA policy

You can search for the optimal policy on classification datasets supported by [huggingface/datasets](https://huggingface.co/datasets) (Take [SST2](https://huggingface.co/datasets/glue#sst2) as an example):
```bash
from taa.search_and_augment import search_and_augment

abspath = '/home/renshuhuai/Text-AutoAugment' # Modify to your working directory
dataset = 'sst2' # any text classification dataset of GLUE (https://huggingface.co/datasets/viewer/?dataset=glue)
n_aug = 8 # augment each text sample n_aug times

# return the augmented train dataset in the form of torch.utils.data.Dataset
augmented_train_dataset = search_and_augment(dataset=dataset, abspath=abspath, n_aug=n_aug)
```

You can also train a model on the datasets augmented by our pre-searched policy (Take [IMDB](https://huggingface.co/datasets/imdb) as an example):
```bash
from taa.search_and_augment import augment_with_presearched_policy

abspath = '/home/renshuhuai/Text-AutoAugment' # Modify to your working directory
dataset = 'imdb' # Can be chosen from ['imdb', 'sst5', 'trec', 'yelp2', 'yelp5']
n_aug = 8 # augment each text sample n_aug times

# return the augmented train dataset in the form of torch.utils.data.Dataset
augmented_train_dataset = augment_with_presearched_policy(dataset=dataset, abspath=abspath, n_aug=n_aug)
```

More pre-searched policies will be coming soon.

#### Fine-tune a new model on the augmented training dataset

After getting `augmented_train_dataset`, you can load it to the huggingface trainer directly. Please refer to [search_augment_train.py](taa/search_augment_train.py) for details. 

### Reproduce results in the paper

Please run [huggingface_lowresource.sh](taa/script/huggingface_lowresource.sh) or [huggingface_imbalanced.sh](taa/script/huggingface_imbalanced.sh).

## Contact

If you have any questions related to the code or the paper, feel free to email Shuhuai (renshuhuai007 [AT] gmail [DOT] com).

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
