# Text-AutoAugment (TAA)
This repository contains the code for our paper [Text AutoAugment: Learning Compositional Augmentation Policy for Text Classification](https://arxiv.org/abs/2109.00523) (EMNLP 2021 main conference).

![Overview of IAIS](figures/taa.png)

## Overview
1. We  present  a  learnable  and  compositional framework for data augmentation.  Our proposed algorithm automatically searches for the optimal compositional policy, which improves the diversity and quality of augmented samples.

2. In low-resource and class-imbalanced regimes of six benchmark datasets, TAA significantly improves the generalization ability of deep neural networks like  BERT and effectively boosts text classification performance.

## Getting Started
1. Prepare environment
    ```bash
    conda create -n taa python=3.6
    conda activate taa
    conda install pytorch torchvision cudatoolkit=10.0 -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch
    pip install -r requirements.txt 
    python -c "import nltk; nltk.download('wordnet'); nltk.download('averaged_perceptron_tagger')"
    ```

2. Modify `dataroot` parameter in `confs/*yaml` and `abspath` parameter in `script/*.sh`:
    - e.g., change `dataroot: /home/renshuhuai/TextAutoAugment/data/aclImdb` in [confs/bert_imdb.yaml](confs/bert_imdb.yaml) to `dataroot: path-to-your-TextAutoAugment/data/aclImdb`
    - change `--abspath '/home/renshuhuai/TextAutoAugment'` in [script/imdb_lowresource.sh](script/imdb_lowresource.sh) to `--abspath 'path-to-your-TextAutoAugment'`

3. Search for the best augmentation policy, e.g., low-resource regime for IMDB:
   
   ```bash
   sh script/imdb_lowresource.sh
   ```
   scripts for policy search in the low-resource and class-imbalanced regime for all datasets are provided in the [script/](script/) fold.

4. Train a model with pre-searched policy in [archive.py](archive.py), e.g., train model in low-resource regime for IMDB: 
    ```bash
   python train.py -c confs/bert_imdb.yaml 
    ```   
   train model on **full** dataset of IMDB:
   ```bash
   python train.py -c confs/bert_imdb.yaml --train-npc -1 --valid-npc -1 --test-npc -1  
   ```
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
