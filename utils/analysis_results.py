import re
import pandas as pd
import numpy as np

aug_pattern = re.compile("\[augment\].*")
pretrain_pattern = re.compile("\[pretrained\].*")
gpu_hours_pattern = re.compile(r"(?<=gpu hours=).+")

aug_acc = []
aug_opt_object = []
aug_micro_f1 = []
aug_macro_f1 = []
aug_weighted_f1 = []
aug_n_dist = []
pre_acc = []
pre_opt_object = []
pre_micro_f1 = []
pre_macro_f1 = []
pre_weighted_f1 = []
pre_n_dist = []
gpu_hours = []
with open(r"models\imdb_Bert_op2_policy4_n-aug16_ir1.00_taa.log", 'r') as f:
    for line in f:
        aug = aug_pattern.findall(line)
        pretrain = pretrain_pattern.findall(line)
        gpu_hours = gpu_hours_pattern.findall(line)
        if len(aug) > 0:
            metrics = re.compile(r"\d+\.\d+").findall(aug[0])
            aug_opt_object.append(float(metrics[0]))
            aug_acc.append(float(metrics[2]))
            aug_micro_f1.append(float(metrics[3]))
            aug_macro_f1.append(float(metrics[4]))
            aug_weighted_f1.append(float(metrics[5]))
            aug_n_dist.append(float(metrics[6]))
        if len(pretrain) > 0:
            metrics = re.compile(r"\d+\.\d+").findall(pretrain[0])
            pre_opt_object.append(float(metrics[0]))
            pre_acc.append(float(metrics[2]))
            pre_micro_f1.append(float(metrics[3]))
            pre_macro_f1.append(float(metrics[4]))
            pre_weighted_f1.append(float(metrics[5]))
            pre_n_dist.append(float(metrics[6]))
        if len(gpu_hours) > 0:
            gpu_hours.append(float(gpu_hours[0]))
    aug_opt_object_final = [np.mean(aug_opt_object), np.std(aug_opt_object)]
    aug_opt_object_final.extend(aug_opt_object)
    aug_acc_final = [np.mean(aug_acc), np.std(aug_acc)]
    aug_acc_final.extend(aug_acc)
    aug_micro_f1_final = [np.mean(aug_micro_f1), np.std(aug_micro_f1)]
    aug_micro_f1_final.extend(aug_micro_f1)
    aug_macro_f1_final = [np.mean(aug_macro_f1), np.std(aug_macro_f1)]
    aug_macro_f1_final.extend(aug_macro_f1)
    aug_weighted_f1_final = [np.mean(aug_weighted_f1), np.std(aug_weighted_f1)]
    aug_weighted_f1_final.extend(aug_weighted_f1)
    aug_n_dist_final = [np.mean(aug_n_dist), np.std(aug_n_dist)]
    aug_n_dist_final.extend(aug_n_dist)

    pre_opt_object_final = [np.mean(pre_opt_object), np.std(pre_opt_object)]
    pre_opt_object_final.extend(pre_opt_object)
    pre_acc_final = [np.mean(pre_acc), np.std(pre_acc)]
    pre_acc_final.extend(pre_acc)
    pre_micro_f1_final = [np.mean(pre_micro_f1), np.std(pre_micro_f1)]
    pre_micro_f1_final.extend(pre_micro_f1)
    pre_macro_f1_final = [np.mean(pre_macro_f1), np.std(pre_macro_f1)]
    pre_macro_f1_final.extend(pre_macro_f1)
    pre_weighted_f1_final = [np.mean(pre_weighted_f1), np.std(pre_weighted_f1)]
    pre_weighted_f1_final.extend(pre_weighted_f1)
    pre_n_dist_final = [np.mean(pre_n_dist), np.std(pre_n_dist)]
    pre_n_dist_final.extend(pre_n_dist)

    gpu_hours_final = [np.mean(gpu_hours), np.std(gpu_hours)]
    gpu_hours_final.extend(gpu_hours)

    result = pd.DataFrame(index=['aug_opt_object', 'aug_acc', 'aug_micro_f1', 'aug_macro_f1', 'aug_weighted_f1',
                                 'aug_n_dist', 'pre_opt_object', 'pre_acc', 'pre_micro_f1', 'pre_macro_f1',
                                 'pre_weighted_f1', 'pre_n_dist', 'gpu_hours'],
                          data=[aug_opt_object_final, aug_acc_final, aug_micro_f1_final, aug_macro_f1_final,
                                aug_weighted_f1_final, aug_n_dist_final, pre_opt_object_final, pre_acc_final,
                                pre_micro_f1_final, pre_macro_f1_final, pre_weighted_f1_final,
                                pre_n_dist_final, gpu_hours_final])
    # result = pd.DataFrame(index=['pre_acc'], data=[pre_acc_final])
    result.to_csv(r"D:/tmp.csv")
