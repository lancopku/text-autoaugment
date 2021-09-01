#!/bin/bash
set -e
export CUDA_VISIBLE_DEVICES=0,1,2,3

for npd in 16
do
  for seed in 59 100 159
  do
    for t in 1 2 3 4 5
    do
      python search.py \
      -c confs/bert_trec.yaml \
      --abspath '/home/renshuhuai/TextAutoAugment' \
      --n-aug $npd \
      --train-npc 20 \
      --valid-npc 10 \
      --test-npc 10 \
      --ir 1 \
      --trail $t \
      --seed $seed \
      --method taa
    done
  done
done

