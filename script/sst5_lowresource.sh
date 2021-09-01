#!/bin/bash
set -e
export CUDA_VISIBLE_DEVICES=0,1,2,3
export LANG=en_US.UTF-8
export LC_ALL=en_US.UTF-8

for seed in 59 100 159
do
  for t in 1 2 3 4 5
  do
    python search.py \
    -c confs/bert_sst5.yaml \
    --abspath '/home/renshuhuai/TextAutoAugment' \
    --n-aug 16 \
    --train-npc 40 \
    --valid-npc 30 \
    --test-npc 30 \
    --ir 1 \
    --trail $t \
    --seed $seed \
    --method taa
  done
done
