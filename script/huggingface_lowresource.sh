#!/bin/bash
set -e
export CUDA_VISIBLE_DEVICES=2,3,4,5
export LANG=en_US.UTF-8
export LC_ALL=en_US.UTF-8

for seed in 59 100 159
do
  for t in 1 2 3 4 5
  do
    python examples/reproduce_experiment.py \
    -c taa/confs/bert_imdb_example.yaml \
    --abspath '/home/renshuhuai/text-autoaugment' \
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
