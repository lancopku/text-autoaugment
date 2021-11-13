#!/bin/bash
set -e
export CUDA_VISIBLE_DEVICES=0,1,2,3

for ir in 0.02 0.05 0.1
do
  for seed in 59 100 159
  do
    for t in 1 2 3 4 5
    do
      python examples/reproduce_experiment.py \
      -c taa/confs/bert_sst2_example.yaml \
      --abspath '/home/renshuhuai/text-autoaugment' \
      --n-aug 1 \
      --num-op 2 \
      --num-policy 4 \
      --train-npc 1000 \
      --valid-npc 30 \
      --test-npc 30 \
      --ir $ir \
      --trail $t \
      --seed $seed \
      --method taa
    done
  done
done
