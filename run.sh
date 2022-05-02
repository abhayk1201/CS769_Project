#!/usr/bin/env bash

python run_multi_task.py \
  --seed 42 \
  --output_dir ./Tmp_Model/MTL \
  --tasks all \
  --sample 'anneal'\
  --multi \
  --do_train \
  --do_eval \
  --do_lower_case \
  --data_dir ./data/ \
  --vocab_file ./uncased_L-12_H-768_A-12/vocab.txt \
  --bert_config_file ./config/pals_config.json \
  --init_checkpoint ./uncased_L-12_H-768_A-12/pytorch_model.bin \
  --max_seq_length 50 \
  --train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 10 \
  --gradient_accumulation_steps 1



  python run_multi_task.py \
  --seed 42 \
  --output_dir ./Tmp_Model/MTL_sqrt \
  --tasks all \
  --sample 'sqrt'\
  --multi \
  --do_train \
  --do_eval \
  --do_lower_case \
  --data_dir ./data/ \
  --vocab_file ./uncased_L-12_H-768_A-12/vocab.txt \
  --bert_config_file ./config/pals_config.json \
  --init_checkpoint ./uncased_L-12_H-768_A-12/pytorch_model.bin \
  --max_seq_length 50 \
  --train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 10 \
  --gradient_accumulation_steps 1


  python run_multi_task.py \
  --seed 42 \
  --output_dir ./Tmp_Model/MTL_prop \
  --tasks all \
  --sample 'prop'\
  --multi \
  --do_train \
  --do_eval \
  --do_lower_case \
  --data_dir ./data/ \
  --vocab_file ./uncased_L-12_H-768_A-12/vocab.txt \
  --bert_config_file ./config/pals_config.json \
  --init_checkpoint ./uncased_L-12_H-768_A-12/pytorch_model.bin \
  --max_seq_length 50 \
  --train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 10 \
  --gradient_accumulation_steps 1


  python run_multi_task.py \
  --seed 42 \
  --output_dir ./Tmp_Model/goemotion \
  --tasks single \
  --task_id 1 \
  --sample 'rr'\
  --multi \
  --do_train \
  --do_eval \
  --do_lower_case \
  --data_dir ./data/ \
  --vocab_file ./uncased_L-12_H-768_A-12/vocab.txt \
  --bert_config_file ./config/pals_config.json \
  --init_checkpoint ./uncased_L-12_H-768_A-12/pytorch_model.bin \
  --max_seq_length 50 \
  --train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 10 \
  --gradient_accumulation_steps 1


  python run_multi_task.py \
  --seed 42 \
  --output_dir ./Tmp_Model/sst \
  --tasks single \
  --task_id 0 \
  --sample 'rr'\
  --multi \
  --do_train \
  --do_eval \
  --do_lower_case \
  --data_dir ./data/ \
  --vocab_file ./uncased_L-12_H-768_A-12/vocab.txt \
  --bert_config_file ./config/pals_config.json \
  --init_checkpoint ./uncased_L-12_H-768_A-12/pytorch_model.bin \
  --max_seq_length 50 \
  --train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 10 \
  --gradient_accumulation_steps 1

