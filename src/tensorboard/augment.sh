#!/usr/bin/env bash


train_data=../data/raw_train_all_subset_kg_epoch_led_onecol.csv

echo "start augmenting ......"
python ./utils/augment.py -train ${train_data}