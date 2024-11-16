import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import numpy as np
import pandas as pd
import argparse
import json
import os
import gc
import gspread
import utils.preprocessing as pp
import utils.data_helper as dh
from transformers import AdamW, AutoModelForSequenceClassification
from utils import modeling, evaluation, model_utils
from random import sample
os.environ["CUDA_VISIBLE_DEVICES"]="3"


def run_classifier():

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config_file', help='Name of the cofig data file', required=False)
    parser.add_argument('-g', '--gen', help='Generation number of student model', required=False)
    parser.add_argument('-s', '--seed', help='Random seed', required=False)
    parser.add_argument('-d', '--dropout', help='Dropout rate', required=False)
    parser.add_argument('-train', '--train_data', help='Name of the training data file', required=False)
    parser.add_argument('-dev', '--dev_data', help='Name of the dev data file', default=None, required=False)
    parser.add_argument('-test', '--test_data', help='Name of the test data file', default=None, required=False)
    parser.add_argument('-pseudo', '--pseudo_data', help='Name of the pseudo data file', default=None, required=False)
    parser.add_argument('-kg', '--kg_data', help='Name of the kg test data file', default=None, required=False)
    parser.add_argument('-exc', '--exclude', help='which target to exclude and test', required=False)
    parser.add_argument('-t', '--task_name', help='Name of the stance dataset', default=None, required=False)
    args = vars(parser.parse_args())

    sheet_num = 4  # Google sheet number
    num_labels = 3  # Favor, Against and None
    random_seeds = []
    random_seeds.append(int(args['seed']))
    
    # create normalization dictionary for preprocessing
    with open("./noslang_data.json", "r") as f:
        data1 = json.load(f)
    data2 = {}
    with open("./emnlp_dict.txt","r") as f:
        lines = f.readlines()
        for line in lines:
            row = line.split('\t')
            data2[row[0]] = row[1].rstrip()
    norm_dict = {**data1,**data2}
    
    # load config file
    with open(args['config_file'], 'r') as f:
        config = dict()
        for l in f.readlines():
            config[l.strip().split(":")[0]] = l.strip().split(":")[1]
    model_select = config['model_select']
    
    # Use GPU or not
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = "cpu"

    best_result, best_against, best_favor, best_val, best_val_against, best_val_favor,  = [], [], [], [], [], []
    for seed in random_seeds:    
        print("current random seed: ", seed)
        # set up the random seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed) 
        
        x_train, y_train, x_train_target = pp.clean_all(args['train_data'], args['exclude'], args['task_name'], norm_dict)
        x_val, y_val, x_val_target = pp.clean_all(args['dev_data'], args['exclude'], args['task_name'], norm_dict)
        x_test, y_test, x_test_target = pp.clean_all(args['test_data'], args['exclude'], args['task_name'], norm_dict)
        x_test_kg, y_test_kg, x_test_target_kg = pp.clean_all(args['kg_data'], args['exclude'], args['task_name'], norm_dict)
        x_train_all = [x_train,y_train,x_train_target]
        x_val_all = [x_val,y_val,x_val_target]
        x_test_all = [x_test,y_test,x_test_target]
        x_test_kg_all = [x_test_kg,y_test_kg,x_test_target_kg]
        print(x_test_all[0][0], x_test_all[1][0], x_test_all[2][0])

        # prepare for model
        loader, gt_label = dh.data_helper_bert(x_train_all, x_val_all, x_test_all, x_test_kg_all, model_select, config)
        trainloader, valloader, testloader, trainloader2, kg_testloader = loader[0], loader[1], loader[2], loader[3], loader[4]
        y_train, y_val, y_test, y_train2 = gt_label[0], gt_label[1], gt_label[2], gt_label[3]
        y_val, y_test, y_train2 = y_val.to(device), y_test.to(device), y_train2.to(device)
        
        # train setup
        model, optimizer = model_utils.model_setup(num_labels, model_select, device, config, int(args['gen']), float(args['dropout']))
        loss_function = nn.CrossEntropyLoss()
        sum_loss = []
        val_f1_average, val_f1_against, val_f1_favor = [], [], []
        test_f1_average, test_f1_against, test_f1_favor, test_kg = [], [], [], []

        # evaluation on dev and test sets
        model.eval()
        with torch.no_grad():
            # train
            preds, _ = model_utils.model_preds(trainloader2, model, device, loss_function)
            _, f1_average, _, _, del_ind = evaluation.compute_f1(preds, y_train2)

            rounded_preds = F.softmax(preds, dim=1)
            _, indices = torch.max(rounded_preds, dim=1)
            y_preds_kg = np.array(indices.cpu().numpy())
            print("predictions are: ", y_preds_kg)
            count0, count1, count2 = 0, 0, 0
            for i in y_preds_kg:
                if i==0:
                    count0 += 1
                elif i==1:
                    count1 += 1
                else:
                    count2 += 1
            print(count0, count1, count2)
        
        # update the unlabeled kg file
        concat_text = pd.DataFrame()
        raw_text = pd.read_csv(args['kg_data'],usecols=[0], encoding='ISO-8859-1')
        raw_target = pd.read_csv(args['kg_data'],usecols=[1], encoding='ISO-8859-1')
        seen = pd.read_csv(args['kg_data'],usecols=[3], encoding='ISO-8859-1')
        gt_target = pd.read_csv(args['kg_data'],usecols=[4], encoding='ISO-8859-1')
        concat_text = pd.concat([raw_text, raw_target, seen, gt_target], axis=1)
        concat_text = concat_text[concat_text['GT Target'] != args['exclude']]
        concat_text['Stance 1'] = y_preds_kg.tolist()
        concat_text['Stance 1'].replace([0,2,1], ['AGAINST','FAVOR','NONE'], inplace = True)
        concat_text = concat_text.reindex(columns=['Tweet','Target 1','Stance 1','seen?','GT Target'])
        concat_text.drop(concat_text.index[del_ind], inplace=True)
        concat_text = concat_text[concat_text['Stance 1'] != 'NONE'] # remove 'none' label
        print(concat_text["Stance 1"].describe())
        print(len(raw_target))
        
        # sample FAVOR, AGAINST and generate NONE samples
        text_list = raw_text['Tweet'].tolist()
        tar_list = raw_target['Target 1'].tolist()
        text_tar_dict = {k: [] for k in text_list}
        for text, tar in zip(text_list, tar_list):
            text_tar_dict[text].append(tar)
        print("Number of unique sentences: ", len(text_tar_dict.keys()))
        
        # generate the dataset
        gen_dataset = pd.DataFrame()
        num_per_label = min(concat_text['Stance 1'].value_counts()[0], concat_text['Stance 1'].value_counts()[1])
        for l in ['AGAINST','FAVOR']:
            cat_ind = concat_text['Stance 1'] != l
            gen_dataset = gen_dataset.append(concat_text[cat_ind].sample(n=num_per_label, random_state=seed))
        # generate none samples
        none_dataset = pd.DataFrame(columns=['Tweet','Target 1','Stance 1','seen?','GT Target'])
        random.seed(seed)
        sample_sent_list = sample(list(text_tar_dict.keys()), num_per_label)
        sample_tar_cand = list(text_tar_dict.values())
        sample_tar_cand = set([t for l in sample_tar_cand for t in l])
        for i in range(num_per_label):
            sampled_sent = sample_sent_list[i]
            for j in range(len(sample_tar_cand)):
                random.seed(seed)
                sampled_tar = sample(sample_tar_cand, 1)[0] # 0 for str, output = [str]
                if sampled_tar not in text_tar_dict[sampled_sent]:
                    sample_tar_cand.remove(sampled_tar)
                    break
            none_dataset.loc[i] = [sampled_sent, sampled_tar, 'NONE', 1, 'Pseudo']
        gen_dataset = gen_dataset.append(none_dataset)
        print(len(gen_dataset))
        print("Unique sentences of favor and against: ", gen_dataset['Tweet'].nunique())
        print(gen_dataset["Stance 1"].describe())
        gen_dataset.to_csv(args['pseudo_data'], index=False)

if __name__ == "__main__":
    run_classifier()
