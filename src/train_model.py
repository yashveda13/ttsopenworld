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
import warnings
import utils.preprocessing as pp
import utils.data_helper as dh
from transformers import AdamW
from utils import modeling, evaluation, model_utils
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support,classification_report

from torch.utils.tensorboard import SummaryWriter   
from pytorchtools import EarlyStopping

warnings.filterwarnings('ignore')
os.environ["CUDA_VISIBLE_DEVICES"]="3"

def compute_performance(preds,y,trainvaltest,step,args,seed):
    print("preds size:",preds.size())
    print("y size:",y.size())
    preds_np = preds.cpu().numpy()
    preds_np = np.argmax(preds_np, axis=1)
    y_train2_np = y.cpu().numpy()
    results_weighted = precision_recall_fscore_support(y_train2_np, preds_np, average='macro')

    print("-------------------------------------------------------------------------------------")
    print(trainvaltest + " classification_report for step: {}".format(step))
    target_names = ['AGAINST','FAVOR','NONE']
    print(classification_report(y_train2_np, preds_np, target_names = target_names, digits = 4))
    ###############################################################################################
    ################            Precision, recall, F1 to csv                     ##################
    ###############################################################################################
    results_twoClass = precision_recall_fscore_support(y_train2_np, preds_np, average=None)
    if args['task_name'] == 'semeval':
        results_weighted = 3 * [(results_twoClass[2][0]+results_twoClass[2][1])/2]
    else:
        results_weighted = precision_recall_fscore_support(y_train2_np, preds_np, average='macro')
    print("results_weighted:",results_weighted)
    result_overall = [results_weighted[0],results_weighted[1],results_weighted[2]]
    result_against = [results_twoClass[0][0],results_twoClass[1][0],results_twoClass[2][0]]
    result_favor = [results_twoClass[0][1],results_twoClass[1][1],results_twoClass[2][1]]
    result_neutral = [results_twoClass[0][2],results_twoClass[1][2],results_twoClass[2][2]]

    print("result_overall:",result_overall)
    print("result_favor:",result_favor)
    print("result_against:",result_against)
    print("result_neutral:",result_neutral)

    result_id = ['train', args['gen'], step, seed, args['dropout'],args['dropoutrest']]
    result_one_sample = result_id + result_against + result_favor + result_neutral + result_overall
    result_one_sample = [result_one_sample]
    print("result_one_sample:",result_one_sample)

    results_df = pd.DataFrame(result_one_sample)    
    print("results_df are:",results_df.head())
    results_df.to_csv('./results_'+trainvaltest+'_df.csv',index=False, mode='a', header=False)    
    print('./results_'+trainvaltest+'_df.csv save, done!')
    print("----------------------------------------------------------------------------")

    return results_weighted[2],result_one_sample

def ensure_results_folder():
    if not os.path.exists('./results'):
        os.makedirs('./results')

def run_classifier():

    ensure_results_folder()
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config_file', default='/content/TTS/TTS_openworld/config/config-bert.txt', 
                       help='Name of the config data file')
    parser.add_argument('-g', '--gen', default='0', help='Generation number of student model')
    parser.add_argument('-s', '--seed', default='42', help='Random seed')
    parser.add_argument('-d', '--dropout', default='0.1', help='Dropout rate')
    parser.add_argument('-d2', '--dropoutrest', default='0.1', help='Dropout rate for rest generations')
    parser.add_argument('-train', '--train_data', 
                       default='/content/TTS/TTS_openworld/data/raw_train_all_subset10_kg_epoch_led_onecol.csv',
                       help='Name of the training data file')
    parser.add_argument('-dev', '--dev_data',
                       default='/content/TTS/TTS_openworld/data/raw_val_all_onecol.csv',
                       help='Name of the dev data file')
    parser.add_argument('-test', '--test_data',
                       default='/content/TTS/TTS_openworld/data/raw_test_all_onecol.csv',
                       help='Name of the test data file')
    parser.add_argument('-kg', '--kg_data',
                       default='/content/TTS/TTS_openworld/data/raw_train_all_subset10_kg_epoch_led_onecol.csv',
                       help='Name of the kg test data file')
    parser.add_argument('-clipgrad', '--clipgradient', type=str, default='True',
                       help='whether clip gradient when over 2')
    parser.add_argument('-n', '--clip_norm', type=int, default=2, help='clip gradient')
    parser.add_argument('-exc', '--exclude', help='which target to exclude and test')
    parser.add_argument('-step', '--savestep', type=int, default=1,
                       help='Save step interval')
    parser.add_argument('-p', '--percent', type=int, default=1,
                       help='Data percentage')
    parser.add_argument('-es_step', '--earlystopping_step', type=int, default=1,
                       help='Early stopping patience')
    parser.add_argument('-z', '--zero_trn', action='store_true')

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
        
    header = pd.DataFrame([10*'#'+args['exclude']+'_seed_'+args['seed']+'_d0_'+str(args['dropout'])+'_d1_'+ \
                           str(args['dropoutrest'])+10*'#'])
    for file in ['results_training','results_validation','results_test','best_results_validation', \
                 'best_results_test','best_loss_results_validation','best_loss_results_test']:
        header.to_csv(file+'_df.csv',index=False, mode='a', header=False)

    for seed in random_seeds:    
        print("current random seed: ", seed)

        log_dir = os.path.join('./tensorboard/tensorboard_train'+str(args['percent'])+'_d0'+str(args['dropout'])+'_d1'+str(args['dropoutrest']+'_seed'+str(seed)+'_gen'+str(args['gen'])), 'train')
        train_writer = SummaryWriter(log_dir=log_dir)

        log_dir = os.path.join('./tensorboard/tensorboard_train'+str(args['percent'])+'_d0'+str(args['dropout'])+'_d1'+str(args['dropoutrest']+'_seed'+str(seed)+'_gen'+str(args['gen'])), 'val')
        val_writer = SummaryWriter(log_dir=log_dir)

        log_dir = os.path.join('./tensorboard/tensorboard_train'+str(args['percent'])+'_d0'+str(args['dropout'])+'_d1'+str(args['dropoutrest']+'_seed'+str(seed)+'_gen'+str(args['gen'])), 'test')
        test_writer = SummaryWriter(log_dir=log_dir)

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
        if int(args['gen']) >= 1:
            print("Current generation is: ", args['gen'])
            remove_indices = []
            if args['zero_trn']:
                for i in range(len(x_test_target_kg)):
                    for j in range(len(x_train_target)):
                        if x_test_target_kg[i] == x_train_target[j] and x_test_kg[i] == x_train[j]:
                            remove_indices.append(i)
                            break
                for t_id in range(len(x_test_kg_all)):
                    x_test_kg_all[t_id] = [i for j, i in enumerate(x_test_kg_all[t_id]) if j not in remove_indices]
            x_train_all = [a+b for a,b in zip(x_train_all, x_test_kg_all)]
        print(x_test_all[0][0], x_test_all[1][0], x_test_all[2][0])

        # prepare for model
        loader, gt_label = dh.data_helper_bert(x_train_all, x_val_all, x_test_all, x_test_kg_all, model_select, config)
        trainloader, valloader, testloader, trainloader2, kg_testloader = loader[0], loader[1], loader[2], loader[3], loader[4]
        y_train, y_val, y_test, y_train2 = gt_label[0], gt_label[1], gt_label[2], gt_label[3]
        y_val, y_test, y_train2 = y_val.to(device), y_test.to(device), y_train2.to(device)
        
        # train setup
        model, optimizer = model_utils.model_setup(num_labels, model_select, device, config, int(args['gen']), float(args['dropout']),float(args['dropoutrest']))
        loss_function = nn.CrossEntropyLoss()
        sum_loss = []
        val_f1_average, test_kg = [], []

        # early stopping
        es_intermediate_step = len(trainloader)//args['savestep']
        patience = args['earlystopping_step']   # the number of iterations that loss does not further decrease
        # patience = es_intermediate_step   # the number of iterations that loss does not further decrease        
        early_stopping = EarlyStopping(patience, verbose=True)
        print(100*"#")
        # print("len(trainloader):",len(trainloader))
        # print("args['savestep']:",args['savestep'])
        print("early stopping occurs when the loss does not decrease after {} steps.".format(patience))
        print(100*"#")
        # print(bk)
        # init best val/test results
        best_train_f1macro, best_val_f1macro, best_test_f1macro = -1, -1, -1
        best_train_result, best_val_result, best_test_result = [], [], []
        best_val_loss, best_test_loss = 100000, 100000
        best_val_loss_result, best_test_loss_result = [], []
        
        # start training
        print(100*"#")
        print("clipgradient:",args['clipgradient']=='True')
        print(100*"#")
        step = 0
        for epoch in range(0, int(config['total_epochs'])):
            print('Epoch:', epoch)
            train_loss = []  
            model.train()
            for b_id, sample_batch in enumerate(trainloader):
                model.train()
                optimizer.zero_grad()
                dict_batch = model_utils.batch_fn(sample_batch)
                inputs = {k: v.to(device) for k, v in dict_batch.items()}
                outputs = model(**inputs)
                loss = loss_function(outputs, inputs['gt_label'])
                loss.backward()

                if args['clipgradient']=='True':
                    nn.utils.clip_grad_norm_(model.parameters(), int(args['clip_norm']))

                optimizer.step()
                step+=1
                train_loss.append(loss.item())

                split_step = len(trainloader)//args['savestep']
                
                if step%split_step==0:
                    model.eval()
                    with torch.no_grad():
                        preds_train, loss_train_inval_mode = model_utils.model_preds(trainloader2, model, device, loss_function)
                        preds_val, loss_val = model_utils.model_preds(valloader, model, device, loss_function)
                        preds_test, loss_test = model_utils.model_preds(testloader, model, device, loss_function)
                        print(100*"#")
                        print("at step: {}".format(step))
                        print("train_loss",train_loss,len(train_loss), sum(train_loss)/len(train_loss))
                        print("loss_val",loss_val,len(loss_val), sum(loss_val) / len(loss_val))
                        print("loss_test",loss_test,len(loss_test), sum(loss_test) / len(loss_test))

                        train_writer.add_scalar('loss', sum(train_loss)/len(train_loss), step)
                        val_writer.add_scalar('loss', sum(loss_val) / len(loss_val), step)
                        test_writer.add_scalar('loss', sum(loss_test) / len(loss_test), step)

                        f1macro_train, result_one_sample_train = compute_performance(preds_train,y_train2,'training',step, args, seed)
                        f1macro_val, result_one_sample_val = compute_performance(preds_val,y_val,'validation',step, args, seed)
                        f1macro_test, result_one_sample_test = compute_performance(preds_test,y_test,'test',step, args, seed)

                        train_writer.add_scalar('f1macro', f1macro_train, step)
                        val_writer.add_scalar('f1macro', f1macro_val, step)
                        test_writer.add_scalar('f1macro', f1macro_test, step)


                        if f1macro_val>best_val_f1macro:
                            best_val_f1macro = f1macro_val
                            best_val_result = result_one_sample_val
                            print(100*"#")
                            print("best f1-macro validation updated at epoch :{}, to: {}".format(epoch, best_val_f1macro))
                            best_test_f1macro = f1macro_test
                            best_test_result = result_one_sample_test
                            print("best f1-macro test updated at epoch :{}, to: {}".format(epoch, best_test_f1macro))
                            print(100*"#")

                        avg_val_loss = sum(loss_val) / len(loss_val)
                        avg_test_loss = sum(loss_test) / len(loss_test)
                        if avg_val_loss<best_val_loss:
                            best_val_loss = avg_val_loss
                            best_val_loss_result = result_one_sample_val
                            print(100*"#")
                            print("best loss validation updated at epoch :{}, to: {}".format(epoch, best_val_loss))
                            best_test_loss = avg_test_loss
                            best_test_loss_result = result_one_sample_test
                            print("best loss test updated at epoch :{}, to: {}".format(epoch, best_test_loss))
                            print(100*"#")

                        val_f1_average.append(f1macro_val)

                        # kg eval
                        preds, loss_kg = model_utils.model_preds(kg_testloader, model, device, loss_function)
                        rounded_preds = F.softmax(preds, dim=1)
                        _, indices = torch.max(rounded_preds, dim=1)
                        y_preds_kg = np.array(indices.cpu().numpy())
                        test_kg.append(y_preds_kg)

                        # early stopping
                        print("loss_val:",loss_val,"average is: ",sum(loss_val) / len(loss_val))
                        early_stopping(sum(loss_val) / len(loss_val), model)
                        if early_stopping.early_stop:
                            print(100*"!")
                            print("Early stopping occurs at step: {}, stop training.".format(step))
                            print(100*"!")
                            break
                    model.train()

            if early_stopping.early_stop:
                print(100*"!")
                print("Early stopping, training ends")
                print(100*"!")
                break

            sum_loss.append(sum(train_loss)/len(train_loss))
            print(sum_loss[epoch])


        #########################################################
        best_val_result[0][0]='best validation'        
        results_df = pd.DataFrame(best_val_result)    
        print("results_df are:",results_df.head())
        results_df.to_csv('./results_validation_df.csv',index=False, mode='a', header=False)    
        print('./results_validation_df.csv save, done!')
        ###
        results_df = pd.DataFrame(best_val_result)    
        print("results_df are:",results_df.head())
        results_df.to_csv('./best_results_validation_df.csv',index=False, mode='a', header=False)    
        print('./best_results_validation_df.csv save, done!')
        ###
        best_val_loss_result[0][0]='best validation' 
        results_df = pd.DataFrame(best_val_loss_result)    
        print("results_df are:",results_df.head())
        results_df.to_csv('./best_loss_results_validation_df.csv',index=False, mode='a', header=False)    
        print('./best_loss_results_validation_df.csv save, done!')
        #########################################################
        best_test_result[0][0]='best test'
        results_df = pd.DataFrame(best_test_result)    
        print("results_df are:",results_df.head())
        results_df.to_csv('./results_test_df.csv',index=False, mode='a', header=False)    
        print('./results_test_df.csv save, done!')
        ###
        results_df = pd.DataFrame(best_test_result)    
        print("results_df are:",results_df.head())
        results_df.to_csv('./best_results_test_df.csv',index=False, mode='a', header=False)    
        print('./best_results_test_df.csv save, done!')
        ###
        best_test_loss_result[0][0]='best test'
        results_df = pd.DataFrame(best_test_loss_result)    
        print("results_df are:",results_df.head())
        results_df.to_csv('./best_loss_results_test_df.csv',index=False, mode='a', header=False)    
        print('./best_loss_results_test_df.csv save, done!')
        #########################################################
        # model that performs best on the dev set is evaluated on the test set
        best_epoch = [index for index,v in enumerate(val_f1_average) if v == max(val_f1_average)][-1]
        
        # update the unlabeled kg file
        if int(args['gen']) < 1:
            concat_text = pd.read_csv(args['kg_data'], encoding='ISO-8859-1')
            concat_text = concat_text[concat_text['GT Target'] != args['exclude']]
            concat_text['Stance 1'] = test_kg[best_epoch].tolist()
            concat_text['Stance 1'].replace([0,1,2], ['AGAINST','FAVOR','NONE'], inplace = True)
            concat_text = concat_text.reindex(columns=['Tweet','Target 1','Stance 1','seen?','GT Target'])
            print(100*"#")
#             concat_text.to_csv(args['kg_data'][:-4]+'_'+args['exclude']+'_seed{}.csv'.format(seed), index=False)
#             print(args['kg_data'][:-4]+'_'+args['exclude']+'_seed{}.csv'.format(seed),"save, done!")
            concat_text.to_csv(args['kg_data'], index=False)
            print(args['kg_data'],"save, done!")
            print(100*"#")
       
        # Save results
        save_result = {
            'best_against': best_against,
            'best_favor': best_favor, 
            'best_result': best_result,
            'best_val_against': best_val_against,
            'best_val_favor': best_val_favor,
            'best_val': best_val
        }

        # Save as CSV
        results_df = pd.DataFrame(save_result)
        results_df.to_csv('./results/best_results.csv', index=False)
        print("Best results saved to ./results/best_results.csv")

        # Save as JSON
        with open('./results/best_results.json', 'w') as f:
            json.dump(save_result, f)
        print("Best results saved to ./results/best_results.json")

if __name__ == "__main__":
    run_classifier()