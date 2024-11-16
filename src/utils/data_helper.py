import torch
import transformers
from torch.utils.data import TensorDataset, DataLoader
from transformers import BertTokenizer, AutoTokenizer, BertweetTokenizer, BartTokenizer
transformers.logging.set_verbosity_error()
    

# Tokenization
def convert_data_to_ids(tokenizer, target, text, label, config):
    
    concat_sent = []
    for tar, sent in zip(target, text):
        concat_sent.append([' '.join(sent), 'I am in favor of '+' '.join(tar)+" !"])
    encoded_dict = tokenizer.batch_encode_plus(
                    concat_sent,
                    add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                    max_length = int(config['max_tok_len']), # Pad & truncate all sentences.
                    padding = 'max_length',
                    return_attention_mask = True,   # Construct attn. masks.
                    truncation = True,
               )
    encoded_dict['gt_label'] = label
    
    return encoded_dict


# BERT/BERTweet tokenizer    
def data_helper_bert(x_train_all, x_val_all, x_test_all, x_test_kg_all, model_select, config):
    print('Loading data')
    
    x_train, y_train, x_train_target = x_train_all[0], x_train_all[1], x_train_all[2]
    x_val, y_val, x_val_target = x_val_all[0], x_val_all[1], x_val_all[2]
    x_test, y_test, x_test_target = x_test_all[0], x_test_all[1], x_test_all[2]
    x_test_kg, y_test_kg, x_test_target_kg = x_test_kg_all[0], x_test_kg_all[1], x_test_kg_all[2]
    
    print("Length of original x_train: %d"%(len(x_train)))
    print("Length of original x_val: %d"%(len(x_val)))
    print("Length of original x_test: %d"%(len(x_test)))
    
    # get the tokenizer
    if model_select == 'Bertweet':
        tokenizer = BertweetTokenizer.from_pretrained("vinai/bertweet-base", normalization=True)
    elif model_select == 'Bart' or model_select == 'Bart-MNLI':
        tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-mnli", normalization=True)
    elif model_select == 'Bert':
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)

    # Create empty tensors for empty datasets
    empty_loader = DataLoader(TensorDataset(
        torch.tensor([], dtype=torch.long),
        torch.tensor([], dtype=torch.long),
        torch.tensor([], dtype=torch.long)
    ), batch_size=int(config['batch_size']))
    empty_labels = torch.tensor([], dtype=torch.long)

    # tokenization and loader creation
    train_encoded_dict = convert_data_to_ids(tokenizer, x_train_target, x_train, y_train, config)
    trainloader, y_train = data_loader(train_encoded_dict, int(config['batch_size']), model_select, 'train')
    trainloader2, y_train2 = data_loader(train_encoded_dict, int(config['batch_size']), model_select, 'train2')

    # Handle validation data
    if len(x_val) > 0:
        val_encoded_dict = convert_data_to_ids(tokenizer, x_val_target, x_val, y_val, config)
        valloader, y_val = data_loader(val_encoded_dict, int(config['batch_size']), model_select, 'val')
    else:
        valloader, y_val = empty_loader, empty_labels

    # Handle test data
    if len(x_test) > 0:
        test_encoded_dict = convert_data_to_ids(tokenizer, x_test_target, x_test, y_test, config)
        testloader, y_test = data_loader(test_encoded_dict, int(config['batch_size']), model_select, 'test')
    else:
        testloader, y_test = empty_loader, empty_labels

    # Handle KG test data
    if len(x_test_kg) > 0:
        test_kg_encoded_dict = convert_data_to_ids(tokenizer, x_test_target_kg, x_test_kg, y_test_kg, config)
        kg_testloader, _ = data_loader(test_kg_encoded_dict, int(config['batch_size']), model_select, 'kg')
    else:
        kg_testloader, _ = empty_loader, empty_labels
    
    print("Length of final x_train: %d"%(len(y_train)))
    
    return (trainloader, valloader, testloader, trainloader2, kg_testloader), (y_train, y_val, y_test, y_train2)


def data_loader(x_all, batch_size, model_select, mode):
    
    x_input_ids = torch.tensor(x_all['input_ids'], dtype=torch.long)
    x_atten_masks = torch.tensor(x_all['attention_mask'], dtype=torch.long)
    y = torch.tensor(x_all['gt_label'], dtype=torch.long)
    if model_select == 'Bert':
        x_seg_ids = torch.tensor(x_all['token_type_ids'], dtype=torch.long)
        tensor_loader = TensorDataset(x_input_ids, x_atten_masks, x_seg_ids, y)
    else:
        tensor_loader = TensorDataset(x_input_ids, x_atten_masks, y)

    if mode == 'train':
        data_loader = DataLoader(tensor_loader, shuffle=True, batch_size=batch_size)
    else:
        data_loader = DataLoader(tensor_loader, shuffle=False, batch_size=batch_size)

    return data_loader, y
