import re
import wordninja
import csv
import pandas as pd
from utils import augment

# Data Loading
def load_data(filename, file_exc, task_name):
    concat_text = pd.DataFrame()
    raw_text = pd.read_csv(filename, usecols=[0], encoding='ISO-8859-1')
    raw_target = pd.read_csv(filename, usecols=[1], encoding='ISO-8859-1')
    raw_label = pd.read_csv(filename, usecols=[2], encoding='ISO-8859-1')
    seen = pd.read_csv(filename, usecols=[3], encoding='ISO-8859-1')
    gt_target = pd.read_csv(filename, usecols=[4], encoding='ISO-8859-1')
    label = pd.DataFrame.replace(raw_label, ['AGAINST','FAVOR','NONE'], [0,1,2])
    concat_text = pd.concat([raw_text, label, raw_target, seen, gt_target], axis=1)
    concat_text.rename(columns={'Stance 1':'Stance','Target 1':'Target'}, inplace=True)
    
    if task_name == 'vast':
        if 'train' not in filename:
            concat_text = concat_text[concat_text['seen?'] != 1]  
    else:
        if 'train' not in filename:
            concat_text = concat_text[concat_text['seen?'] != 1]      
            concat_text = concat_text[concat_text['GT Target'] == file_exc]
        else:
            concat_text = concat_text[concat_text['GT Target'] != file_exc]
        
    return concat_text

# Data Cleaning
def data_clean(strings, norm_dict):
    # Remove URLs
    clean_data = re.sub(r'http\S+|www.\S+', '', strings)
    
    # Remove emojis and special characters
    clean_data = re.sub(r'[^\w\s#@,\.!?&/\<>=$]', '', clean_data)
    
    # Remove #SemST
    clean_data = re.sub(r"#SemST", "", clean_data)
    
    # Split into tokens
    clean_data = re.findall(r"[A-Za-z#@]+|[,.!?&/\<>=$]|[0-9]+", clean_data)
    clean_data = [[x.lower()] for x in clean_data]
    
    # Handle normalization and hashtags
    for i in range(len(clean_data)):
        if clean_data[i][0] in norm_dict.keys():
            clean_data[i] = norm_dict[clean_data[i][0]].split()
            continue
        if clean_data[i][0].startswith("#") or clean_data[i][0].startswith("@"):
            clean_data[i] = wordninja.split(clean_data[i][0])
    
    clean_data = [j for i in clean_data for j in i]
    return clean_data

# Clean All Data
def clean_all(filename, file_exc, task_name, norm_dict):
    """
    Clean and preprocess data from input file
    
    Args:
        filename: path to input CSV file
        file_exc: target to exclude
        task_name: name of the task (e.g. 'semeval')
        norm_dict: dictionary for text normalization
    
    Returns:
        clean_data: list of cleaned text
        label: list of labels
        x_target: list of targets
    """
    try:
        # Load data
        data = pd.read_csv(filename, encoding='ISO-8859-1')
        if file_exc != 'none':
            data = data[data['GT Target'] != file_exc]
            
        if len(data) == 0:
            print(f"Warning: No data found for {filename} with exclude={file_exc}")
            return [], [], []
            
        # Extract text, targets and labels
        clean_data = data['Tweet'].tolist()
        x_target = data['Target 1'].tolist()
        label = data['Stance 1'].tolist()
        
        # Convert stance labels to integers
        label = ['AGAINST' if x=='AGAINST' else 'FAVOR' if x=='FAVOR' else 'NONE' for x in label]
        label = [0 if x=='AGAINST' else 1 if x=='FAVOR' else 2 for x in label]
        
        # Calculate and print statistics
        if len(clean_data) > 0:
            avg_len = sum(len(x.split()) for x in clean_data) / len(clean_data)
            print("average length: ", avg_len)
            print("num of subset: ", len(clean_data))
            
        return clean_data, label, x_target
        
    except Exception as e:
        print(f"Error processing file {filename}: {str(e)}")
        return [], [], []