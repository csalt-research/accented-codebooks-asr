import pandas as pd
import numpy as np
import json
import argparse
import warnings
warnings.filterwarnings("ignore", category=FutureWarning) 

def get_parser():
    parser = argparse.ArgumentParser(prog='create_dataset', description='Script to generate sampled splits from the original dataset')
    parser.add_argument('--input-dir', required=True, help='Path to the directory containing original splits')
    
    return parser

def create_splits(main_dir):
    train = pd.read_csv(f"{main_dir}/train.tsv", sep='\t')

    langs = ['us','england','australia', 'canada', 'scotland']

    print('Train random sample 100 hour')
    new_train = train.sample(frac=0.161, random_state=0)

    for lang in langs:  
        data = new_train[new_train["accent"]==lang]
        print(f"\tAccent:{lang}\t Total number of examples:{data.shape[0]}\t Total Duration: {data['duration'].sum()/3600:.2f} hours")
        
    print(f"Total number of examples:{new_train.shape[0]}\t Total Duration: {new_train['duration'].sum()/3600:.2f} hours")
    print()

    new_train.to_csv(f"{main_dir}/train_random_100h.tsv", index =False, sep = '\t')
    
    print('Train equal sample 100 hour')

    new_train = pd.DataFrame(columns=['client_id', 'path', 'sentence', 'accent','duration'])

    mappings = {'us':0.05,'england':0.18,'australia':0.5, 'canada':0.5, 'scotland':1.0}
    for lang in langs:
        frac = mappings[lang]
        data = train[train["accent"]==lang].sample(frac=frac,random_state=0)
        new_train = new_train.append(data,ignore_index = True)
        print(f"\tAccent:{lang}\t Total number of examples:{data.shape[0]}\t Total Duration: {data['duration'].sum()/3600:.2f} hours")
    
    print(f"Total number of examples:{new_train.shape[0]}\t Total Duration: {new_train['duration'].sum()/3600:.2f} hours")
    

    new_train.to_csv(f"{main_dir}/train_equi_100h.tsv", index =False, sep = '\t')

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    
    create_splits(args.input_dir)

