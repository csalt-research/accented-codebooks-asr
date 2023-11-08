import os
from collections import defaultdict
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import subprocess as sp
import numpy as np
import argparse
import warnings
warnings.filterwarnings("ignore", category=FutureWarning) 

def get_parser():
    parser = argparse.ArgumentParser(prog='create_dataset', description='Script to generate accented datasplit from the commonvoice dataset')
    parser.add_argument('--input-dir', required=True, help='Give the path to directory containing `data.tsv` file')
    parser.add_argument('--output-dir', required=True, help='The path to the directory to host all generated files.')
    
    return parser

def sanity_check(train, test, dev):
    # check if the speakers are disjoint in train, test and dev

    train_spkrs = train['client_id'].unique()
    test_spkrs = test['client_id'].unique()
    dev_spkrs = dev['client_id'].unique()

    for spkr in train_spkrs:
        if spkr in test_spkrs or spkr in dev_spkrs:
            raise Exception(f"Oopsie!! The speaker {spkr} is found in train and test/dev")
        
    for spkr in dev_spkrs:
        if spkr in train_spkrs or spkr in test_spkrs:
            raise Exception(f"Oopsie!! The speaker {spkr} is found in dev and train/test")
    
    for spkr in test_spkrs:
        if spkr in train_spkrs or spkr in dev_spkrs:
            raise Exception(f"Oopsie!! The speaker {spkr} is found in test and train/dev")

def print_transcript_overlaps(train,test, dev, spks):
    
    no_of_transcripts = [[0]*7 for _ in range(3)]
    durations = [[0]*7 for _ in range(3)]
    no_of_unique_transcripts = [[0]*7 for _ in range(3)]
    
    # only train
    temp = train[train['client_id'].isin(spks['train'])]

    print(f"only train:\n \
        No of transcripts/egs: {temp.shape[0]}\n \
        Total duration of transcripts/egs: {temp['duration'].sum()/3600.0:.2f} hrs\n \
        Average duration of transcripts/egs: {temp['duration'].mean():.2f} scnds\n \
        No of unique transcripts: {temp['sentence'].unique().shape[0]} \n \
        No of speakers: {temp['client_id'].unique().shape[0]} \n \
        Average length of sentences: {temp['sentence'].str.split().apply(len).mean():.2f}")
    no_of_transcripts[0][0] = temp.shape[0]
    no_of_unique_transcripts[0][0] = temp['sentence'].unique().shape[0]
    durations[0][0] = f"{temp['duration'].sum()/3600.0:.2f}"

    # only test 
    temp = test[test["client_id"].isin(spks['test'])]

    print(f"only test:\n \
        No of transcripts/egs: {temp.shape[0]}\n \
        Total duration of transcripts/egs: {temp['duration'].sum()/3600.0:.2f} hrs\n \
        Average duration of transcripts/egs: {temp['duration'].mean():.2f} scnds\n \
        No of unique transcripts: {temp['sentence'].unique().shape[0]} \n \
        No of speakers: {temp['client_id'].unique().shape[0]} \n \
        Average length of sentences: {temp['sentence'].str.split().apply(len).mean():.2f}")
    no_of_transcripts[1][1] = temp.shape[0]
    no_of_unique_transcripts[1][1] = temp['sentence'].unique().shape[0]
    durations[1][1] = f"{temp['duration'].sum()/3600.0:.2f}"

    # only dev  
    temp = dev[dev["client_id"].isin(spks['dev'])]

    print(f"only dev:\n \
        No of transcripts/egs: {temp.shape[0]}\n \
        Total duration of transcripts/egs: {temp['duration'].sum()/3600.0:.2f} hrs\n \
        Average duration of transcripts/egs: {temp['duration'].mean():.2f} scnds\n \
        No of unique transcripts: {temp['sentence'].unique().shape[0]} \n \
        No of speakers: {temp['client_id'].unique().shape[0]} \n \
        Average length of sentences: {temp['sentence'].str.split().apply(len).mean():.2f}")
    no_of_transcripts[2][2] = temp.shape[0]
    no_of_unique_transcripts[2][2] = temp['sentence'].unique().shape[0]
    durations[2][2] = f"{temp['duration'].sum()/3600.0:.2f}"

    # train x dev
    temp = train[train["client_id"].isin(spks['trainxdev'])]
    temp2 = dev[dev["client_id"].isin(spks['trainxdev'])]

    print(f"train x dev:\n \
        No of transcripts/egs: train - {temp.shape[0]}  dev - {temp2.shape[0]}\n \
        Total duration of transcripts/egs: train - {temp['duration'].sum()/3600.0:.2f} hrs  dev - {temp2['duration'].sum()/3600.0:.2f} hrs\n \
        Average duration of transcripts/egs: train - {temp['duration'].mean():.2f} scnds  dev - {temp2['duration'].mean():.2f} scnds\n \
        No of unique transcripts: train - {temp['sentence'].unique().shape[0]}  dev - {temp2['sentence'].unique().shape[0]} \n \
        No of speakers: train - {temp['client_id'].unique().shape[0]}  dev - {temp2['client_id'].unique().shape[0]} \n \
        Average length of sentences: train - {temp['sentence'].str.split().apply(len).mean():.2f}  dev - {temp2['sentence'].str.split().apply(len).mean():.2f}")
    no_of_transcripts[0][4] = temp.shape[0]
    no_of_unique_transcripts[0][4] = temp['sentence'].unique().shape[0]
    durations[0][4] = f"{temp['duration'].sum()/3600.0:.2f}"

    no_of_transcripts[2][4] = temp2.shape[0]
    no_of_unique_transcripts[2][4] = temp2['sentence'].unique().shape[0]
    durations[2][4] = f"{temp2['duration'].sum()/3600.0:.2f}"
    
    # train x test 
    temp = train[train["client_id"].isin(spks['trainxtest'])]
    temp2 = test[test["client_id"].isin(spks['trainxtest'])]

    print(f"train x test:\n \
        No of transcripts/egs: train - {temp.shape[0]}  test - {temp2.shape[0]}\n \
        Total duration of transcripts/egs: train - {temp['duration'].sum()/3600.0:.2f} hrs  test - {temp2['duration'].sum()/3600.0:.2f} hrs\n \
        Average duration of transcripts/egs: train - {temp['duration'].mean():.2f} scnds  test - {temp2['duration'].mean():.2f} scnds\n \
        No of unique transcripts: train - {temp['sentence'].unique().shape[0]}  test - {temp2['sentence'].unique().shape[0]} \n \
        No of speakers: train - {temp['client_id'].unique().shape[0]}  test - {temp2['client_id'].unique().shape[0]} \n \
        Average length of sentences: train - {temp['sentence'].str.split().apply(len).mean():.2f}  test - {temp2['sentence'].str.split().apply(len).mean():.2f}")
    no_of_transcripts[0][3] = temp.shape[0]
    no_of_unique_transcripts[0][3] = temp['sentence'].unique().shape[0]
    durations[0][3] = f"{temp['duration'].sum()/3600.0:.2f}"

    no_of_transcripts[1][3] = temp2.shape[0]
    no_of_unique_transcripts[1][3] = temp2['sentence'].unique().shape[0]
    durations[1][3] = f"{temp2['duration'].sum()/3600.0:.2f}"

    # dev x test
    temp = dev[dev["client_id"].isin(spks['devxtest'])]
    temp2 = test[test["client_id"].isin(spks['devxtest'])]

    print(f"dev x test:\n \
        No of transcripts/egs: dev - {temp.shape[0]}  test - {temp2.shape[0]}\n \
        Total duration of transcripts/egs: dev - {temp['duration'].sum()/3600.0:.2f} hrs  test - {temp2['duration'].sum()/3600.0:.2f} hrs\n \
        Average duration of transcripts/egs: dev - {temp['duration'].mean():.2f} scnds  test - {temp2['duration'].mean():.2f} scnds\n \
        No of unique transcripts: dev - {temp['sentence'].unique().shape[0]}  test - {temp2['sentence'].unique().shape[0]} \n \
        No of speakers: dev - {temp['client_id'].unique().shape[0]}  test - {temp2['client_id'].unique().shape[0]} \n \
        Average length of sentences: dev - {temp['sentence'].str.split().apply(len).mean():.2f}  test - {temp2['sentence'].str.split().apply(len).mean():.2f}")
    no_of_transcripts[2][5] = temp.shape[0]
    no_of_unique_transcripts[2][5] = temp['sentence'].unique().shape[0]
    durations[2][5] = f"{temp['duration'].sum()/3600.0:.2f}"

    no_of_transcripts[1][5] = temp2.shape[0]
    no_of_unique_transcripts[1][5] = temp2['sentence'].unique().shape[0]
    durations[1][5] = f"{temp2['duration'].sum()/3600.0:.2f}"

    # train x dev x test
    temp = train[train["client_id"].isin(spks['trainxtestxdev'])]
    temp2 = dev[dev["client_id"].isin(spks['trainxtestxdev'])]
    temp3 = test[test["client_id"].isin(spks['trainxtestxdev'])]

    print(f"train x dev x test:\n \
        No of transcripts/egs: train - {temp.shape[0]}  dev - {temp2.shape[0]}  test - {temp3.shape[0]}\n \
        Total duration of transcripts/egs: train - {temp['duration'].sum()/3600.0:.2f} hrs  dev - {temp2['duration'].sum()/3600.0:.2f} hrs  test - {temp3['duration'].sum()/3600.0:.2f} hrs\n \
        Average duration of transcripts/egs: train - {temp['duration'].mean():.2f} scnds  dev - {temp2['duration'].mean():.2f} scnds  test - {temp3['duration'].mean():.2f} scnds\n \
        No of unique transcripts: train - {temp['sentence'].unique().shape[0]}  dev - {temp2['sentence'].unique().shape[0]}  test - {temp3['sentence'].unique().shape[0]} \n \
        No of speakers: train - {temp['client_id'].unique().shape[0]}  dev - {temp2['client_id'].unique().shape[0]}  test - {temp3['client_id'].unique().shape[0]} \n \
        Average length of sentences: train - {temp['sentence'].str.split().apply(len).mean():.2f}  dev - {temp2['sentence'].str.split().apply(len).mean():.2f}  test - {temp3['sentence'].str.split().apply(len).mean():.2f} ")
    no_of_transcripts[0][6] = temp.shape[0]
    no_of_unique_transcripts[0][6] = temp['sentence'].unique().shape[0]
    durations[0][6] = f"{temp['duration'].sum()/3600.0:.2f}"

    no_of_transcripts[2][6] = temp2.shape[0]
    no_of_unique_transcripts[2][6] = temp2['sentence'].unique().shape[0]
    durations[2][6] = f"{temp2['duration'].sum()/3600.0:.2f}"
    
    no_of_transcripts[1][6] = temp3.shape[0]
    no_of_unique_transcripts[1][6] = temp3['sentence'].unique().shape[0]
    durations[1][6] = f"{temp3['duration'].sum()/3600.0:.2f}"

def get_splits(data):
    train,test,dev = np.array([]), np.array([]), np.array([])
    speaker_list = {}

    # Get frequency of transcripts
    freq = data["sentence"].value_counts(ascending=True).reset_index()
    freq.columns = ['sentence','freq'] 

    # non unique data
    non_uniq_sent = freq[freq["freq"] >= 3 ]["sentence"].sort_values()
    shuffled_non_uniq_sent = non_uniq_sent.sample(frac=1,random_state = random_seed)
    non_uniq_data = data[data["sentence"].isin(non_uniq_sent)]
    shuffled_non_uniq_data = non_uniq_data.sample(frac=1,random_state=random_seed)

    size = shuffled_non_uniq_sent.shape[0]
    ratios = [
        int(tr_ratio*size),
        int((tr_ratio+tr_dev_ratio)*size),
        int((tr_ratio+tr_dev_ratio+tr_tst_ratio)*size),
        int(tr_rt*size),
        int((tr_rt+dev_ratio)*size),
        int((tr_rt+dev_ratio+dev_tst_ratio)*size)
    ]
    '''
        split details
        1: train
        2: train x dev
        3: train x test
        4: train x test x dev
        5: dev
        6: test x dev
        7: test
    '''
    splits = np.split(shuffled_non_uniq_sent,ratios)

    # train x test x dev 
    spkrs = shuffled_non_uniq_data[shuffled_non_uniq_data["sentence"].isin(splits[3])]["client_id"].unique()
    train_spkrs, test_spkrs, dev_spkrs  = np.split(spkrs,[int(0.6*spkrs.shape[0]),int(0.8*spkrs.shape[0])])
    train = np.concatenate((train,train_spkrs))
    dev = np.concatenate((dev,dev_spkrs))
    test = np.concatenate((test,test_spkrs))
    shuffled_non_uniq_data = shuffled_non_uniq_data[~shuffled_non_uniq_data["client_id"].isin(spkrs)]
    speaker_list['trainxtestxdev'] = spkrs

    # dev x test
    spkrs = shuffled_non_uniq_data[shuffled_non_uniq_data["sentence"].isin(splits[5])]["client_id"].unique()
    dev_spkrs, test_spkrs = np.split(spkrs,[int(0.4*spkrs.shape[0])])
    dev = np.concatenate((dev,dev_spkrs))
    test = np.concatenate((test,test_spkrs))
    shuffled_non_uniq_data = shuffled_non_uniq_data[~shuffled_non_uniq_data["client_id"].isin(spkrs)]
    speaker_list['devxtest'] = spkrs

    # train x test
    spkrs = shuffled_non_uniq_data[shuffled_non_uniq_data["sentence"].isin(splits[2])]["client_id"].unique()
    train_spkrs, test_spkrs = np.split(spkrs,[int(0.8*spkrs.shape[0])])
    train = np.concatenate((train,train_spkrs))
    test = np.concatenate((test,test_spkrs))
    shuffled_non_uniq_data = shuffled_non_uniq_data[~shuffled_non_uniq_data["client_id"].isin(spkrs)]
    speaker_list['trainxtest'] = spkrs

    # train x dev
    spkrs = shuffled_non_uniq_data[shuffled_non_uniq_data["sentence"].isin(splits[1])]["client_id"].unique()
    train_spkrs, dev_spkrs = np.split(spkrs,[int(0.8*spkrs.shape[0])])
    train = np.concatenate((train,train_spkrs))
    dev = np.concatenate((dev,dev_spkrs))
    shuffled_non_uniq_data = shuffled_non_uniq_data[~shuffled_non_uniq_data["client_id"].isin(spkrs)]
    speaker_list['trainxdev'] = spkrs

    # dev
    spkrs = shuffled_non_uniq_data[shuffled_non_uniq_data["sentence"].isin(splits[4])]["client_id"].unique()
    dev = np.concatenate((dev,spkrs))
    shuffled_non_uniq_data = shuffled_non_uniq_data[~shuffled_non_uniq_data["client_id"].isin(spkrs)]
    speaker_list['dev'] = spkrs

    # test
    spkrs = shuffled_non_uniq_data[shuffled_non_uniq_data["sentence"].isin(splits[-1])]["client_id"].unique()
    subset_spkrs = np.random.choice(spkrs,int(spkrs.shape[0]*0.1)) 
    test = np.concatenate((test,subset_spkrs))
    shuffled_non_uniq_data = shuffled_non_uniq_data[~shuffled_non_uniq_data["client_id"].isin(spkrs)]
    speaker_list['test'] = subset_spkrs

    # train
    spkrs = shuffled_non_uniq_data[shuffled_non_uniq_data["sentence"].isin(splits[0])]["client_id"].unique()
    train = np.concatenate((train,spkrs))
    shuffled_non_uniq_data = shuffled_non_uniq_data[~shuffled_non_uniq_data["client_id"].isin(spkrs)]
    speaker_list['train'] = spkrs


    data = data[~data["client_id"].isin(non_uniq_data["client_id"].unique())]

    # approximately unique sentences
    shuffled_uniq_data = data.sample(frac=1,random_state=random_seed)
    speakers_list = shuffled_uniq_data['client_id'].unique()

    tr, tst, dv = np.split(speakers_list,[int(tr_rt*data.shape[0]),int((tr_rt+tst_rt)*data.shape[0])])

    train = np.concatenate((train,tr))
    test = np.concatenate((test,tst))
    dev = np.concatenate((dev,dv))

    speaker_list['train'] = np.concatenate((speaker_list['train'],tr))
    speaker_list['test'] = np.concatenate((speaker_list['test'],tst))
    speaker_list['dev'] = np.concatenate((speaker_list['dev'],dv))

    return train, test, dev, speaker_list

if __name__ == '__main__':
    
    parser = get_parser()
    args = parser.parse_args()
    
    input_dir = args.input_dir
    output_dir = args.output_dir
    train_langs = ['us','england','australia', 'canada', 'scotland', 'indian']
    ignore_langs = []
    
    train_split = 0.8
    dev_split = 0.1
    test_split = 0.1
    random_seed = 100
    np.random.seed(random_seed)

    no_egs_test_per_lang = 10_000
    tr_ratio=0.7375
    tst_ratio=0.0375
    dev_ratio=0.0375
    tr_tst_ratio=0.025
    tr_dev_ratio=0.025
    dev_tst_ratio=0.025
    tr_tst_dev_ratio=0.0125

    tr_rt = tr_ratio+tr_dev_ratio+tr_tst_ratio+tr_tst_dev_ratio
    tst_rt = tst_ratio+tr_tst_ratio+dev_tst_ratio+tr_tst_dev_ratio
    dev_rt = dev_ratio+tr_dev_ratio+dev_tst_ratio+tr_tst_dev_ratio
    
    print(input_dir + 'data.tsv')
    if not os.path.isfile(input_dir + 'data.tsv'):
        raise Exception(f"Oopsie!! The data.tsv file seems to be missing")

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir) 

    
    print("Import validated Data")
    data = pd.read_csv(input_dir + 'data.tsv',sep='\t')

    data = data[['client_id', 'path', 'sentence', 'accent','duration']]
    data = data[data['accent'].notna()]

    print("Dataset division")
    for accent in pd.unique(data['accent']):
        print('\t',end="")
        print(f"{accent}: ({data[data['accent']== accent].shape[0]}, {sum(data[data['accent']== accent]['duration'])/3600.0:.2f})")
    print()

    train_data = pd.DataFrame(columns=['client_id', 'path', 'sentence', 'accent','duration'])
    dev_data = pd.DataFrame(columns=['client_id', 'path', 'sentence', 'accent','duration'])
    test_data = pd.DataFrame(columns=['client_id', 'path', 'sentence', 'accent','duration'])
    speakers_list = {}
    speakers_list_accentwise = {}
    for accent in sorted(pd.unique(data['accent'])):
        if accent in train_langs:
            accented_data  = data[data['accent']==accent]
            no_egs = accented_data.shape[0]
            
            train_spkrs, test_spkrs, dev_spkrs, speaker_list = get_splits(accented_data)

            if accent != 'indian': 
                train_data = train_data.append(accented_data[accented_data['client_id'].isin(train_spkrs)],ignore_index = True)
                dev_data = dev_data.append(accented_data[accented_data['client_id'].isin(dev_spkrs)],ignore_index = True)

            test_data = test_data.append(accented_data[accented_data['client_id'].isin(test_spkrs)],ignore_index = True)

            speakers_list_accentwise[accent] = speaker_list
            for key in speaker_list:
                if key in speakers_list:
                    speakers_list[key] = np.concatenate((speakers_list[key],speaker_list[key]))
                else:
                    speakers_list[key] = speaker_list[key]

        elif accent not in ignore_langs:
            accented_data  = data[data['accent']==accent]
            no_egs = accented_data.shape[0]
            shuffled_data = accented_data.sample(frac=1,random_state=random_seed)
            test_data = test_data.append(accented_data,ignore_index = True)
            speakers_list_accentwise[accent] = {'test':accented_data['client_id'].unique()}
            if 'test' in speakers_list:
                speakers_list['test'] = np.concatenate((speakers_list['test'],accented_data['client_id'].unique()))
            else:
                speakers_list['test'] = accented_data['client_id'].unique()

    test_small_data = test_data.sample(frac=0.15, random_state=random_seed) # test set reduction
    dev_small_data = dev_data.sample(frac=0.25, random_state=random_seed) # test set reduction

    sanity_check(train_data, test_data, dev_data)
    print_transcript_overlaps(train_data,test_data,dev_data,speakers_list)

    print()
    print("Train set distribution")
    for accent in train_data["accent"].unique():
        t = train_data[train_data['accent']==accent]
        print(f"\t{accent}: ({t.shape[0]}, {t['duration'].sum()/3600.0:.2f} hrs)")

    print(f"\tTotal duration: ({train_data.shape[0]}, {train_data['duration'].sum()/3600.0:.2f} hrs)")

    print(f"\n \
        No of transcripts/egs: {train_data.shape[0]}\n \
        Total duration of transcripts/egs: {train_data['duration'].sum()/3600.0:.2f} hrs\n \
        Average duration of transcripts/egs: {train_data['duration'].mean():.2f} scnds\n \
        No of unique transcripts: {train_data['sentence'].unique().shape[0]} \n \
        No of speakers: {train_data['client_id'].unique().shape[0]} \n \
        Average length of sentences: {train_data['sentence'].str.split().apply(len).mean():.2f}")

    print()
    print("Dev set distribution")
    for accent in dev_data["accent"].unique():
        t = dev_data[dev_data['accent']==accent]
        print(f"\t{accent}: ({t.shape[0]}, {t['duration'].sum()/3600.0:.2f} hrs)")

    print(f"\tTotal duration: ({dev_data.shape[0]}, {dev_data['duration'].sum()/3600.0:.2f} hrs)")

    print(f"\n \
        No of transcripts/egs: {dev_data.shape[0]}\n \
        Total duration of transcripts/egs: {dev_data['duration'].sum()/3600.0:.2f} hrs\n \
        Average duration of transcripts/egs: {dev_data['duration'].mean():.2f} scnds\n \
        No of unique transcripts: {dev_data['sentence'].unique().shape[0]} \n \
        No of speakers: {dev_data['client_id'].unique().shape[0]} \n \
        Average length of sentences: {dev_data['sentence'].str.split().apply(len).mean():.2f}")

    print()
    print("Test set distribution")
    for accent in test_data["accent"].unique():
        t = test_data[test_data['accent']==accent]
        print(f"\t{accent}: ({t.shape[0]}, {t['duration'].sum()/3600.0:.2f} hrs)")

    print(f"\tTotal duration: ({test_data.shape[0]}, {test_data['duration'].sum()/3600.0:.2f} hrs)")

    print(f"\n \
        No of transcripts/egs: {test_data.shape[0]}\n \
        Total duration of transcripts/egs: {test_data['duration'].sum()/3600.0:.2f} hrs\n \
        Average duration of transcripts/egs: {test_data['duration'].mean():.2f} scnds\n \
        No of unique transcripts: {test_data['sentence'].unique().shape[0]} \n \
        No of speakers: {test_data['client_id'].unique().shape[0]} \n \
        Average length of sentences: {test_data['sentence'].str.split().apply(len).mean():.2f}")
    
    print()
    print("Dev small set distribution")
    for accent in dev_small_data["accent"].unique():
        t = dev_small_data[dev_small_data['accent']==accent]
        print(f"\t{accent}: ({t.shape[0]}, {t['duration'].sum()/3600.0:.2f} hrs)")

    print(f"\tTotal duration: ({dev_small_data.shape[0]}, {dev_small_data['duration'].sum()/3600.0:.2f} hrs)")

    print(f"\n \
        No of transcripts/egs: {dev_small_data.shape[0]}\n \
        Total duration of transcripts/egs: {dev_small_data['duration'].sum()/3600.0:.2f} hrs\n \
        Average duration of transcripts/egs: {dev_small_data['duration'].mean():.2f} scnds\n \
        No of unique transcripts: {dev_small_data['sentence'].unique().shape[0]} \n \
        No of speakers: {dev_small_data['client_id'].unique().shape[0]} \n \
        Average length of sentences: {dev_small_data['sentence'].str.split().apply(len).mean():.2f}")
    
    print()
    print("Test small set distribution")
    for accent in test_small_data["accent"].unique():
        t = test_small_data[test_small_data['accent']==accent]
        print(f"\t{accent}: ({t.shape[0]}, {t['duration'].sum()/3600.0:.2f} hrs)")

    print(f"\tTotal duration: ({test_small_data.shape[0]}, {test_small_data['duration'].sum()/3600.0:.2f} hrs)")

    print(f"\n \
        No of transcripts/egs: {test_small_data.shape[0]}\n \
        Total duration of transcripts/egs: {test_small_data['duration'].sum()/3600.0:.2f} hrs\n \
        Average duration of transcripts/egs: {test_small_data['duration'].mean():.2f} scnds\n \
        No of unique transcripts: {test_small_data['sentence'].unique().shape[0]} \n \
        No of speakers: {test_small_data['client_id'].unique().shape[0]} \n \
        Average length of sentences: {test_small_data['sentence'].str.split().apply(len).mean():.2f}")

    train_data.to_csv(output_dir+'train.tsv',sep='\t', index=False)
    test_data.to_csv(output_dir+'test.tsv',sep='\t', index=False)
    dev_data.to_csv(output_dir+'dev.tsv',sep='\t', index=False)
    
    test_small_data.to_csv(output_dir+'test_small.tsv',sep='\t', index=False)
    dev_small_data.to_csv(output_dir+'dev_small.tsv',sep='\t', index=False)