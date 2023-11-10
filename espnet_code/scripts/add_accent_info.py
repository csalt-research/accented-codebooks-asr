import os 
import csv
import json
import argparse

acc2index = {
    'australia': 0,
    'canada': 1,
    'england': 2,
    'scotland': 3,
    'us': 4,
}

def get_parser():
    
    parser = argparse.ArgumentParser(prog='add_accent', description='Script to add accent labels to the json files created by espnet')
    parser.add_argument('--input-csv', required=True, help='Path to the csv used for creating the corresponding json file')
    parser.add_argument('--input-json', required=True, help='Path to the json file to which accent labels are to be added')
    parser.add_argument('--output-json', required=True, help='Path where the new json files should be saved')
    
    return parser


def add_accent_info(src,trgt):
    d = {}
    for row in src:
        d[row[0]] = row[-2]
    
    cnt = {0:0,1:0,2:0,3:0,4:0}
    print(len(list(trgt["utts"])))
    for key in list(trgt["utts"]):
        key1 = key
        # Handle speed perturbation
        key = key.replace("sp0.9-","").replace("sp1.0-","").replace("sp1.1-","")
        try:
            trgt["utts"][key1]["accent"] = acc2index[d[key.split("-")[0]]]
            trgt["utts"][key1]["input"][0]["feat"] = trgt["utts"][key1]["input"][0]["feat"].replace('data3','data2')
            cnt[acc2index[d[key.split("-")[0]]]] += 1
        except:
            del trgt["utts"][key1]
        
    print(len(list(trgt["utts"])))
    print(f"{cnt} examples added")
    return trgt


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    
    trgt_file = args.input_json
    output_file = args.output_json
    
    src_file = args.input_csv
    
    with open(src_file,'r') as f:
        reader = csv.reader(f,delimiter='\t')
        src_data = [line for line in reader]
    
    with open(trgt_file,'r') as f:
        trgt_data = json.load(f)
    trgt_data = add_accent_info(src_data,trgt_data)

    # write json file
    json_object = json.dumps(trgt_data, indent = 4)

    with open(output_file, "w") as outfile:
        outfile.write(json_object)