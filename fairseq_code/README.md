# Hubert Pre-training Instructions

The repository contains the modified version of Hubert pre-training implementation from Fairseq.

## Prerequisites and Installation


1. Clone the repository containing our code and dataset:
    ```sh
    git clone https://github.com/csalt-research/accented-codebooks-asr/tree/accented-pretraining
    ```

2. Move to the pre-training directory:
    ```sh
    cd pretraining
    ```

3. Install all the requirements of Fairseq:
    ```sh
    pip install -e .
    ```

## Training
Hubert pre-training have be done in two iterations. For both iterations we have initialised Hubert from the Librispeech checkpoint [link](https://github.com/facebookresearch/fairseq/tree/main/examples/hubert).
1. **Data Prepration**: Create the audio CSV in the same format as mentioned [here](https://github.com/facebookresearch/fairseq/tree/main/examples/hubert/simple_kmeans). A sample script is provided [here](https://github.com/csalt-research/accented-codebooks-asr/blob/accented-pretraining/fairseq_code/scripts/accented_split.py)
   
2. **Extract Features**
   
**Iteration 1**: Extract features from the 6th layer of the Hubert model pre-trained on LibriSpeech following the instruction [here](https://github.com/facebookresearch/fairseq/tree/main/examples/hubert/simple_kmeans)

**Iteration 2**: Now use the Hubert model from Iteration 1 to extract the the features for 2nd iteration

3. **Fit K-Means Model and Get Labels from K-Means Model**
   
For both iterations fit a K-means model with 500 clusters on 10% of the train split and get the labels of the train and validation set following the instructions [here](https://github.com/facebookresearch/fairseq/tree/main/examples/hubert/simple_kmeans).
Also, create a dummy dictionary of the same number of clusters.

3.  **Model Configuration**
To run different configurations, modify the `hubert_accent.yaml` file located at `fairseq-local/examples/hubert/config/pretrain`.

 Number of Accents and Codebook Entries
    
    no_accents: 5
    no_codebook_entries: 50
    

Add the layer indices (zero-indexed) in `codebook_layers` on which codebook cross-attention required:
    
    codebook_layers: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    

To freeze the codebook entries 
    
    freeze_codebooks: true
    
4. **Start the training**
```
python fairseq_cli/hydra_train.py --config-dir fairseq/examples/hubert/config/pretrain --config-name hubert_accent task.data=<path-of-data-csv> task.label_dir=<path-of-extracted-labels> task.labels='["km"]' model.label_rate=50
```

