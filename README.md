# StorageDeivceModel

This repo contains the codes of the exp7 in the 09-02 weekly report. 

1. Files:
    
    dataset_7: contains the train and test dataset.
    
    model: the storageNet
    
    TRACEPROFILE: trace after replay (for the generation of the true LDS)
    
    TRACETOREPLAY: trace to replay (for the generation of the prev LDS (input))
    
    weights: save the weights of the model
    
    eval.py: run the multi-step prediction
    
    parser & parser_script.sh: generate the ‘dataset_7’ dataset from TRACEPROFILE & TRACETOREPLAY
    
    train.py train storageNet model
    
2. Generate the dataset:
    
    ```bash
    ./parser_script.sh
    ```
    
3. train: (around 10 min on my local machine)
    
    ```bash
    python3 train.py
    ```
    
4. multi-step evaluation:
    
    ```bash
    python eval.py
    ```