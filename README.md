# Medical Temporal pattern detection 

This repository contains the code related to the paper `Temporal deep learning framework for retinopathy prediction in type 1 diabetes patients`

## Table of contents
* [General info](#general-info)
* [Technologies](#technologies)
* [Implemented Models](#implemented-models)
* [Model configuration](#model-configuration)
* [Run Hyper-parameters search](#run-hyper-parameters-fine-tuning)
* [Run Training](#run-training)
* [Run Testing](#run-testing)


## General info
- This github project includes deep learning architectures that aim to learn temporal 
patterns from medical time series in order to conduct binary predictions. 

- The application use-case studied in the paper is `Retinopathy {rediction`:  the outcome of whether a type 1 diabetes patient will develop a retinopathy complication or not after his visit. This problem is posed as a binary classification one, and true retinopathy labels were created by checking the retinopathy status at the following visit. 


- The problem is formulated as finding the best DL model that correctly classifies the presence of retinopathy or not
 and gives in addition the attention weights and patient's hidden representation that explain such prediction.

## Technologies
This project is implemented using python3 and is based on the following packages:
* PyTorch
* fast.ai
* sklearn
* numpy 

## Installation guidelines: 
- Clone the repo: 
```
git clone https://github.com/sararb/Temporal-Deep-Learning-for-Medical-time-series.git
cd Temporal-Deep-Learning-for-Medical-time-series
```
- Create a conda environment
```
conda create --name temporal_model python=3.7
conda activate temporal_model
```
- Install required packages
```
pip install -r requirements.txt
```


## Implemented Models
* BILSTM_attention: Bi-directional LSTM model with attention mechanism 
* <a href="http://www.sciencedirect.com/science/article/pii/S1532046417300710" target="_blank"> CLSTM_Net</a>: Contextual LSTM model that formulates the forget gate as a function of timedelta between timestamps
* <a href="http://arxiv.org/abs/1911.12864" target="_blank">AttnModel</a>: Self-attention mechanism that formulates positional embedding as a function of time. 

## Model configuration

- For each model, a yaml file is defined at "config_files" directory and it contains all the parameters needed for
 defining the architecture and the training experiment. 

- The config directory is split into two sections:  
    > `hyperparam_finetune` including configs related to hyper-parameter optimization experiements.  
    > `final_best_models` including configs related to the best model resulting for the hyperparameter search. 
 
- The config file is organized in three main sections: 
    1. Paths to data and results directories
    2. Training parameters
    3. Model's parameters 
    
## Data preparation 
- The input to the models are the sequence of HbA1c measurements and the time delta between two consecutive measurements. 
- The model handles normalized values. 
- To define and test the models, we used a cross-validation process where the raw data was divided to
 Train (80%), Validation (10%) and Test (10%) sets.
- At each batch iteration the input is a matrix of patients' sequences with same length. 

To generate the patients dictionaries to use for the study, you need to modify `data_path` and `out_dir` in `./data/data.py` script to specify the path to the input table to use and the output directory where to store the results. Then launch the process via: 
```
python data.py
```
## Run Hyper-parameters fine tuning

The script processes the YAML file containing the search spaces of the hyper-parameters and launches Bayesian optimization trials using the Optuna package \cite{optuna_2019}. It is also linked to the ``Weight And Biases" platform \cite{wandb} to track the evolution of the performance metrics.

- The command line for launching the hyper-parameter fine-tuning is:
```
CUDA_VISIBLE_DEVICES=0  python run_hyperparam.py --config config_files/hyperparam_finetune/LSTM/no_time.yaml --trunc_max_len 151 --side_info_mode seq_concat  --min_measurement 3
```

## Run training
The script processes the YAML file detailing the parameters and instantiates the related model architecture. Then, it runs the training experiments, logs the training progression, and finally saves the results of test data on disk. The output directory also includes the model's checkpoints and training history.

The command line for launching the  training experiment is:
```
CUDA_VISIBLE_DEVICES=0  python train_model.py --config config_files/final_best_models/LSTM/15 --bag_number 1 --trunc_max_len 151  --kfolds 1 --test 
```

Where : 
* --config: Specify the path to the yaml configuration file.
* --model_type: The model's architecture to use. 
* --do_test: if specified, the predictions and their attentions weights are computed after training is finished.


## Run Testing
If we have already trained the model and we would like to test it on test data, the command line is: 

```
python do_test.py --config $PATH_TO_CONFIG  --model_type $MODEL --save_dir $PATH_TO_SAVE_RESULTS --checkpoint $PATH_TO_SAVED_MODEL --trunc_max_len $MAX_SEQ_LEN
```

An example of this command in our experiments is : 
```
python do_test.py --config ./config_files/final_best_models/ATTENTION/3/time_concat_soft.yaml --model_type attention --save_dir ~/these_repo/ --checkpoint /home/rabhi/dataset/temporal_hba1c/results/attention/algorithm_time_concat_soft_min_meas_3_max_meas_151_side_features_duree_non_suivi_norm_/6/1/checkpoints/attention_auc_score_0.863.pth --trunc_max_len 151
```

Where: 
* --config: Specify the path to the yaml configuration file.
* --model_type: The model's architecture to use : "LSTM_CNN", "BI_LSTM", "C_LSTM", "SelfAttn".
* --checkpoint: path to the trained model's weights (.pth file)
* --savedir:  path to directory where to save test results 
* --trunc_max_len: the maximum length for sequence padding
