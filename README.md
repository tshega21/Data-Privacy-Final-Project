# Personalized federated learning with New Sampling Method in privacy-preserving context

This code is adapted from Karhoutam - https://github.com/KarhouTam/Per-FedAvg


## Per-FedAVG-BASE
    - personalized federated learning
## Per-FedAvg- DP
    - personalized federated learning with differential privacy
## Per-FedAvg -DP-FINAL
    - personalized federated learning with differential privacy and sampling method

## Instructions: For each of these folders, execute the following within the directory. 

### Requirements
```
torch~=1.10.2
path~=16.4.0
numpy~=1.21.2
fedlab~=1.1.4
torchvision~=0.11.3
rich~=12.2.0
```
###
```
pip install -r requirements.txt
```
then, please proceed with:

```
pip uninstall pynvml
```
to get rid of dependencies.

### Preprocess dataset

```
cd data; python preprocess.py
```
###

```
python main.py
```

### Hyperparameters are defined in utils.py 

`--global_epochs`: Num of communication rounds. Default: `150`

`--local_epochs`: Num of local training rounds. Default: `4`

`--pers_epochs`: Num of personalization rounds (while in evaluation phase). Default: `1`

`--dataset`: Name of experiment dataset. Default: `mnist`

`--fraction`: Percentage of training clients in all alients. Default: `0.9`

`--client_num_per_round`: Num of clients that participating training at each communication round. Default: `5`

`--alpha`: Learning rate $\alpha$ in paper. Default: `0.01`

`--beta`: Learning rate $\beta$ in paper. Default: `0.001`

`--gpu`: Non-zero value for using CUDA; `0` for using CPU. Default: `1`

`--batch_size`: Batch size of client local dataset. Default: `40`.

`--eval_while_training`: Non-zero value for performing evaluation while in training phase. Default: `1`

`--valset_ratio`: Percentage of validation set in client local dataset. Default: `0.1`

`--hf`: Non-zero value for performing Per-FedAvg(HF); `0` for Per-FedAvg(FO). Default: `1`

`--seed`: Random seed for init model parameters and selected clients. Default: `17`

`--log`: Non-zero value for recording experiment's output to a `.html` file in `./log`. Default: `0`

### To reproduce Data
Please go into utils.py in the data folder in order to set the deafult seed to whatever you may need it to be.
