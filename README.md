# AutoSTG

This is the PyTorch implementation of AutoSTG.

---

## Requirements for Reproducibility

### System Requirements:
- System: Ubuntu 16.04
- Language: Python 3.5
- Devices: a single GeForce GTX 1080 Ti CPU

### Library Requirements:
- numpy == 1.19.1
- pandas == 1.1.1
- torch == 1.1.0
- torchvision == 0.3.0
- tables == 3.6.1
- ruamel.yaml == 0.16.12

---
## Data Preparation
Unzip [dataset/dataset.zip](dataset/dataset.zip) with the following command:
```
cd dataset
unzip ./dataset.zip
```


## Description of Traffic Data

The description please refers to the repository of [DCRNN](https://github.com/liyaguang/DCRNN).

---

## Model Training

[src/train_on_gpu0.sh](src/train_on_gpu0.sh) gives an example to search and train the model on the two datasets:

1. `cd src/`.
2. The settings of the models are in the folder [model](/model), saved as yaml format. 
   - For METR-LA: [METR_LA_AutoSTG.yaml](model/METR_LA_AutoSTG.yaml)
   - For PEMS-BAY: [METR_LA_AutoSTG.yaml](model/PEMS_BAY_AutoSTG.yaml)
3. All trained model will be saved in `param/`. 
4. Searching and training with the given shell script:
   1. `cd src/` .
   2. `bash train_on_gpu0.sh`. The code will firstly load the best epoch from `params/`, and then train the models for `[epoch]`. 

## Model Testing

[src/test_on_gpu0.sh](src/test_on_gpu0.sh) gives an example to test the model on the two datasets using the trained models. Here are the instructions to execute the shell script:
1. `cd src/`
2. `bash test_on_gpu0.sh`.

**Note that:** The given pre-trained models are trained under PyTorch 1.1.0 and can be loaded under other compatible versions (PyTorch 1.5.0 and 1.6.0 with Python 3.7 are tested) with expected behaviors (i.e., the same predicting error).  

---

## License

AutoSTG is released under the MIT License (refer to the LICENSE file for details).
