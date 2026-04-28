# Reti-Pioneer

![img](figures/logo.png)

[AI Framework for Multidisease Detection via Retinal Imaging](https://www.nature.com/articles/s41591-026-04359-w)

For more information, please visit our [website](https://www.retipioneer.cn).

## News

- 2026/03: version 1.0 released

## Hardware and software requirements

- A consumer-grade GPU (~6GB) is recommended for model training and testing.
- The code is tested on Ubuntu 20 and Windows 11.
- Python 3.12 is recommended.

## Data

- [UK Biobank](https://www.ukbiobank.ac.uk/) and tertiary hospital centres in China are used for training and fine-tuning in this study.
- We recommend preprocessing the data to accelerate the training process in the following ways (N, M and D denote the numbuer of samples, clinical variables and diseases):
    - Use RETFound to extract the deep features and save it to `UKB_RETF.npz` with format
        - "left": N x 1024 numpy array
        - "right": N x 1024 numpy array
    - Save the clinical variables, quality scores, and center ID to `UKB_mqd.npz` with format
        - "m": N x M clinical variables
        - "mn": N clinical variable names
        - "ql": N x 3 left quality scores
        - "qr": N x 3 right quality scores
        - "center": N center IDs
    - Save diseases information to `UKB_y{n}.npz` with format
        - "y": N x D numpy array
- Place preprocessed data (`UKB_RETF.npz`, `UKB_mqd.npz` and `UKB_y{n}.npz`) on `data/UKBCompressed` or specify the path in code.
- A small (dummy) example is provide in `data/UKBCompressed`.

## Installation Guide

- Clone this project.
- Use uv to create the virtual environment and install the dependencies.
```
git clone https://github.com/lyhyl/Reti-Pioneer.git
cd Reti-Pioneer
uv sync
.venv\bin\activate
```
Installation will take a few minutes on machines with good internet connection speeds.

### Data Preparation

Since data from different sources exhibit distinct characteristics, they require different preprocessing methods. We recommend preprocessing all data independently before training or inference. Currently, only resizing (to 224×224) and center cropping are applied during inference; other preprocessing methods (e.g., padding) should be used as needed.

In our training process, for UKB fundus images, we first extract a central 1400×1400 square and then resize it to 224x224. For fundus images from other sources, we pad them into squares before resizing to 224x224.

### Training

- Run the main script.
```
python main.py
```
The training will take a few minutes on machines with a consumer-grade GPU.
The trained model will be saved to `ckpt` folder.

### Inference

- Run the inference script.
```
python inference.py
```

## Citation

```
Zhang, X., Li, Q., Liang, Y. et al. AI framework for multidisease detection via retinal imaging. Nat Med (2026). https://doi.org/10.1038/s41591-026-04359-w
```
