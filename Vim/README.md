

## Quick Start

- Python 3.10.13

  - `conda create -n your_env_name python=3.10.13`

- torch 2.1.1 + cu118
  - `pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu118`

- Requirements: vim_requirements.txt
  - `pip install -r fambav/vim_requirements.txt`

- Install ``causal_conv1d`` and ``mamba``
  - `pip install -e causal_conv1d>=1.1.0`
  - `pip install -e mamba-1p1p1`

- It is easy to use docker
  - `docker pull zhuohaol/skeletonkey:latest`
  - `docker run -it --rm --gpus all --ipc=host --name skeletonkey zhuohaol/skeletonkey:latest`
  
  

## Train and Fine-tune
Refer to `test.sh` where I noted for different models and datasets. The scripts are referred to files in `/scripts`. 

`models_mamba.py` defines vim models.

`datasets.py` defines datasets.

`main.py` is the entry point for training and fine-tuning.