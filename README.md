# Large-Scale 3D Ground-Motion Synthesis with Physics-Inspired Latent Operator Flow Matching

## Simulation pipeline
![image](fig/data.PNG)

## Model architecture 
![image](fig/model.PNG)

## Inference 
![image](fig/inference.PNG)

## Setup and quick start 
First download the processed test dataset from [https://huggingface.co/datasets/Yaozhong/GMFlow](https://huggingface.co/datasets/Yaozhong/GMFlow), place the 300 test events under ``dataset`` folder
To set up the environment, create a conda environment

```
# clone project
git clone https://github.com/yzshi5/GMFlow.git
cd gmflow

# create conda environment
conda env create -f environment.yml

# Activate the `mino` environment
conda activate gmflow
```


## Video 
![Mw4.4 Point-Source event](./gif/M44_point_source.gif)

![Mw7.0 Finite-rupture event](./gif/M7_rupture_source.gif)
