# 3DOR_Finetuning
Code, data and checkpoints for reproducing results in "Fine-tuning 3D foundation models for geometric object retrieval" - Anon.

## Instructions

Download the data for all datasets (except proprietary) in this folder: https://huggingface.co/datasets/Anon3DOR/3DOR_pointclouds/blob/main/data_3DOR.zip

Unzip the `data_final/` directory in the root of this repository.

Create a conda environment (assuming a conda version is installed on the system) and run script for all datasets:
```
source install.sh
source 3DOR_results.sh
```

We found the results to be most consistent over various runs with the NDCG metric.
All results in the paper are the average of 5 consecutive runs of this script.


Feel free to create a github issue when problems arise!
