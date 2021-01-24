# Predicting phenotypic polymyxin resistance in *Klebsiella pneumoniae*

# Installation

1. Clone GitHub repository
2. Execute the `Makefile`, it will create a python virtual environment and install all dependencies

# About the used data

All the input data used was obtained from https://github.com/crowegian/AMR_ML. All datasets are `.csv` organized as a binary matrix.
There are two types of data:

- Datasets without GWAS filtering -> Uses a metadata and full.

- Datasets with GWAS filtering -> Uses a metadata, gwas, and full.

Check the previous repository to get a deeper insight, and also read https://msystems.asm.org/content/5/3/e00656-19. 

# Running the pipeline

Firstly, you need to activate the virtual environment running:
`source .env/bin/activate`

## Running Model Training
The following script will train all the models, it should take at least 6h. The trained models will be saved into the models folder, the same will happen with the roc_auc plots. 
`python PredictPRinKP.py`

## Running prediction 

## Extracting feature importance