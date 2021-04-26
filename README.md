# Predicting Phenotypic Polymyxin Resistance in Klebsiella Pneumoniae

  

This repository compromises all the processes and resources that were used to create the ML pipeline of "Predicting Phenotypic Polymyxin Resistance in *_Klebsiella Pneumoniae"._*

  

# About this repository

  

![Predicting%20Phenotypic%20Polymyxin%20Resistance%20in%20Kleb%20e0e9934f7689470c8b047acbbbbe3341/Screen_Shot_2021-04-25_at_17.16.37.png](https://github.com/jcsn13/PredictingPRinKB/blob/master/images/Screen_Shot_2021-04-25_at_17.16.37.png)

  

Since the use of machine learning approaches for scientific studies has gained much prominence, utilizing them in scientific researches can deliver accurate results and save a huge amount of resources. Based on that, Nenad Macesic performed the research article "Predicting Phenotypic Polymyxin Resistance in Klebsiella pneumoniae through Machine Learning Analysis of Genomic Data"([https://msystems.asm.org/content/5/3/e00656-19](https://msystems.asm.org/content/5/3/e00656-19)) to explore how ML analysis would perform, due to the resource intensity and difficulty to perform Phenotypic polymyxin susceptibility tests accurately.

  

In the article, Nenad and his team presented high accurate results and emphasized that the using of Genome-wide association study filtering outperforms the regular machine learning approach. Our objective in this repository is to apply a combination of ML algorithms and feature selection to extract better performance metrics from these models using the same data, and also we will explore the reasons for such outputs.

  

Later on, we will use our own data to test those models. To do so, we need to run a specific bioinformatics pipeline to generate the features that we used to train the model.

  

## Team

  

### CNS Lab

  

Computational Natural Science Lab is a Research Group founded by professors and students of CESAR School University to explore and search about topics related to computational biology.

  

### Members

  

- José Cláudio Soares Neto(jcsn@cesar.school) → Volunteer Researcher

- Erico Souza Teixeira(est@cesar.school) → Thesis Advisor

  

## Roadmap

  

- [ ]  Predicting PR in *_Klebsiella Pneumoniae_* Pipeline

	- [x]  ML Pipeline

		- [x]  Study and Understand the datasets

		- [x]  Develop a new ML Pipeline using different approaches

		- [x]  Compare performance metrics with the previous Pipeline

	- [ ]  Bioinformatics Pipeline

		- [ ]  Perform variant calling on obtained isolates(FASTA files)

		- [ ]  Detect insertion sequences(ISseerker and BLAST)

		- [ ]  Generate input matrix with obtained isolates

		- [ ]  Test ML pipelines with our isolates

	- [ ]  Merge Pipelines

		- [ ]  Automate input matrix generation

  

# ****Installation****

  

![Predicting%20Phenotypic%20Polymyxin%20Resistance%20in%20Kleb%20e0e9934f7689470c8b047acbbbbe3341/Screen_Shot_2021-04-25_at_17.08.06.png](https://github.com/jcsn13/PredictingPRinKB/blob/master/images/Screen_Shot_2021-04-25_at_17.08.06.png)

  

1. Clone GitHub repository

2. Execute the `Makefile`, it will create a python virtual environment and install all dependencies

  

# ****About the data****

  

![Predicting%20Phenotypic%20Polymyxin%20Resistance%20in%20Kleb%20e0e9934f7689470c8b047acbbbbe3341/Screen_Shot_2021-04-25_at_17.09.27.png](https://github.com/jcsn13/PredictingPRinKB/blob/master/images/Screen_Shot_2021-04-25_at_17.09.27.png)

  

All the input data used was obtained from [https://github.com/crowegian/AMR_ML](https://github.com/crowegian/AMR_ML). All datasets are `.csv` organized as a binary matrix. There are two types of data:

  

- Datasets without GWAS filtering → Uses metadata and full.

- Datasets with GWAS filtering → Uses metadata, gwas, and full.

  

Check the previous repository to get a deeper insight, and also read: [https://msystems.asm.org/content/5/3/e00656-19](https://msystems.asm.org/content/5/3/e00656-19).

  

# ****Running the pipeline****

  

![Predicting%20Phenotypic%20Polymyxin%20Resistance%20in%20Kleb%20e0e9934f7689470c8b047acbbbbe3341/Screen_Shot_2021-04-25_at_17.12.12.png](https://github.com/jcsn13/PredictingPRinKB/blob/master/images/Screen_Shot_2021-04-25_at_17.12.12.png)

  

Firstly, you need to activate the virtual environment running: `source .env/bin/activate`

  

## ****Running Model Training****

  

The following script will train all the models, should take at least 6h. The trained models will be saved into the models folder, the same will happen with the roc_auc plots.

  

`python PredictPRinKP.py`

  

## Running Prediction

  

For running predictions make sure that all .`pkl` files are in their respective directory, then run the following command:

  

`python RunPrediction.py`

  

This is a faster way of getting models performancem since they were trained before and also a good way of testing our model with unseen data.

  

# Obtained Results

  

![Predicting%20Phenotypic%20Polymyxin%20Resistance%20in%20Kleb%20e0e9934f7689470c8b047acbbbbe3341/Screen_Shot_2021-04-25_at_17.46.30.png](https://github.com/jcsn13/PredictingPRinKB/blob/master/images/Screen_Shot_2021-04-25_at_17.46.30.png)

  

As our work was based on the research paper mentioned above, we followed the pipeline used for the reference-based and GWAS approaches represented below:

  

## Their Pipeline

  

![Predicting%20Phenotypic%20Polymyxin%20Resistance%20in%20Kleb%20e0e9934f7689470c8b047acbbbbe3341/Screen_Shot_2021-04-25_at_18.03.09.png](https://github.com/jcsn13/PredictingPRinKB/blob/master/images/Screen_Shot_2021-04-25_at_18.03.09.png)

  

## Our Pipeline

  

![Predicting%20Phenotypic%20Polymyxin%20Resistance%20in%20Kleb%20e0e9934f7689470c8b047acbbbbe3341/Untitled_Diagram-2.png](https://github.com/jcsn13/PredictingPRinKB/blob/master/images/Untitled_Diagram-2.png)

  

We used the same datasets that were used to generate the ML models of the referenced paper. However, for each approach, we selected a different algorithm for feature selection, the RFE(Recursive Feature Elimination). Furthermore, we selected different hyperparameters for each model, resulting in a better performance overall.

  

To compare our models with the ones reported in the article, we used AUROC score with  0.95 confidence interval calculation using 10 random  sub-samples.

  

## Performance Comparison
| Approach (CI 95%)                                     | CUIMC                             | Non-CUIMC                 | All                      |
|------------------------------------------------------|-----------------------------------|---------------------------|--------------------------|
| Ref-based(paper)                                     | Random Forest 0.885 (0.849, 0.92) | GBTC 0.933 (0.884, 0.982) | GBTC 0.894 (0.838, 0.95) |
| Ref-based with GWAS(paper)                           | SVC 0.893 (0.864, 0.922)          | SVC 0.933 (0.888, 0.979)  | SVC 0.931 (0.915, 0.947) |
| Ref-based value with polymyxin exposure  data(paper) | GBTC 0.923 (0.88, 0.965)          | -                         | -                        |
| Ref-based(ours)                                      | LR 0.951(0.928, 0.973)            | SVC 0.939 (0.895, 0.984)  | LR 0.915 (0.897, 0.933)  |
| Ref-based with GWAS(ours)                            | LR 0.931 (0.905, 0.956)           | SVC 0.982 (0.974, 0.990)  | SVC 0.956 (0.945, 0.966) |
| Ref-based value with polymyxin exposure  data(ours)  | SDGC 0.964 (0.95, 0.978)          | -                         | -                        |
