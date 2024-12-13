# Conotoxin Classification and Analysis Repository

This repository contains files and scripts related to the classification and analysis of conotoxin sequences, including training and testing datasets, performance metrics, and prediction outputs. Below is a detailed description of the files included:

## Datasets

- drQiu_trainingData.xlsx: Contains the conotoxin sequences **nonIonToxinCombined.xlsx**: A dataset of non-ion channel targeting sequences.

- `og_teretoxins.csv`: dataset of teretoxins.

- predictions_og_teretoxin.csv: The output file showing the predicted class for each teretoxin (e.g., calcium, potassium, sodium, or non-ion channel targeting).

## Scripts and Files for ROC Curve Analysis

- roc_sep_sh.sh: An sbatch script to generate the ROC curve.

- roc_sep.py: A Python script for creating ROC curves for each class.

- roc_auc_multiclass_separated.png: ROC curve for each class.

- roc_auc_multiclass.png:  ROC curve plot that includes all classes and the average.

## Performance Metrics Analysis

- test_metrics_sh.sh: An sbatch script to calculate model performance metrics, including accuracy, precision, recall, and F1 score.

- test_metrics.py: A Python script for calculating the performance metrics.

## Teretoxin Predictions

- test_teretoxin_sh.sh: An sbatch script to predict the classes of teretoxins using the previously trained model.

- test_teretoxin.py: A Python script to predict teretoxin classes.

## How to Use

### Run ROC Curve Analysis:

Use roc_sep_sh.sh to generate the ROC curve. The script uses roc_sep.py to create ROC curve

### Calculate Performance Metrics:

Use test_metrics_sh.sh to compute metrics like accuracy, precision, recall, and F1 score. The script calls test_metrics.py for these calculations.

### Predict Teretoxin Classes:

Use test_og_teretoxin_sh.sh to predict the classes of teretoxins. The script utilizes test_teretoxin.py to classify each teretoxin and outputs results in predictions_og_teretoxin.csv.

## Outputs

- roc_auc_multiclass_separated.png: ROC curves for individual classes.

- roc_auc_multiclass.png: Combined ROC curve plot with average metrics.

- predictions_og_teretoxin.csv: Predicted classifications of teretoxins.



