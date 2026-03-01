# Conotoxin Classification and Analysis Repository

This repository contains files and scripts related to the classification and analysis of conotoxin sequences, including training and testing datasets, performance metrics, and prediction outputs. Below is a detailed description of the files included:

## Datasets

- drQiu_trainingData.xlsx: Contains the conotoxin sequences **nonIonToxinCombined.xlsx**: A dataset of non-ion channel targeting sequences.

- `og_teretoxins.csv`: dataset of teretoxins.

- predictions_og_teretoxin.csv: The output file showing the predicted class for each teretoxin (e.g., calcium, potassium, sodium, or non-ion channel targeting).

## Scripts and Files for ROC Curve Analysis

- roc_sep_sh.sh: An sbatch script to generate the ROC curve.
- roc_font_fixed.py: A Python script for creating ROC curves. All classes included in training.
- roc_cone.py: A python script for creating per taxon ROC curves.

- roc_sep.py: A Python script for creating ROC curves for each class. (Cone excluded in training)

- roc_auc_multiclass_separated.png: ROC curve for each class.

- roc_auc_multiclass.png:  ROC curve plot that includes all classes and the average.

## Performance Metrics Analysis

- test_metrics_sh.sh: An sbatch script to calculate model performance metrics, including accuracy, precision, recall, and F1 score.

- test_metrics.py: A Python script for calculating the performance metrics.

## Teretoxin Predictions

- test_teretoxin_sh.sh: An sbatch script to predict the classes of teretoxins using the previously trained model.

- test_teretoxin.py: A Python script to predict teretoxin classes.

## Direct Inference Pipeline (Reviewer Update)

- `marc_model.pkl`: The pre-trained Random Forest model, allowing users to run predictions instantly without needing to retrain from scratch.
- `predict_marc.py`: A standalone Python script that automatically extracts ESM-2 features and predicts targets directly from a user-provided CSV file. Includes a confidence threshold that outputs "No Prediction" for low-confidence sequences.
- `predict_marc_sh.sh`: An sbatch script for running the standalone prediction pipeline on a SLURM cluster.
- `marc.py` & `marc_sh.sh`: The core training script and corresponding sbatch file used to generate the pre-trained model.

## How to Use

### Run ROC Curve Analysis:

Use roc_sep_sh.sh to generate the ROC curve. The script uses roc_sep.py to create ROC curve. Similarly use roc_font_fixed.py and roc_cone.py.

### Calculate Performance Metrics:

Use test_metrics_sh.sh to compute metrics like accuracy, precision, recall, and F1 score. The script calls test_metrics.py for these calculations.

### Predict Teretoxin Classes:

Use test_og_teretoxin_sh.sh to predict the classes of teretoxins. The script utilizes test_teretoxin.py to classify each teretoxin and outputs results in predictions_og_teretoxin.csv.

### How to predict ion channels for new peptides
1. **Set up the environment:**<br>
Install all required dependencies using the provided environment.yml file:<br>
<pre>conda env create -f environment.yml <br>
conda activate py371 <br> </pre>
2. **Prepare the project:** <br>
Make sure this repository is cloned and all scripts are in the same directory: <br>
<pre>git clone https://github.com/holfordlab/Holford_Lab.git <br>
cd Holford_Lab <br></pre>
3. **Format your input data** <br>
Prepare your input dataset in the same format as `og_teretoxins.csv`. This file should include peptide sequences and any other necessary columns required by the script.
4. **Update the script** <br>
Open the `test_teretoxin.py` file and update the dataset file name from `og_teretoxins.csv` to the name of your custom dataset file.
5. **Run the prediction** <br>
You can either run the script directly: <br>
<pre>python test_teretoxin.py <br></pre>
or submit it as a job on a cluster using: <br>
<pre>sbatch test_og_teretoxin_sh.sh</pre>
Ion channel predictions will be saved in `predictions_og_teretoxin.csv`.

### Quick Prediction via Command Line (No Retraining Needed)

We have streamlined the prediction process so users can classify new sequences using the pre-trained model (`marc_model.pkl`) via a single command:

1. **Format your dataset:** Ensure your input CSV has a column named `sequence` (or `seq`) containing the peptide sequences.
2. **Run the prediction:** Pass your CSV file directly as an argument in the terminal. You do not need to modify the Python code.
<pre>python predict_marc.py your_custom_dataset.csv</pre>

*(If using a SLURM cluster, update the input filename inside `predict_marc_sh.sh` and run: `sbatch predict_marc_sh.sh`)*

**Output:** The script will extract the ESM-2 features and generate a new file named `predicted_your_custom_dataset.csv` containing the classifications and confidence scores.

## Outputs

- roc_auc_multiclass_separated.png: ROC curves for individual classes.

- roc_auc_multiclass.png: Combined ROC curve plot with average metrics.

- predictions_og_teretoxin.csv: Predicted classifications of teretoxins.



