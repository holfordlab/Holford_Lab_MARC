import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import esm
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from sklearn.datasets import make_classification
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve, auc, roc_auc_score
from tabulate import tabulate
import random
import string
import seaborn as sns
from sklearn.multiclass import OneVsRestClassifier
import joblib
train_df = pd.read_excel('toxinCombined_Finalv3.xlsx')
train_df_nont=pd.read_excel('nonIonToxinCombined.xlsx')
cone = pd.read_excel('drQiu_trainingData.xlsx')
cone['commonName'] = 'Cone Snail'
cone.rename(columns={'Sequence':'sequence'},inplace=True)
cone.rename(columns={'Entry' : 'entry'}, inplace=True)
def determine_ion_channel_class(row):
    if row['match'] == row['Zhang result']:
        return row['match']
    elif row['match'] == 'Other':
        return row['Zhang result']
    elif row['Zhang result'] == 'Other':
        return row['match']
    else:
        return None
cone['ionChannelClass'] = cone.apply(determine_ion_channel_class, axis=1)
cone = cone[cone['ionChannelClass'].notna()]

cone['ionChannelClass'] = cone['ionChannelClass'].replace({
    'K': 'Potassium',
    'Ca': 'Calcium',
    'Na': 'Sodium'
})
expected_columns = [
    'index', 'commonName', 'ionChannelClass', 'entry', 'entryName', 'organism', 'proteinName',
    'sequence', 'signal', 'pro', 'pep', 'post', 'sequenceCaution',
    'alternativeProducts', 'signalPeptide', 'proPeptide', 'peptide',
    'postTranslationalModification', 'disulfideBonds', 'pubmedID'
]
for col in expected_columns:
    if col not in cone.columns:
        cone[col] = 'dummy_data'
cone = cone[expected_columns]
train_df_nont['ionChannelClass']='nonToxin'
train_df_nont = train_df_nont[['index', 'commonName', 'ionChannelClass','entry', 'entryName', 'organism', 'proteinName',
       'sequence', 'signal', 'pro', 'pep', 'post', 'sequenceCaution',
       'alternativeProducts', 'signalPeptide', 'proPeptide', 'peptide',
       'postTranslationalModification', 'disulfideBonds', 'pubmedID'
       ]]
train_df = pd.concat([train_df, train_df_nont,cone], ignore_index=True)
train_df['stratify'] = train_df['commonName'] + train_df['ionChannelClass']
duplicates_all_occurrences = train_df[train_df.duplicated(keep=False)]
stratify_column = train_df['stratify']
train_df,test_df= train_test_split(train_df, test_size=0.2, random_state=42, stratify = stratify_column)
train_df = train_df[['ionChannelClass', 'sequence', 'entry' , 'stratify']]
train_df.drop('stratify', axis=1, inplace=True)
sequences_train = train_df['sequence']
# Load ESM-2 model
model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
batch_converter = alphabet.get_batch_converter()
model.eval()
def get_stable_embedding(sequence):
    """
    Takes a protein sequence (str) and returns a FIXED size vector (1280,).
    No padding issues. No variable lengths.
    """
    data = [("protein1", sequence)]
    batch_labels, batch_strs, batch_tokens = batch_converter(data)

    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[33], return_contacts=False)

    # Extract the raw embedding for the last layer
    # Shape is (Batch, Sequence_Length, 1280)
    token_representations = results["representations"][33]

    # CRITICAL STEP: MEAN POOLING
    # We ignore the first token (start) and last token (end) and average the rest.
    # This guarantees the output is ALWAYS (1280,) regardless of protein length.
    seq_len = (batch_tokens != alphabet.padding_idx).sum(1).item()
    # token_representations[0, 1 : seq_len - 1] selects the actual amino acids
    mean_embedding = token_representations[0, 1 : seq_len - 1].mean(0)

    return mean_embedding.numpy()
X_train_embeddings = []
sequences = train_df['sequence'].tolist()
for seq in sequences:
    vec = get_stable_embedding(seq)
    X_train_embeddings.append(vec)
x = np.array(X_train_embeddings)
train_labels = train_df['ionChannelClass']
label_encoder = OneHotEncoder(handle_unknown='ignore')
original_labels=['Sodium','Potassium','Calcium','nonToxin']
label_encoder.fit(np.array(original_labels).reshape(-1,1))
encoded_class_names = list(label_encoder.categories_)
tr_encoded_data = label_encoder.fit_transform(train_labels.values.reshape(-1,1))
sm_1 = SMOTE(random_state=42)
X_res, y_res = sm_1.fit_resample(x,  tr_encoded_data.toarray())
class_counts = tr_encoded_data.toarray().sum(axis=0)
ion_channels = ['Ca', 'K', 'Na','nonToxin']
original_counts = tr_encoded_data.toarray().sum(axis=0)
smote_counts =y_res.sum(axis=0)
max_dep =15
n_estimators = 94
rfc = RandomForestClassifier(n_estimators=n_estimators, random_state=42, max_depth=max_dep)
rfc.fit(X_res, y_res)
joblib.dump(rfc, 'marc_model.pkl')
print("Model saved as 'marc_model.pkl'")
