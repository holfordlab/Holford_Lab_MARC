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
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve, auc
from tabulate import tabulate
import random
import string

train_df = pd.read_excel('toxinCombined_Finalv3.xlsx')
train_df_nont=pd.read_excel('nonIonToxinCombined.xlsx')
train_df_nont['ionChannelClass']='nonToxin'
train_df_nont = train_df_nont[['index', 'commonName', 'ionChannelClass','entry', 'entryName', 'organism', 'proteinName',
       'sequence', 'signal', 'pro', 'pep', 'post', 'sequenceCaution',
       'alternativeProducts', 'signalPeptide', 'proPeptide', 'peptide',
       'postTranslationalModification', 'disulfideBonds', 'pubmedID'
       ]]
train_df = pd.concat([train_df, train_df_nont], ignore_index=True)
train_df['stratify'] = train_df['commonName'] + train_df['ionChannelClass']
train_df = train_df[['ionChannelClass', 'sequence', 'entry', 'stratify']]
print("Total Sodium count is : " , sum(train_df['ionChannelClass']=='Sodium'))
print("Total Potassium count is : " , sum(train_df['ionChannelClass']=='Potassium'))
print("Total Calcium count is : " , sum(train_df['ionChannelClass']=='Calcium'))
print("Total non toxin count is : " , sum(train_df['ionChannelClass']=='nonToxin'))
duplicates_all_occurrences = train_df[train_df.duplicated(keep=False)]
#print("All Occurrences of Duplicates:")
#print(duplicates_all_occurrences)
stratify_column = train_df['stratify']
train_df,test_df= train_test_split(train_df, test_size=0.2, random_state=42, stratify = stratify_column)
train_df.drop('stratify', axis=1, inplace=True)
test_df.drop('stratify', axis=1, inplace=True)
train_l = list(train_df['sequence'])
test_l = list(test_df['sequence'])
leak = 0
for test_index, seq in enumerate(test_l):
    if seq in train_l:
        train_index = train_l.index(seq)
        leak += 1
        #print(f'leak {leak}: Sequence {seq} found in test_l at index {test_index} and in train_l at index {train_index}')
# Assuming 'SEQUENCE' is the column you want to check for existence
sequences_train = train_df['sequence']
# Create a boolean mask to identify rows in test_df that have 'SEQUENCE' values present in train_df
mask = test_df['sequence'].isin(sequences_train)
# Remove the rows from test_df where 'SEQUENCE' exists in train_df
test_df = test_df[~mask]
# Load ESM-2 model
model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
batch_converter = alphabet.get_batch_converter()
model.eval()  # disables dropout for deterministic results
dummy_train = [(tuple(list(x))[1], tuple(list(x))[0])  for x in list(train_df[['sequence','entry']].values)]
tr_batch_labels, tr_batch_strs, tr_batch_tokens = batch_converter(dummy_train)
tr_batch_lens = (tr_batch_tokens != alphabet.padding_idx).sum(1)
tr_token_representations = np.load('esm_features_all_toxins_nontoxin.npy')
# Generate per-sequence representations via averaging
# NOTE: token 0 is always a beginning-of-sequence token, so the first residue is token 1.
tr_sequence_representations = []
for i, tokens_len in enumerate(tr_batch_lens):
    tr_sequence_representations.append(tr_token_representations[i, 1 : tokens_len - 1].mean(0))
train_labels = train_df['ionChannelClass']
label_encoder = OneHotEncoder(handle_unknown='ignore')

#add list of labels to fit the data

original_labels = ['Sodium', 'Potassium', 'Calcium', 'nonToxin']
label_encoder.fit(np.array(original_labels).reshape(-1,1))
encoded_class_names = list(label_encoder.categories_)

tr_encoded_data = label_encoder.transform(train_labels.values.reshape(-1,1))
rfc = RandomForestClassifier(n_estimators=15, random_state=42)
x = tr_sequence_representations
# begin smote
sm_1 = SMOTE(random_state=42)
X_res, y_res = sm_1.fit_resample(x,  tr_encoded_data.toarray())
print("shape of resampled y values of training set : " , np.shape(y_res))
class_counts = tr_encoded_data.toarray().sum(axis=0)
print("class counts : ", class_counts)
# Given data
ion_channels = ['Ca', 'K', 'Na','nonToxin']
original_counts = tr_encoded_data.toarray().sum(axis=0)
smote_counts =y_res.sum(axis=0) # Updated counts after applying SMOTE
# Set up the figure and axes
fig, ax = plt.subplots(figsize=(8, 6))

# Plotting the original and SMOTE counts as two separate bars
width = 0.4
x = range(len(ion_channels))

# ax.bar(x, original_counts, width=width, label='Original Counts')
# ax.bar([i + width for i in x], smote_counts, width=width, label='SMOTE Counts')

# # Adding labels and title
# ax.set_xlabel('Ion Channels')
# ax.set_ylabel('Number of Samples')
# ax.set_title('Class Distribution Before and After SMOTE')
# ax.set_xticks([i + width/2 for i in x])
# ax.set_xticklabels(ion_channels)
# ax.legend()

# Display the plot
plt.tight_layout()
plt.show()
rfc.fit(X_res, y_res)
y_pred = rfc.predict(X_res)
# overall accuracy
accuracy = accuracy_score(y_res, y_pred)
precision = precision_score(y_res, y_pred, average='macro')
recall = recall_score(y_res, y_pred, average='macro')
f1 = f1_score(y_res, y_pred, average='macro')
print("Initial random forest model performance: ")
print("Training accuracy : ", accuracy)
print("Training precision : ", precision)
print("Training recall : ", recall)
print("Training f1 score : ", f1)
test_data = [(tuple(list(x))[1], tuple(list(x))[0])  for x in list(test_df[['sequence','entry']].values)]
test_batch_labels, test_batch_strs, test_batch_tokens = batch_converter(test_data)
test_batch_lens = (test_batch_tokens != alphabet.padding_idx).sum(1)
# Extract per-residue representations (on CPU)
with torch.no_grad():
    test_results = model(test_batch_tokens, repr_layers=[33], return_contacts=False)
test_token_representations = test_results["representations"][33]
np.save('esm_features_all_toxins_nontoxin_test.npy', test_token_representations)
test_token_representations = np.load('esm_features_all_toxins_nontoxin_test.npy')
# Generate per-sequence representations via averaging
# NOTE: token 0 is always a beginning-of-sequence token, so the first residue is token 1.
test_sequence_representations = []
for i, tokens_len in enumerate(test_batch_lens):
    test_sequence_representations.append(test_token_representations[i, 1 : tokens_len - 1].mean(0))
test_labels = test_df['ionChannelClass']
transformed_labels =label_encoder.transform(test_labels.values.reshape(-1,1))
x_test = test_sequence_representations
y_pred_test = rfc.predict(x_test)

def func_to_check(max_dep, n_estimators):
  rfc = RandomForestClassifier(n_estimators=n_estimators, random_state=42, max_depth=max_dep)
  #x = [tensor.numpy() for tensor in tr_sequence_representations]
  x = tr_sequence_representations
  rfc.fit(X_res, y_res)
  y_pred = rfc.predict(x)
  accuracy = accuracy_score(tr_encoded_data.toarray(), y_pred)
  precision = precision_score(tr_encoded_data.toarray(), y_pred, average='macro')
  recall = recall_score(tr_encoded_data.toarray(), y_pred, average='macro')
  f1 = f1_score(tr_encoded_data.toarray(), y_pred, average='macro')
  y_pred_test = rfc.predict(x_test)
  #overall accuracy
  test_accuracy = accuracy_score(transformed_labels.toarray(), y_pred_test)
  #Precision, recall, and F1 for each class

  
  test_precision = precision_score(transformed_labels.toarray(), y_pred_test, average=None) 
  test_recall = recall_score(transformed_labels.toarray(), y_pred_test, average=None)
  test_f1 = f1_score(transformed_labels.toarray(), y_pred_test, average=None)

  print("Overall training accuracy :", accuracy, "Training precision: ",precision, "Training recall: ", recall, "Training F1 score: ", f1 )
  print("overall test accuracy : " , test_accuracy ) 
  for i, class_name in enumerate(encoded_class_names):
      print("encoded_class_names : ", encoded_class_names)
      print("i : ", i)
      print("class_name :", class_name)
      print("test_precision total : ", test_precision)
      print("test_recall total : " , test_recall)
      print("test_f1 : " , test_f1 )
      print(f"{class_name}: Precision: {test_precision[i]:.4f}, Recall: {test_recall[i]:.4f}, F1: {test_f1[i]:.4f}")
  


n_estimators_range = np.linspace(10, 200, 10).astype(int)
print("printed metrics are training accuracy, training precision, training recall, training f1 score, testing accuracy, testing precision, testing recall, testing f1 score.")
for n_estimators in n_estimators_range:
  for max_dep in [5, 7, 8, 10, 15, 20, 25, 30, 40]:
      print("max depth: ", max_dep, "n_ estimators:", n_estimators, "metrics:")
      func_to_check(max_dep, n_estimators)
      
