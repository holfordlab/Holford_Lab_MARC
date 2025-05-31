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

train_df = pd.read_excel('toxinCombined_Finalv3.xlsx')
train_df_nont=pd.read_excel('nonIonToxinCombined.xlsx')
cone = pd.read_excel('drQiu_trainingData.xlsx')
cone['commonName'] = 'Cone Snail'

print ("cone head: ", cone.head() )  
cone.rename(columns={'Sequence':'sequence'},inplace=True)
cone.rename(columns={'Entry' : 'entry'}, inplace=True)
# Function to determine ionChannelClass
def determine_ion_channel_class(row):
    if row['match'] == row['Zhang result']:
        return row['match']
    elif row['match'] == 'Other':
        return row['Zhang result']
    elif row['Zhang result'] == 'Other':
        return row['match']
    else:
        return None
# Apply the function to each row to create the ionChannelClass column
cone['ionChannelClass'] = cone.apply(determine_ion_channel_class, axis=1)

# Remove rows where ionChannelClass is None
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


# Add dummy data for missing columns
for col in expected_columns:
    if col not in cone.columns:
        cone[col] = 'dummy_data'

# Remove any extra columns
cone = cone[expected_columns]


train_df_nont['ionChannelClass']='nonToxin'
train_df_nont = train_df_nont[['index', 'commonName', 'ionChannelClass','entry', 'entryName', 'organism', 'proteinName',
       'sequence', 'signal', 'pro', 'pep', 'post', 'sequenceCaution',
       'alternativeProducts', 'signalPeptide', 'proPeptide', 'peptide',
       'postTranslationalModification', 'disulfideBonds', 'pubmedID'
       ]]
train_df = pd.concat([train_df, train_df_nont,cone], ignore_index=True)
train_df['stratify'] = train_df['commonName'] + train_df['ionChannelClass']
#train_df = train_df[['ionChannelClass', 'sequence', 'entry', 'stratify']]
print("Total Sodium count is : " , sum(train_df['ionChannelClass']=='Sodium'))
print("Total Potassium count is : " , sum(train_df['ionChannelClass']=='Potassium'))
print("Total Calcium count is : " , sum(train_df['ionChannelClass']=='Calcium'))
print("Total non toxin count is : " , sum(train_df['ionChannelClass']=='nonToxin'))
duplicates_all_occurrences = train_df[train_df.duplicated(keep=False)]
#print("All Occurrences of Duplicates:")
#print(duplicates_all_occurrences)
stratify_column = train_df['stratify']
train_df,test_df= train_test_split(train_df, test_size=0.2, random_state=42, stratify = stratify_column)
train_df = train_df[['ionChannelClass', 'sequence', 'entry' , 'stratify']] 
train_df.drop('stratify', axis=1, inplace=True)
test_df.drop('stratify', axis=1, inplace=True)
test_df = test_df[test_df['commonName'] == 'Sea Anemone']
print("Total Sodium count IN SNAKE is : " , sum(test_df['ionChannelClass']=='Sodium'))
print("Total Potassium count IN SNAKE is : " , sum(test_df['ionChannelClass']=='Potassium'))
print("Total Calcium count IN SNAKE is : " , sum(test_df['ionChannelClass']=='Calcium'))
print("Total non toxin count IN SNAKE is : " , sum(test_df['ionChannelClass']=='nonToxin'))
test_df = test_df[['ionChannelClass', 'sequence' , 'entry']]
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
#with torch.no_grad():
 #   tr_results = model(tr_batch_tokens, repr_layers =[33], return_contacts = False)
#tr_token_representations = tr_results["representations"][33]
#np.save('esm_features_train_fixed4_13.npy', tr_token_representations)
tr_token_representations = np.load('esm_features_seq.npy')

# Generate per-sequence representations via averaging
# NOTE: token 0 is always a beginning-of-sequence token, so the first residue is token 1.

tr_sequence_representations = []
for i, tokens_len in enumerate(tr_batch_lens):
    tr_sequence_representations.append(tr_token_representations[i, 1 : tokens_len - 1].mean(0))
train_labels = train_df['ionChannelClass']
label_encoder = OneHotEncoder(handle_unknown='ignore')
original_labels=['Sodium','Potassium','Calcium','nonToxin']
label_encoder.fit(np.array(original_labels).reshape(-1,1))
encoded_class_names = list(label_encoder.categories_)
tr_encoded_data = label_encoder.fit_transform(train_labels.values.reshape(-1,1))
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
np.save('esm_features_roc_Sea_Anemone.npy', test_token_representations)
#test_token_representations = np.load('esm_features_seq_test.npy')
# Generate per-sequence representations via averaging
# NOTE: token 0 is always a beginning-of-sequence token, so the first residue is token 1.
test_sequence_representations = []
for i, tokens_len in enumerate(test_batch_lens):
    test_sequence_representations.append(test_token_representations[i, 1 : tokens_len - 1].mean(0))
test_labels = test_df['ionChannelClass']
transformed_labels =label_encoder.transform(test_labels.values.reshape(-1,1))
x_test = test_sequence_representations
x_test = np.stack([x.numpy() for x in x_test])
print("type(x_test): ",type(x_test))
print("len(x_test) : " , len(x_test))
print("[type(x) for x in x_test[:5]] : ", [type(x) for x in x_test[:5]])
print("[np.shape(x) for x in x_test[:5]] : " ,[np.shape(x) for x in x_test[:5]])
y_pred_test = rfc.predict(x_test)
'''
def func_to_check(max_dep, n_estimators):
  rfc = RandomForestClassifier(n_estimators=n_estimators, random_state=42, max_depth=max_dep)
  x = tr_sequence_representations
  rfc.fit(X_res, y_res)
  y_pred = rfc.predict(x)
  accuracy = accuracy_score(tr_encoded_data.toarray(), y_pred)
  precision = precision_score(tr_encoded_data.toarray(), y_pred, average='macro')
  recall = recall_score(tr_encoded_data.toarray(), y_pred, average='macro')
  f1 = f1_score(tr_encoded_data.toarray(), y_pred, average='macro')
  y_pred_test = rfc.predict(x_test)
  test_accuracy = accuracy_score(transformed_labels.toarray(), y_pred_test)
  test_precision = precision_score(transformed_labels.toarray(), y_pred_test, average='weighted')
  test_recall = recall_score(transformed_labels.toarray(), y_pred_test, average='weighted')
  test_f1 = f1_score(transformed_labels.toarray(), y_pred_test, average='weighted')
  return accuracy, precision, recall, f1 , test_accuracy, test_precision, test_recall, test_f1


n_estimators_range = np.linspace(10, 200, 10).astype(int)
print("printed metrics are training accuracy, training precision, training recall, training f1 score, testing accuracy, testing precision, testing recall, testing f1 score.")
for n_estimators in n_estimators_range:
  for max_dep in [5, 7, 8, 10, 15, 20, 25, 30, 40]:
      print("max depth: ", max_dep, "n_ estimators:", n_estimators, "metrics:", func_to_check(max_dep, n_estimators))'''
      
# inverse transform pipeline implementation
# best parameters for number of trees and max depth based on previous experiment is :
max_dep =15
n_estimators = 94
rfc = RandomForestClassifier(n_estimators=n_estimators, random_state=42, max_depth=max_dep)
x = tr_sequence_representations
rfc.fit(X_res, y_res)
y_pred = rfc.predict(x)
accuracy = accuracy_score(tr_encoded_data.toarray(), y_pred)
precision = precision_score(tr_encoded_data.toarray(), y_pred, average='macro')
recall = recall_score(tr_encoded_data.toarray(), y_pred, average='macro')
f1 = f1_score(tr_encoded_data.toarray(), y_pred, average='macro')
print("type(x_test): ",type(x_test))
print("len(x_test) : " , len(x_test))
print("[type(x) for x in x_test[:5]] : ", [type(x) for x in x_test[:5]])
print("[np.shape(x) for x in x_test[:5]] : " ,[np.shape(x) for x in x_test[:5]])
y_pred_test = rfc.predict(x_test)
#starting ROC AUC
plt.rcParams['font.family']='Cambria'

classifier = OneVsRestClassifier(RandomForestClassifier(n_estimators = n_estimators, random_state=42, max_depth = max_dep))
y_score = classifier.fit(X_res, y_res).predict_proba(x_test)
print("y_score :  ", y_score)
print("y_score type : " , type(y_score))
print("y_score shape : " , np.shape(y_score))
# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(4):
    fpr[i], tpr[i], _ = roc_curve(transformed_labels.toarray()[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(transformed_labels.toarray().ravel(), y_score.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Plot all ROC curves
#plt.figure()
fig, ax = plt.subplots(figsize=(10,7))
lw = 2
# Assuming label_encoder is already fitted
class_labels_roc = label_encoder.categories_[0]  # Store categories in a variable
class_labels_roc = ['non ion channel targeting' if label == 'nonToxin' else label for label in class_labels_roc]
# Plot ROC curve for each class using class names
for i in range(4):
    ax.plot(fpr[i], tpr[i], lw=lw, label='ROC curve of {0} (area = {1:0.2f})'.format(class_labels_roc[i], roc_auc[i]))

# Plot ROC curve for each class
#for i in range(4):
   # plt.plot(fpr[i], tpr[i], lw=lw, label='ROC curve of class {0} (area = {1:0.2f})'.format(i, roc_auc[i]))

# Plot micro-average ROC curve
ax.plot(fpr["micro"], tpr["micro"],
         label='micro-average ROC curve (area = {0:0.2f})'.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

ax.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel('False Positive Rate',fontsize=24)
ax.set_ylabel('True Positive Rate',fontsize=24)
ax.set_title('ROC multi-class',fontsize=32,fontweight='bold')
ax.legend(loc='center left', bbox_to_anchor=(1.02,0.5), fontsize=24, frameon=False)
#plt.tight_layout()
fig.subplots_adjust(right=0.75)
# Save the plot to a file
plt.savefig('roc_auc_multiclass_Sea_Anemone.png',dpi=300, bbox_inches='tight')




# getting class labels

'''
encoded_class_names = list(label_encoder.categories_)



#print metrics

test_accuracy = accuracy_score(transformed_labels.toarray(), y_pred_test)
print(f"Overall Test Accuracy: {test_accuracy:.4f}\n")

# Precision, recall, and F1 for each class
test_precision = precision_score(transformed_labels.toarray(), y_pred_test, average=None)
test_recall = recall_score(transformed_labels.toarray(), y_pred_test, average=None)
test_f1 = f1_score(transformed_labels.toarray(), y_pred_test, average=None)


#print the whole test_precision, test_recall, test_f1
print("printing all the metrics for all classes, these must be 4 for each value")
print("test_precision : ", test_precision)
print("test_recall :" , test_recall)
print("test_f1 : " , test_f1)

# Printing metrics for each class by name
for i, class_name in enumerate(encoded_class_names):
    print(f"{class_name}: Precision: {test_precision[i]:.4f}, Recall: {test_recall[i]:.4f}, F1: {test_f1[i]:.4f}")





# inverse transform outputs the y_pred_test

# Getting the inverse transform of predictions to get class names
y_pred_test_inv = label_encoder.inverse_transform(y_pred_test)
print("y_pred_test_in : " , y_pred_test_inv)
print("test_labels : " , test_labels)
print("type of y_pred_test_inv : " , type(y_pred_test_inv))
print("type of test_labels : " , type(test_labels))

# Flatten y_pred_test_inv if it's 2D
y_pred_test_inv_flat = y_pred_test_inv.flatten()
print("type of y_pred_test_inv_flat : " , type(y_pred_test_inv_flat) )
print("y_pred_test_inv_flat : " , y_pred_test_inv_flat)

# Optionally, convert test_labels to a numpy array
test_labels_array = test_labels.values
print("type of test_labels_array :", type(test_labels_array))
print(" test_labels_array : " , test_labels_array )




# Confusion Matrix
cm = confusion_matrix(test_labels_array, y_pred_test_inv_flat)

# Plot the confusion matrix
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=encoded_class_names, yticklabels=encoded_class_names)
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.show()

# Store the actual and predicted labels in a DataFrame
results_df = pd.DataFrame({
    'Actual Labels': y_true_inv,
    'Predicted Labels': test_labels
})

# Save to a CSV file
results_df.to_csv('roc_predictions.csv', index=False)

print("CSV file with actual and predicted labels has been saved.")'''
