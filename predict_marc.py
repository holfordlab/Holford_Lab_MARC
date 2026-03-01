import sys
import pandas as pd
import numpy as np
import joblib
import torch
import esm
import os

if len(sys.argv) < 2:
    print("Error: You must provide the input CSV filename.")
    print("Usage: python predict_marc.py your_file.csv")
    sys.exit(1)
input_csv = sys.argv[1]
model_file = "marc_model.pkl"  # Make sure this matches your saved model name
output_csv = "predicted_" + input_csv

if not os.path.exists(input_csv):
    print(f"Error: File '{input_csv}' not found.")
    sys.exit(1)

try:
    df = pd.read_csv(input_csv)
    
    # Smart column finding: looks for 'sequence', 'seq', or 'Sequence'
    # Case insensitive and ignores columns like 'Sequence Caution'
    seq_col = None
    for col in df.columns:
        if 'seq' in col.lower() and 'caution' not in col.lower():
            seq_col = col
            break
    
    if seq_col is None:
        raise ValueError("CSV must have a column named 'sequence', 'seq', or similar.")
    
    print(f"Found sequence column: '{seq_col}'")
    # Ensure sequences are strings and handle NaNs
    sequences = df[seq_col].astype(str).tolist()
    print(f"Loaded {len(sequences)} sequences.")

except Exception as e:
    print(f"Error loading CSV: {e}")
    sys.exit(1)

model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
batch_converter = alphabet.get_batch_converter()
model.eval()
def get_stable_embedding(sequence):
    """
    Converts sequence to fixed (1280,) vector.
    Handles variable lengths automatically.
    """
    # Truncate if too long (ESM limit is usually 1022 + 2 special tokens)
    if len(sequence) > 1022:
        sequence = sequence[:1022]

    data = [("protein1", sequence)]
    batch_labels, batch_strs, batch_tokens = batch_converter(data)

    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[33], return_contacts=False)

    token_representations = results["representations"][33]

    # Mean Pooling: Exclude start/end tokens, average the rest
    seq_len = (batch_tokens != alphabet.padding_idx).sum(1).item()

    # Safety for extremely short sequences (< 3 AAs)
    if seq_len < 3:
        # Return a zero vector or the mean of the whole thing
        return token_representations[0].mean(0).numpy()

    # Valid sequence: average from index 1 to end-1 (skipping start/end tokens)
    return token_representations[0, 1 : seq_len - 1].mean(0).numpy()
print("\nGenerating ESM features...")
print(" This might take a while depending on dataset size.")

X_embeddings = []
total = len(sequences)

for i, seq in enumerate(sequences):
    try:
        vec = get_stable_embedding(seq)
        X_embeddings.append(vec)
    except Exception as e:
        print(f"\nError at index {i}: {e}")
        # Fallback to avoid crash (returns vector of zeros)
        X_embeddings.append(np.zeros(1280))

X = np.array(X_embeddings)
print(f"\nDone! Feature matrix shape: {X.shape}")
clf = joblib.load(model_file)
probs = clf.predict_proba(X)
class_names = ['Calcium', 'Potassium', 'Sodium', 'nonToxin']
prob_matrix = np.column_stack([p[:, 1] for p in probs])
final_preds = []
confidences = []
THRESHOLD = 0.45
for row in prob_matrix:
    max_prob = np.max(row)
    argmax = np.argmax(row)
    
    confidences.append(max_prob)
    
    if max_prob < THRESHOLD:
        final_preds.append("No Prediction")
    else:
        final_preds.append(class_names[argmax])

# Add results to dataframe
df['MARC_Prediction'] = final_preds
df['Confidence_Score'] = confidences

# Save to new CSV
df.to_csv(output_csv, index=False)
print(f"Success! Predictions saved to: {output_csv}")
