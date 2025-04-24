import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

# Load data
chunk_size = 10000  # You can adjust this value based on your file size
scaler = StandardScaler()
model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')

# Initialize empty lists to store data
X_list = []
y_list = []

# Read the CSV file in chunks
chunks = pd.read_csv('data/creditcard.csv', chunksize=chunk_size)

for chunk in chunks:
    # Print the columns to check if 'Amount' exists
    print(f"Columns in the current chunk: {chunk.columns.tolist()}")
    
    # Check if 'Amount' column exists before scaling
    if 'Amount' in chunk.columns:
        chunk['Amount'] = scaler.fit_transform(chunk[['Amount']])
    else:
        print("'Amount' column not found in this chunk.")

    # Check if 'Time' column exists before scaling
    if 'Time' in chunk.columns:
        chunk['Time'] = scaler.fit_transform(chunk[['Time']])
    else:
        print("'Time' column not found in this chunk.")

    # Split features and target
    X_chunk = chunk.drop('Class', axis=1, errors='ignore')  # Ignore errors if 'Class' column is missing
    y_chunk = chunk.get('Class', pd.Series())  # Use pd.Series() if 'Class' is missing

    # Append the chunks to the list
    X_list.append(X_chunk)
    y_list.append(y_chunk)

# Concatenate the chunks into a single DataFrame
X = pd.concat(X_list, ignore_index=True)
y = pd.concat(y_list, ignore_index=True)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Train model
model.fit(X_train, y_train)

# Save model
joblib.dump(model, 'model/fraud_model.pkl')

print("Model training complete and saved.")
