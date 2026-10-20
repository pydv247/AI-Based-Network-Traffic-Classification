import os
import joblib
import numpy as np
import pandas as pd
from scipy.io import arff
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# 1. Search and collect all Scenario A2 ARFF files
print("[Step 1/7] Searching for dataset files...")
a2_files = []
for root, dirs, files in os.walk("data"):
  for file in files:
    if file.endswith(".arff") and "Scenario A2" in root:
      a2_files.append(os.path.join(root, file))

if not a2_files:
  raise FileNotFoundError(
      "No .arff files found in 'data/' directory under 'Scenario A2'."
  )

print(f"Found {len(a2_files)} files. Loading flows into memory...")

# 2. Read all files into DataFrames
df_list = []
for i, f in enumerate(a2_files, 1):
  print(f"  -> Loading ({i}/{len(a2_files)}): {os.path.basename(f)}")
  raw_data, _ = arff.loadarff(f)
  temp_df = pd.DataFrame(raw_data)
  df_list.append(temp_df)

# 3. Concatenate and clean data
print("[Step 2/7] Combining datasets...")
df = pd.concat(df_list, ignore_index=True)

print("[Step 3/7] Cleaning features and decoding categorical values...")
# Decode byte strings to normal text
for col in df.select_dtypes([object]):
  df[col] = df[col].str.decode("utf-8")

# Replace infinities and drop nulls
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)
print(
    f"Dataset ready. Remaining valid flows: {df.shape[0]}, Features:"
    f" {df.shape[1] - 1}"
)

# 4. Separate features (X) and target class (y)
print("[Step 4/7] Preparing labels and feature matrices...")
label_col = df.columns[-1]
X = df.drop(columns=[label_col])
y = df[label_col]

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# 5. Split train/test (80/20) and normalize features
print("[Step 5/7] Performing 80/20 train/test split and feature scaling...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 6. Train Random Forest Classifier
print("[Step 6/7] Training Random Forest baseline model...")
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train_scaled, y_train)

# 7. Evaluate
print("\n" + "=" * 60)
print("BASELINE MODEL EVALUATION RESULTS (Review II Deliverable)")
print("=" * 60)
y_pred = rf.predict(X_test_scaled)
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# 8. Save artifacts for Streamlit UI
print("[Step 7/7] Saving model artifacts to disk...")
joblib.dump(rf, "rf_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(label_encoder, "label_encoder.pkl")
joblib.dump(list(X.columns), "feature_names.pkl")

print(
    "\n[Success] 'rf_model.pkl', 'scaler.pkl', 'label_encoder.pkl', and"
    " 'feature_names.pkl' saved successfully!"
)