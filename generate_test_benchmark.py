import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score

# 1. Load trained model artifacts
print("Loading model pipeline artifacts...")
rf = joblib.load("rf_model.pkl")
scaler = joblib.load("scaler.pkl")
encoder = joblib.load("label_encoder.pkl")
features = joblib.load("feature_names.pkl")

# 2. Load dataset files
vpn_file = "TimeBasedFeatures-Dataset-120s-VPN.csv"
novpn_file = "TimeBasedFeatures-Dataset-120s-NO-VPN.csv"

vpn_df = pd.read_csv(vpn_file)
novpn_df = pd.read_csv(novpn_file)

# Standardize column naming
vpn_df.columns = vpn_df.columns.str.strip()
novpn_df.columns = novpn_df.columns.str.strip()

label_col = "class1" if "class1" in vpn_df.columns else "label"

# Prefix VPN classes to ensure 14 distinct target classes
vpn_df[label_col] = vpn_df[label_col].apply(
    lambda x: f"VPN-{x}" if not str(x).startswith("VPN-") else str(x)
)

full_df = pd.concat([vpn_df, novpn_df], ignore_index=True)
full_df = full_df.replace([np.inf, -np.inf], np.nan).dropna(
    subset=features + [label_col]
)

# 3. Sample 10 representative flows per class (140 total flows)
SAMPLE_PER_CLASS = 10
sampled_records = []

for label, group in full_df.groupby(label_col):
  n_samples = min(len(group), SAMPLE_PER_CLASS)
  # Use random_state=42 for deterministic benchmark testing
  sampled_records.append(group.sample(n=n_samples, random_state=42))

test_batch = (
    pd.concat(sampled_records, ignore_index=True)
    .sample(frac=1.0, random_state=42)
    .reset_index(drop=True)
)

# Save the generated test batch CSV with true labels included
export_cols = features + [label_col]
output_filename = "test_benchmark_140.csv"
test_batch[export_cols].to_csv(output_filename, index=False)
print(f"\n[+] Created benchmark file: '{output_filename}' ({len(test_batch)} flows)")

# 4. Measure Inference Efficiency & Accuracy
scaled_X = scaler.transform(test_batch[features])
predictions_idx = rf.predict(scaled_X)
predictions_label = encoder.inverse_transform(predictions_idx)
true_labels = test_batch[label_col].values

acc = accuracy_score(true_labels, predictions_label) * 100
macro_f1 = f1_score(true_labels, predictions_label, average="macro") * 100

print("\n" + "=" * 55)
print("              EVALUATION EFFICIENCY REPORT")
print("=" * 55)
print(f" Total Test Flows Evaluated : {len(test_batch)}")
print(f" Correct Classifications   : {(true_labels == predictions_label).sum()} / {len(test_batch)}")
print(f" Overall Batch Accuracy    : {acc:.2f}%")
print(f" Macro F1-Score            : {macro_f1:.2f}%")
print("=" * 55)
print("\nPer-Class Breakdown:")
print(classification_report(true_labels, predictions_label, digits=2))