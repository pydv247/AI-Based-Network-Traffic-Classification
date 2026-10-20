import os
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
import torch
import torch.nn as nn

# 1. Define 1D-CNN Architecture
class Traffic1DCNN(nn.Module):
    def __init__(self, in_features, num_classes):
        super(Traffic1DCNN, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(4)
        )
        self.fc_classifier = nn.Sequential(
            nn.Linear(64 * 4, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.conv_block(x)
        x = x.view(x.size(0), -1)
        x = self.fc_classifier(x)
        return x

# 2. Load Pipeline Artifacts & Test Data
scaler = joblib.load("scaler.pkl")
encoder = joblib.load("label_encoder.pkl")
features = joblib.load("feature_names.pkl")
rf = joblib.load("rf_model.pkl")

df = pd.read_csv("test_traffic_sample.csv")
X_raw = scaler.transform(df[features])
y_true = df["class1"].values
y_true_idx = encoder.transform(y_true)

# Load 1D-CNN weights
cnn = Traffic1DCNN(len(features), len(encoder.classes_))
cnn.load_state_dict(torch.load("traffic_cnn_model.pth"))
cnn.eval()

# 3. Benchmark Random Forest
t0 = time.perf_counter()
rf_preds_idx = rf.predict(X_raw)
rf_latency = ((time.perf_counter() - t0) / len(df)) * 1000
rf_preds = encoder.inverse_transform(rf_preds_idx)
rf_acc = accuracy_score(y_true, rf_preds) * 100
rf_f1 = f1_score(y_true, rf_preds, average="macro") * 100
rf_size = os.path.getsize("rf_model.pkl") / (1024 * 1024) if os.path.exists("rf_model.pkl") else 0.0

# 4. Benchmark 1D-CNN
tensor_x = torch.tensor(X_raw, dtype=torch.float32).unsqueeze(1)
t0 = time.perf_counter()
with torch.no_grad():
    cnn_outputs = cnn(tensor_x)
    _, cnn_preds_idx = torch.max(cnn_outputs, 1)
cnn_latency = ((time.perf_counter() - t0) / len(df)) * 1000
cnn_acc = accuracy_score(y_true_idx, cnn_preds_idx.numpy()) * 100
cnn_f1 = f1_score(y_true_idx, cnn_preds_idx.numpy(), average="macro") * 100
cnn_size = os.path.getsize("traffic_cnn_model.pth") / (1024 * 1024) if os.path.exists("traffic_cnn_model.pth") else 0.0

# 5. Print Comparison Table
print("\n" + "=" * 70)
print("     HEAD-TO-HEAD MODEL BENCHMARK: REVIEW II vs. REVIEW III")
print("=" * 70)
print(f"{'Performance Metric':<24} | {'Random Forest (Baseline)':<24} | {'1D-CNN (Deep Learning)':<22}")
print("-" * 70)
print(f"{'Batch Accuracy (%)':<24} | {rf_acc:<24.2f} | {cnn_acc:<22.2f}")
print(f"{'Macro F1-Score (%)':<24} | {rf_f1:<24.2f} | {cnn_f1:<22.2f}")
print(f"{'Inference Latency':<24} | {f'{rf_latency:.3f} ms / flow':<24} | {f'{cnn_latency:.3f} ms / flow':<22}")
print(f"{'Model Size (Disk)':<24} | {f'{rf_size:.2f} MB':<24} | {f'{cnn_size:.2f} MB':<22}")
print("=" * 70)