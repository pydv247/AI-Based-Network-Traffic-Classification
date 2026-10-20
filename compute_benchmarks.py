"""Compute REAL benchmark numbers for the Streamlit app.

Run this once after training (train_cnn_bilstm.py and hybrid_ensemble.py).
It uses the SAME train/validation/test split as the training scripts and
evaluates every model on the untouched test split.

Usage (PowerShell, inside the project folder):
    python compute_benchmarks.py VPN.arff NO-VPN.arff --label class1

Needs:
    feature_names.pkl, label_encoder.pkl, dl_preprocessor.pkl,
    traffic_cnn_bilstm_ensemble.pth, hybrid_models.pkl
Writes:
    benchmarks.json   (accuracy, macro F1, confusion matrix, latency)
    rf_baseline.pkl   (Random Forest baseline trained on the TRAIN split)
"""
import argparse
import json
import sys
import time

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser()
parser.add_argument("data_files", nargs="+", help="one or more datasets")
parser.add_argument("--label", default="class1", help="label column")
args = parser.parse_args()

DATA_FILES = args.data_files
LABEL_COL = args.label
MAX_ROWS = 150000
device = torch.device("cpu")

feature_names = list(joblib.load("feature_names.pkl"))
label_encoder = joblib.load("label_encoder.pkl")
classes = list(label_encoder.classes_)
num_classes = len(classes)


# ---------------- Load data (identical to the training scripts) ----------
def load_table(path):
    if path.lower().endswith(".arff"):
        from scipy.io import arff

        data, _ = arff.loadarff(path)
        table = pd.DataFrame(data)
        for col in table.columns:
            if table[col].dtype == object:
                table[col] = table[col].apply(
                    lambda v: v.decode("utf-8") if isinstance(v, bytes) else v
                )
        return table
    return pd.read_csv(path)


def norm(name):
    return str(name).strip().strip("'\"").strip().lower()


frames = []
for path in DATA_FILES:
    print(f"[+] Loading {path} ...")
    part = load_table(path)
    part.columns = [c.strip() for c in part.columns]
    frames.append(part)
df = pd.concat(frames, ignore_index=True)

lookup = {norm(c): c for c in df.columns}
rename = {}
for wanted in feature_names + [LABEL_COL]:
    actual = lookup.get(norm(wanted))
    if actual is not None and actual != wanted:
        rename[actual] = wanted
df = df.rename(columns=rename)

missing = [c for c in feature_names + [LABEL_COL] if c not in df.columns]
if missing:
    print(f"[!] Columns not found: {missing}")
    print(list(df.columns))
    sys.exit(1)

df = df[feature_names + [LABEL_COL]].copy()
df = df.replace([np.inf, -np.inf], np.nan).dropna()
df[LABEL_COL] = df[LABEL_COL].astype(str)
df = df[df[LABEL_COL].isin(set(classes))]
if len(df) > MAX_ROWS:
    df = df.sample(MAX_ROWS, random_state=42)

raw = df[feature_names].astype(np.float64)
y = label_encoder.transform(df[LABEL_COL]).astype(np.int64)


def split(X, labels, test_size):
    c = np.bincount(labels)
    strat = labels if c[c > 0].min() >= 2 else None
    return train_test_split(
        X, labels, test_size=test_size, stratify=strat, random_state=42
    )


X_tmp, X_te, y_tmp, y_te = split(raw, y, 0.15)
X_tr, X_val, y_tr, y_val = split(X_tmp, y_tmp, 0.1765)
print(f"[+] Split sizes - train: {len(X_tr)}, val: {len(X_val)}, "
      f"test: {len(X_te)}")


# ---------------- CNN-BiLSTM (same class as the training script) ---------
class CNN_BiLSTM(nn.Module):

    def __init__(self, num_classes=14, num_features=21, arch="plain",
                 embed_dim=32, pool="last"):
        super().__init__()
        self.arch = arch
        self.pool = pool
        in_channels = 1
        if arch == "tokens":
            self.tok_w = nn.Parameter(torch.randn(num_features, embed_dim) * 0.1)
            self.tok_b = nn.Parameter(torch.zeros(num_features, embed_dim))
            in_channels = embed_dim
        self.conv_block = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=1),
            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        self.bilstm = nn.LSTM(
            input_size=128,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
        )
        self.classifier = nn.Sequential(
            nn.Linear(128 * 2 * (2 if pool == "meanmax" else 1), 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        if self.arch == "tokens":
            tokens = x.squeeze(1).unsqueeze(-1) * self.tok_w + self.tok_b
            x = tokens.permute(0, 2, 1)
        out = self.conv_block(x)
        out = out.permute(0, 2, 1)
        lstm_out, _ = self.bilstm(out)
        if self.pool == "meanmax":
            feats = torch.cat(
                [lstm_out.mean(dim=1), lstm_out.max(dim=1).values], dim=1
            )
        else:
            feats = lstm_out[:, -1, :]
        return self.classifier(feats)


def apply_recipe(X, recipe):
    X = X.copy()
    for op, name, a, b in recipe:
        if op == "ratio":
            v = X[a].values / (np.abs(X[b].values) + 1e-6)
        else:
            v = X[a].values - X[b].values
        X[name] = np.clip(v, -1e12, 1e12)
    return X


def full_proba(model, X):
    p = model.predict_proba(X)
    out = np.zeros((len(X), num_classes))
    out[:, model.classes_] = p
    return out


pre = joblib.load("dl_preprocessor.pkl")
ckpt = torch.load("traffic_cnn_bilstm_ensemble.pth", map_location="cpu")
cnn_models = []
for st in ckpt["states"]:
    m = CNN_BiLSTM(**ckpt["config"])
    m.load_state_dict(st)
    m.eval()
    cnn_models.append(m)
print(f"[+] Loaded {len(cnn_models)} CNN-BiLSTM model(s)")


def cnn_proba(X):
    feats = torch.tensor(
        np.asarray(pre.transform(X), dtype=np.float32)
    ).unsqueeze(1)
    out = np.zeros((len(feats), num_classes))
    with torch.no_grad():
        for i in range(0, len(feats), 512):
            xb = feats[i:i + 512]
            p = torch.zeros(xb.size(0), num_classes)
            for m in cnn_models:
                p += torch.softmax(m(xb), dim=1)
            out[i:i + 512] = (p / len(cnn_models)).numpy()
    return out


hy = joblib.load("hybrid_models.pkl")
combo = list(hy["combo"])
trees = hy["trees"]
recipe = hy["recipe"]
print(f"[+] Hybrid combination: {' + '.join(combo)}")


def hybrid_proba(X):
    parts = []
    Xe = None
    for n in combo:
        if n == "CNN-BiLSTM":
            parts.append(cnn_proba(X))
        else:
            if Xe is None:
                Xe = apply_recipe(X, recipe)
            parts.append(full_proba(trees[n], Xe))
    return np.mean(parts, axis=0)


# ---------------- Random Forest baseline (TRAIN split only) --------------
print("[+] Training the Random Forest baseline on the train split ...")
rf = RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=42)
rf.fit(X_tr, y_tr)
joblib.dump(rf, "rf_baseline.pkl", compress=3)
print("[+] Saved rf_baseline.pkl")


def rf_proba(X):
    return full_proba(rf, X)


# ---------------- Evaluate on the untouched TEST split --------------------
def time_ms_per_flow(fn, X, repeats=3):
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn(X)
        best = min(best, time.perf_counter() - t0)
    return 1000.0 * best / len(X)


candidates = {
    "rf": ("Random Forest (Baseline)", rf_proba),
    "cnn": ("1D-CNN + BiLSTM", cnn_proba),
    "hybrid": ("Hybrid Ensemble (" + " + ".join(combo) + ")", hybrid_proba),
}

label_ids = list(range(num_classes))
results = {}
for key, (name, fn) in candidates.items():
    print(f"[+] Evaluating {name} ...")
    proba = fn(X_te)
    pred = proba.argmax(1)
    prec, rec, f1, sup = precision_recall_fscore_support(
        y_te, pred, labels=label_ids, zero_division=0
    )
    results[key] = {
        "name": name,
        "accuracy": round(100.0 * accuracy_score(y_te, pred), 2),
        "macro_f1": round(
            100.0 * f1_score(y_te, pred, average="macro", zero_division=0), 2
        ),
        "weighted_f1": round(
            100.0 * f1_score(y_te, pred, average="weighted", zero_division=0),
            2,
        ),
        "latency_ms_per_flow": round(time_ms_per_flow(fn, X_te), 4),
        "per_class": {
            classes[i]: {
                "precision": round(float(prec[i]), 4),
                "recall": round(float(rec[i]), 4),
                "f1": round(float(f1[i]), 4),
                "support": int(sup[i]),
            }
            for i in label_ids
        },
        "confusion": confusion_matrix(y_te, pred, labels=label_ids).tolist(),
    }

out = {
    "meta": {
        "label_col": LABEL_COL,
        "classes": classes,
        "train_rows": int(len(X_tr)),
        "val_rows": int(len(X_val)),
        "test_rows": int(len(X_te)),
        "data_files": DATA_FILES,
        "note": (
            "All models evaluated on the same held-out test split. The "
            "Random Forest baseline is trained on the train split; the "
            "hybrid trees were refit on train+validation."
        ),
    },
    "models": results,
}
with open("benchmarks.json", "w", encoding="utf-8") as fh:
    json.dump(out, fh, indent=2)

print("\n[+] Test-split results")
print(f"{'Model':52s} {'Acc %':>7s} {'MacroF1 %':>10s} {'ms/flow':>9s}")
for r in results.values():
    print(f"{r['name']:52s} {r['accuracy']:7.2f} {r['macro_f1']:10.2f} "
          f"{r['latency_ms_per_flow']:9.4f}")
print("\n[+] Saved benchmarks.json")