"""Hybrid ensemble: CNN-BiLSTM + tree models, combined by soft voting.

Reuses the CNN-BiLSTM trained by train_cnn_bilstm.py (no deep retraining).
The combination is chosen on the VALIDATION split; the TEST split is only
used at the end to report the final accuracy.

Usage (PowerShell, inside the project folder):
    python hybrid_ensemble.py VPN.arff NO-VPN.arff --label class1

Needs these files from train_cnn_bilstm.py:
    traffic_cnn_bilstm_ensemble.pth, dl_preprocessor.pkl
Writes:
    hybrid_models.pkl  (the chosen tree models and which parts to combine)
"""
import argparse
import itertools
import sys

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser()
parser.add_argument("data_files", nargs="+", help="one or more datasets")
parser.add_argument("--label", default="class1", help="label column")
args = parser.parse_args()

DATA_FILES = args.data_files
LABEL_COL = args.label
MAX_ROWS = 150000
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

feature_names = list(joblib.load("feature_names.pkl"))
label_encoder = joblib.load("label_encoder.pkl")
num_classes = len(label_encoder.classes_)


# ---------------- Load data (identical to train_cnn_bilstm.py) -----------
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
df = df[df[LABEL_COL].isin(set(label_encoder.classes_))]
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
      f"test: {len(X_te)}  (same split as training)")


# ---------------- CNN-BiLSTM (same class as the training script) ----------
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


pre = joblib.load("dl_preprocessor.pkl")
ckpt = torch.load("traffic_cnn_bilstm_ensemble.pth", map_location="cpu")
cnn_models = []
for st in ckpt["states"]:
    m = CNN_BiLSTM(**ckpt["config"]).to(device)
    m.load_state_dict(st)
    m.eval()
    cnn_models.append(m)
print(f"[+] Loaded {len(cnn_models)} CNN-BiLSTM model(s), "
      f"config: {ckpt['config']}")


def cnn_proba(X):
    feats = torch.tensor(
        np.asarray(pre.transform(X), dtype=np.float32)
    ).unsqueeze(1)
    out = np.zeros((len(feats), num_classes))
    with torch.no_grad():
        for i in range(0, len(feats), 512):
            xb = feats[i:i + 512].to(device)
            p = torch.zeros(xb.size(0), num_classes, device=device)
            for m in cnn_models:
                p += torch.softmax(m(xb), dim=1)
            out[i:i + 512] = (p / len(cnn_models)).cpu().numpy()
    return out


def full_proba(model, X):
    """predict_proba aligned to all classes (even if one is missing)."""
    p = model.predict_proba(X)
    out = np.zeros((len(X), num_classes))
    out[:, model.classes_] = p
    return out


# ---------------- Train tree models on the TRAIN split ----------------
trees = {
    "RandomForest": RandomForestClassifier(
        n_estimators=300, n_jobs=-1, random_state=42
    ),
    "ExtraTrees": ExtraTreesClassifier(
        n_estimators=300, n_jobs=-1, random_state=42
    ),
    "GradientBoosting": HistGradientBoostingClassifier(
        max_iter=200, learning_rate=0.08, random_state=42
    ),
}
probs = {"CNN-BiLSTM": (cnn_proba(X_val), cnn_proba(X_te))}
for name, mdl in trees.items():
    print(f"[+] Training {name} ...")
    mdl.fit(X_tr, y_tr)
    probs[name] = (full_proba(mdl, X_val), full_proba(mdl, X_te))


def acc(p, labels):
    return 100.0 * (p.argmax(1) == labels).mean()


print("\n[+] Individual models (validation | test):")
for name, (pv, pt) in probs.items():
    print(f"      {name:18s} {acc(pv, y_val):6.2f}% | {acc(pt, y_te):6.2f}%")

# ---------------- Choose the combination on VALIDATION only --------------
names = list(probs)
results = []
for r in range(1, len(names) + 1):
    for combo in itertools.combinations(names, r):
        pv = np.mean([probs[n][0] for n in combo], axis=0)
        results.append((acc(pv, y_val), len(combo), combo))
results.sort(key=lambda t: (t[0], t[1]), reverse=True)

print("\n[+] Combinations ranked by VALIDATION accuracy (top 6):")
for v, _, combo in results[:6]:
    print(f"      {v:6.2f}%  {' + '.join(combo)}")

best_val, _, best_combo = results[0]
pt = np.mean([probs[n][1] for n in best_combo], axis=0)
final_acc = acc(pt, y_te)
print(f"\n[RESULT] Chosen on validation: {' + '.join(best_combo)}")
print(f"[RESULT] Hybrid test accuracy: {final_acc:.2f}%")
print(f"[RESULT] CNN-BiLSTM alone    : {acc(probs['CNN-BiLSTM'][1], y_te):.2f}%")
print(f"[RESULT] Random Forest alone : {acc(probs['RandomForest'][1], y_te):.2f}%")

preds = pt.argmax(1)
present = np.unique(np.concatenate([y_te, preds]))
print("\n[+] Per-class report (hybrid) on the test split:")
print(
    classification_report(
        y_te,
        preds,
        labels=present,
        target_names=[label_encoder.classes_[i] for i in present],
        zero_division=0,
    )
)

joblib.dump(
    {
        "combo": list(best_combo),
        "trees": {n: trees[n] for n in best_combo if n in trees},
    },
    "hybrid_models.pkl",
    compress=3,
)
print("[+] Saved hybrid_models.pkl")