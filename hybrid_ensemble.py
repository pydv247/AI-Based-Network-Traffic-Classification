"""Hybrid ensemble v2: CNN-BiLSTM + tuned tree models on extra features.

Steps (the TEST split is only used once, at the very end):
  1. Add engineered ratio/range features for the tree models.
  2. Tune Gradient Boosting with cross-validation on the TRAIN split.
  3. Pick the best soft-voting combination on the VALIDATION split.
  4. Refit the chosen tree models on TRAIN+VALIDATION, then report TEST.

Usage (PowerShell, inside the project folder):
    python hybrid_ensemble.py VPN.arff NO-VPN.arff --label class1

Needs from train_cnn_bilstm.py:
    traffic_cnn_bilstm_ensemble.pth, dl_preprocessor.pkl
Writes:
    hybrid_models.pkl
"""
import argparse
import itertools
import sys

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.base import clone
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.metrics import classification_report
from sklearn.model_selection import RandomizedSearchCV, train_test_split

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

# ---------------- Engineered features (for the tree models) ---------------
RECIPE_ALL = [
    ("ratio", "fiat_biat_total", "total_fiat", "total_biat"),
    ("ratio", "fiat_biat_mean", "mean_fiat", "mean_biat"),
    ("ratio", "fiat_biat_max", "max_fiat", "max_biat"),
    ("ratio", "fiat_biat_min", "min_fiat", "min_biat"),
    ("ratio", "active_idle_mean", "mean_active", "mean_idle"),
    ("ratio", "bytes_per_packet", "flowBytesPerSecond", "flowPktsPerSecond"),
    ("ratio", "flowiat_cv", "std_flowiat", "mean_flowiat"),
    ("ratio", "active_cv", "std_active", "mean_active"),
    ("ratio", "idle_cv", "std_idle", "mean_idle"),
    ("ratio", "duration_per_flowiat", "duration", "mean_flowiat"),
    ("diff", "active_range", "max_active", "min_active"),
    ("diff", "idle_range", "max_idle", "min_idle"),
    ("diff", "flowiat_range", "max_flowiat", "min_flowiat"),
]
RECIPE = [r for r in RECIPE_ALL
          if r[2] in feature_names and r[3] in feature_names]
print(f"[+] Engineered features added for tree models: {len(RECIPE)}")


def apply_recipe(X, recipe):
    X = X.copy()
    for op, name, a, b in recipe:
        if op == "ratio":
            v = X[a].values / (np.abs(X[b].values) + 1e-6)
        else:
            v = X[a].values - X[b].values
        X[name] = np.clip(v, -1e12, 1e12)
    return X


Xe_tr = apply_recipe(X_tr, RECIPE)
Xe_val = apply_recipe(X_val, RECIPE)
Xe_te = apply_recipe(X_te, RECIPE)


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


def acc(p, labels):
    return 100.0 * (p.argmax(1) == labels).mean()


# ---------------- Tune Gradient Boosting (cross-validation on TRAIN) ------
print("[+] Tuning Gradient Boosting with 3-fold CV on the train split "
      "(a few minutes)...")
param_dist = {
    "learning_rate": [0.03, 0.05, 0.08, 0.12],
    "max_leaf_nodes": [15, 31, 63],
    "max_depth": [None, 6, 10],
    "l2_regularization": [0.0, 0.5, 1.0, 5.0],
    "min_samples_leaf": [10, 20, 40],
    "max_iter": [200, 350],
}
search = RandomizedSearchCV(
    HistGradientBoostingClassifier(random_state=42),
    param_dist,
    n_iter=10,
    cv=3,
    scoring="accuracy",
    random_state=42,
    n_jobs=1,
)
search.fit(Xe_tr, y_tr)
print(f"      best CV accuracy: {100 * search.best_score_:.2f}%")
print(f"      best settings   : {search.best_params_}")

trees = {
    "RandomForest": RandomForestClassifier(
        n_estimators=400, n_jobs=-1, random_state=42
    ),
    "ExtraTrees": ExtraTreesClassifier(
        n_estimators=400, n_jobs=-1, random_state=42
    ),
    "GradientBoosting": search.best_estimator_,
}

probs = {"CNN-BiLSTM": (cnn_proba(X_val), cnn_proba(X_te))}
for name, mdl in trees.items():
    print(f"[+] Training {name} ...")
    mdl.fit(Xe_tr, y_tr)
    probs[name] = (full_proba(mdl, Xe_val), full_proba(mdl, Xe_te))

print("\n[+] Individual models, trained on TRAIN only (validation | test):")
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
print(f"\n[+] Chosen on validation: {' + '.join(best_combo)}")

# ---------------- Refit chosen tree models on TRAIN + VALIDATION ---------
Xe_trval = pd.concat([Xe_tr, Xe_val])
y_trval = np.concatenate([y_tr, y_val])
final_trees = {}
for n in best_combo:
    if n in trees:
        print(f"[+] Refitting {n} on train+validation ...")
        final_trees[n] = clone(trees[n]).fit(Xe_trval, y_trval)

parts = []
for n in best_combo:
    if n == "CNN-BiLSTM":
        parts.append(probs[n][1])
    else:
        parts.append(full_proba(final_trees[n], Xe_te))
pt = np.mean(parts, axis=0)
final_acc = acc(pt, y_te)

before = np.mean([probs[n][1] for n in best_combo], axis=0)
print(f"\n[info] Same combination before refit (test): {acc(before, y_te):.2f}%")
print(f"[RESULT] Chosen on validation: {' + '.join(best_combo)}")
print(f"[RESULT] Hybrid v2 test accuracy: {final_acc:.2f}%")
print(f"[RESULT] CNN-BiLSTM alone       : {acc(probs['CNN-BiLSTM'][1], y_te):.2f}%")
print(f"[RESULT] Random Forest (raw)    : 82.63% (earlier baseline)")

preds = pt.argmax(1)
present = np.unique(np.concatenate([y_te, preds]))
print("\n[+] Per-class report (hybrid v2) on the test split:")
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
        "trees": final_trees,
        "recipe": RECIPE,
    },
    "hybrid_models.pkl",
    compress=3,
)
print("[+] Saved hybrid_models.pkl (includes the feature recipe)")