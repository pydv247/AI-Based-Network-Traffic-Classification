import json
import os

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn

# Page Configuration
st.set_page_config(
    page_title="AI-Based Network Traffic Classifier",
    page_icon="🛡️",
    layout="wide",
)

SAMPLE_FILES = ["test_labeled_evaluation.csv", "test_traffic_sample.csv"]
PRED_COLUMNS = {
    "rf": "RandomForest_Prediction",
    "cnn": "CNN_BiLSTM_Prediction",
    "hybrid": "Hybrid_Prediction",
}


def show_df(df):
    """st.dataframe that works on old and new Streamlit versions."""
    try:
        st.dataframe(df, width="stretch")
    except TypeError:
        st.dataframe(df, use_container_width=True)


# ---------------------------------------------------------------- models
class CNN_BiLSTM(nn.Module):
    """Same architecture as train_cnn_bilstm.py (settings come from the
    saved checkpoint, so the app can never drift out of sync)."""

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
    """Engineered features used by the tree models (same as training)."""
    X = X.copy()
    for op, name, a, b in recipe:
        if op == "ratio":
            v = X[a].values / (np.abs(X[b].values) + 1e-6)
        else:
            v = X[a].values - X[b].values
        X[name] = np.clip(v, -1e12, 1e12)
    return X


@st.cache_resource(show_spinner="Loading models...")
def load_artifacts():
    art = {"status": []}
    art["feature_names"] = list(joblib.load("feature_names.pkl"))
    art["label_encoder"] = joblib.load("label_encoder.pkl")
    status = art["status"]

    # Random Forest baseline
    art["rf"] = None
    try:
        if os.path.exists("rf_baseline.pkl"):
            art["rf"] = {"model": joblib.load("rf_baseline.pkl"),
                         "scaler": None}
            status.append(("Random Forest baseline", True, "rf_baseline.pkl"))
        else:
            art["rf"] = {"model": joblib.load("rf_model.pkl"),
                         "scaler": joblib.load("scaler.pkl")}
            status.append((
                "Random Forest baseline", True,
                "legacy rf_model.pkl (run compute_benchmarks.py to create "
                "rf_baseline.pkl)",
            ))
    except Exception as exc:
        status.append(("Random Forest baseline", False, str(exc)))

    # CNN-BiLSTM
    art["cnn"] = None
    try:
        pre = joblib.load("dl_preprocessor.pkl")
        if os.path.exists("traffic_cnn_bilstm_ensemble.pth"):
            ckpt = torch.load("traffic_cnn_bilstm_ensemble.pth",
                              map_location="cpu")
            cfg, states = ckpt["config"], ckpt["states"]
        else:
            ckpt = torch.load("traffic_cnn_bilstm_model.pth",
                              map_location="cpu")
            cfg, states = ckpt["config"], [ckpt["state"]]
        nets = []
        for st_dict in states:
            net = CNN_BiLSTM(**cfg)
            net.load_state_dict(st_dict)
            net.eval()
            nets.append(net)
        art["cnn"] = {"pre": pre, "nets": nets}
        status.append(("1D-CNN + BiLSTM", True,
                       f"{len(nets)} model(s) loaded"))
    except Exception as exc:
        status.append((
            "1D-CNN + BiLSTM", False,
            f"{exc} - retrain with the latest train_cnn_bilstm.py",
        ))

    # Hybrid ensemble
    art["hybrid"] = None
    try:
        hy = joblib.load("hybrid_models.pkl")
        art["hybrid"] = {"combo": list(hy["combo"]), "trees": hy["trees"],
                         "recipe": hy["recipe"]}
        status.append(("Hybrid ensemble", True,
                       " + ".join(art["hybrid"]["combo"])))
    except Exception as exc:
        status.append(("Hybrid ensemble", False, str(exc)))

    # Measured benchmark numbers
    art["bench"] = None
    if os.path.exists("benchmarks.json"):
        try:
            with open("benchmarks.json", encoding="utf-8") as fh:
                art["bench"] = json.load(fh)
        except Exception as exc:
            status.append(("benchmarks.json", False, str(exc)))

    # Real labeled samples for the single-flow demo
    art["sample"] = None
    label_col = (art["bench"] or {}).get("meta", {}).get("label_col", "class1")
    art["label_col"] = label_col
    for fname in SAMPLE_FILES:
        if os.path.exists(fname):
            try:
                sample = pd.read_csv(fname)
                sample.columns = [c.strip() for c in sample.columns]
                needed = art["feature_names"] + [label_col]
                if all(c in sample.columns for c in needed):
                    sample = sample[needed].replace(
                        [np.inf, -np.inf], np.nan).dropna().head(500)
                    sample = sample.reset_index(drop=True)
                    if len(sample):
                        art["sample"] = sample
                        break
            except Exception:
                pass
    return art


art = load_artifacts()
feature_names = art["feature_names"]
label_encoder = art["label_encoder"]
classes = list(label_encoder.classes_)
n_classes = len(classes)


def full_proba(model, X):
    p = model.predict_proba(X)
    out = np.zeros((len(X), n_classes))
    out[:, model.classes_] = p
    return out


def cnn_proba(X):
    cnn = art["cnn"]
    feats = torch.tensor(
        np.asarray(cnn["pre"].transform(X), dtype=np.float32)
    ).unsqueeze(1)
    out = np.zeros((len(feats), n_classes))
    with torch.no_grad():
        for i in range(0, len(feats), 1024):
            xb = feats[i:i + 1024]
            p = torch.zeros(xb.size(0), n_classes)
            for net in cnn["nets"]:
                p += torch.softmax(net(xb), dim=1)
            out[i:i + 1024] = (p / len(cnn["nets"])).numpy()
    return out


def rf_proba(X):
    rf = art["rf"]
    Xin = X if rf["scaler"] is None else rf["scaler"].transform(X)
    return full_proba(rf["model"], Xin)


def predict_all(X):
    """Return {model_key: probability matrix} for every available model."""
    probs = {}
    if art["rf"]:
        probs["rf"] = rf_proba(X)
    if art["cnn"]:
        probs["cnn"] = cnn_proba(X)
    if art["hybrid"]:
        hy = art["hybrid"]
        parts, Xe = [], None
        for name in hy["combo"]:
            if name == "CNN-BiLSTM":
                parts.append(probs["cnn"] if "cnn" in probs else cnn_proba(X))
            else:
                if Xe is None:
                    Xe = apply_recipe(X, hy["recipe"])
                parts.append(full_proba(hy["trees"][name], Xe))
        probs["hybrid"] = np.mean(parts, axis=0)
    return probs


MODEL_LABELS = {}
if art["hybrid"]:
    MODEL_LABELS["hybrid"] = (
        "Hybrid Ensemble (" + " + ".join(art["hybrid"]["combo"]) + ")"
    )
if art["cnn"]:
    MODEL_LABELS["cnn"] = "1D-CNN + BiLSTM (Deep Learning)"
if art["rf"]:
    MODEL_LABELS["rf"] = "Random Forest (Baseline)"

# ---------------------------------------------------------------- sidebar
st.sidebar.header("Model Configuration")
if not MODEL_LABELS:
    st.sidebar.error("No trained model could be loaded. See details below.")
COMPARE = "Compare All Models (Side-by-Side)"
model_choice = st.sidebar.selectbox(
    "Select Architecture", [COMPARE] + list(MODEL_LABELS.values())
)

st.sidebar.markdown("---")
st.sidebar.info(
    "Review 3 Evaluation: Comparing spatial-temporal deep learning against"
    " tree-based baselines and a hybrid ensemble."
)
with st.sidebar.expander("Model status"):
    for name, ok, msg in art["status"]:
        (st.success if ok else st.error)(f"**{name}** - {msg}")

if not MODEL_LABELS:
    st.error("No trained models are available, so nothing can be classified.")
    for name, ok, msg in art["status"]:
        if not ok:
            st.write(f"- {name}: {msg}")
    st.stop()

# ---------------------------------------------------------------- header
st.title("🛡️ AI-Based Network Traffic Classifier")
st.markdown(
    "**Review 3 Dashboard:** Encrypted traffic classification, measured"
    " performance benchmarking, and model comparison."
)

tab1, tab2, tab3 = st.tabs(
    [
        "📊 Comparative Benchmarks",
        "🔍 Single-Flow Inspection",
        "📁 Batch Evaluation & CSV",
    ]
)

# ---------------------------------------------------------------- tab 1
with tab1:
    st.subheader("Model Performance & Benchmark Analysis")
    bench = art["bench"]
    if bench is None:
        st.warning(
            "No measured benchmark results found (benchmarks.json). Run "
            "`python compute_benchmarks.py <dataset files> --label class1` "
            "once to generate them. No placeholder numbers are shown."
        )
    else:
        meta, models = bench["meta"], bench["models"]
        st.caption(
            f"Measured on a held-out test split of {meta['test_rows']} flows "
            f"(train {meta['train_rows']}, validation {meta['val_rows']}). "
            + meta.get("note", "")
        )
        base = models.get("rf")
        cols = st.columns(len(models))
        for col, (key, m) in zip(cols, models.items()):
            delta = None
            if base is not None and key != "rf":
                delta = f"{m['accuracy'] - base['accuracy']:+.2f}% vs baseline"
            col.metric(m["name"] + " - Accuracy", f"{m['accuracy']:.2f}%",
                       delta)

        summary = pd.DataFrame(
            [
                {
                    "Model": m["name"],
                    "Accuracy (%)": m["accuracy"],
                    "Macro F1 (%)": m["macro_f1"],
                    "Weighted F1 (%)": m["weighted_f1"],
                    "Batch latency (ms/flow, CPU)": m["latency_ms_per_flow"],
                }
                for m in models.values()
            ]
        )
        show_df(summary)

        st.markdown("---")
        st.markdown("### Confusion Matrix (measured)")
        st.write(
            "Rows are the true class and columns the predicted class, "
            "computed on the test split."
        )
        pick = st.selectbox(
            "Model", list(models.keys()),
            format_func=lambda k: models[k]["name"], key="cm_model",
        )
        cm = pd.DataFrame(models[pick]["confusion"], index=meta["classes"],
                          columns=meta["classes"])
        show_df(cm)

        st.markdown("### Per-class F1 score")
        f1_table = pd.DataFrame(
            {
                m["name"]: {c: v["f1"] for c, v in m["per_class"].items()}
                for m in models.values()
            }
        )
        any_model = next(iter(models.values()))
        f1_table["Test support"] = pd.Series(
            {c: v["support"] for c, v in any_model["per_class"].items()}
        )
        show_df(f1_table.round(3))

# ---------------------------------------------------------------- tab 2
with tab2:
    st.subheader("Interactive Single-Flow Packet Inspection")
    st.write(
        "Enter raw flow features (the same units as the dataset), or "
        "prefill them from a real labeled sample, then classify."
    )

    sample = art["sample"]

    def _apply_sample():
        idx = st.session_state.get("sample_pick", 0)
        if sample is None or idx == 0:
            return
        row = sample.iloc[idx - 1]
        for i, feat in enumerate(feature_names):
            st.session_state[f"feat_{i}"] = float(np.nan_to_num(row[feat]))
        st.session_state["true_label"] = str(row[art["label_col"]])

    if "feat_0" not in st.session_state:
        for i, feat in enumerate(feature_names):
            st.session_state[f"feat_{i}"] = (
                float(np.nan_to_num(sample.iloc[0][feat]))
                if sample is not None else 0.0
            )
        st.session_state["true_label"] = (
            str(sample.iloc[0][art["label_col"]]) if sample is not None
            else None
        )
        st.session_state["sample_pick"] = 1 if sample is not None else 0

    if sample is not None:
        st.selectbox(
            "Prefill from a labeled sample row",
            list(range(len(sample) + 1)),
            format_func=lambda i: (
                "(enter values manually)" if i == 0
                else f"Row {i} - true class: {sample.iloc[i - 1][art['label_col']]}"
            ),
            key="sample_pick",
            on_change=_apply_sample,
        )

    input_vals = []
    cols = st.columns(3)
    for i, feat in enumerate(feature_names):
        with cols[i % 3]:
            val = st.number_input(feat, key=f"feat_{i}", format="%.4f")
            input_vals.append(val)

    if st.button("Classify Flow"):
        X_input = pd.DataFrame([input_vals], columns=feature_names)
        probs = predict_all(X_input.astype(float))

        def summarize(key):
            p = probs[key][0]
            idx = int(np.argmax(p))
            return classes[idx], 100.0 * float(p[idx])

        true_label = st.session_state.get("true_label")
        if true_label and st.session_state.get("sample_pick", 0) != 0:
            st.caption(f"True class of the selected sample: `{true_label}`")

        if model_choice == COMPARE:
            keys = list(MODEL_LABELS.keys())
            for col, key in zip(st.columns(len(keys)), keys):
                label, conf = summarize(key)
                with col:
                    st.markdown(f"**{MODEL_LABELS[key]}**")
                    st.markdown(f"Predicted class: `{label}`")
                    st.markdown(f"Confidence: `{conf:.2f}%`")
        else:
            key = next(k for k, v in MODEL_LABELS.items() if v == model_choice)
            label, conf = summarize(key)
            st.success(
                f"**{MODEL_LABELS[key]} result:** `{label}` "
                f"(confidence `{conf:.2f}%`)"
            )
            top = np.argsort(probs[key][0])[::-1][:3]
            show_df(
                pd.DataFrame(
                    {
                        "Class": [classes[j] for j in top],
                        "Probability (%)": [
                            round(100.0 * float(probs[key][0][j]), 2)
                            for j in top
                        ],
                    }
                )
            )

# ---------------------------------------------------------------- tab 3
with tab3:
    st.subheader("Batch CSV Evaluation & Logging")
    uploaded_file = st.file_uploader(
        "Upload Network Traffic CSV", type=["csv"]
    )

    if uploaded_file is None:
        st.info("Upload a CSV file to run batch evaluation.")
    else:
        try:
            batch_df = pd.read_csv(uploaded_file)
            batch_df.columns = [c.strip() for c in batch_df.columns]
            st.write(
                f"Successfully loaded batch dataset with `{len(batch_df)}` "
                "rows."
            )

            missing = [c for c in feature_names if c not in batch_df.columns]
            if missing:
                st.error(f"The CSV is missing these feature columns: {missing}")
            else:
                X_all = (
                    batch_df[feature_names]
                    .apply(pd.to_numeric, errors="coerce")
                    .replace([np.inf, -np.inf], np.nan)
                )
                valid = X_all.notna().all(axis=1).values
                if valid.sum() == 0:
                    st.error("No row has a complete set of numeric features.")
                else:
                    if (~valid).any():
                        st.warning(
                            f"{int((~valid).sum())} row(s) with missing or "
                            "invalid values were skipped (shown as n/a)."
                        )
                    probs = predict_all(X_all[valid].astype(float))
                    result = batch_df.copy()
                    for key, proba in probs.items():
                        col = np.full(len(batch_df), "n/a", dtype=object)
                        col[valid] = label_encoder.inverse_transform(
                            proba.argmax(1)
                        )
                        result[PRED_COLUMNS[key]] = col

                    label_col = art["label_col"]
                    if label_col in batch_df.columns:
                        truth = batch_df[label_col].astype(str).values
                        known = valid & np.isin(truth, classes)
                        if known.sum() > 0:
                            st.markdown("**Accuracy on this file** "
                                        f"(`{label_col}` column, "
                                        f"{int(known.sum())} rows)")
                            st.caption(
                                "Only meaningful for flows the models were "
                                "not trained on. Rows that came from the "
                                "training data will score unrealistically "
                                "high."
                            )
                            accs = st.columns(len(probs))
                            for ac, (key, proba) in zip(accs, probs.items()):
                                pred_all = np.full(len(batch_df), "n/a",
                                                   dtype=object)
                                pred_all[valid] = label_encoder.inverse_transform(
                                    proba.argmax(1))
                                a = 100.0 * float(
                                    (pred_all[known] == truth[known]).mean())
                                ac.metric(MODEL_LABELS[key], f"{a:.2f}%")

                    show_df(result.head(10))
                    st.download_button(
                        "Download all predictions (CSV)",
                        result.to_csv(index=False).encode("utf-8"),
                        file_name="predictions.csv",
                        mime="text/csv",
                    )
                    st.success("[✓] Batch inference complete with "
                               "side-by-side model predictions!")
        except Exception as exc:
            st.error(f"Batch evaluation failed: {exc}")