import time
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

print("=" * 60)
print("  TRAINING 1D-CNN NETWORK TRAFFIC CLASSIFIER (REVIEW III)")
print("=" * 60)

# 1. Load Preprocessing Artifacts
scaler = joblib.load("scaler.pkl")
encoder = joblib.load("label_encoder.pkl")
features = joblib.load("feature_names.pkl")
num_classes = len(encoder.classes_)
num_features = len(features)

print(f"[+] Loaded artifacts: {num_features} features, {num_classes} target classes.")

# 2. Load Evaluation / Training Samples
df = pd.read_csv("test_traffic_sample.csv")
label_col = "class1" if "class1" in df.columns else "label"

# Prepare dataset
X = scaler.transform(df[features])
y = encoder.transform(df[label_col])

# Expand dataset slightly to simulate batch sequence training
X = np.repeat(X, 10, axis=0)
y = np.repeat(y, 10, axis=0)

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Convert to PyTorch Tensors (Reshaped for 1D-CNN: [batch, channels=1, sequence_length])
train_tensor_x = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)
train_tensor_y = torch.tensor(y_train, dtype=torch.long)
test_tensor_x = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1)
test_tensor_y = torch.tensor(y_test, dtype=torch.long)

train_loader = DataLoader(
    TensorDataset(train_tensor_x, train_tensor_y), batch_size=16, shuffle=True
)


# 3. Define 1D-CNN Deep Learning Architecture
class Traffic1DCNN(nn.Module):

  def __init__(self, in_features, num_classes):
    super(Traffic1DCNN, self).__init__()
    self.conv_block = nn.Sequential(
        nn.Conv1d(
            in_channels=1,
            out_channels=32,
            kernel_size=3,
            stride=1,
            padding=1,
        ),
        nn.BatchNorm1d(32),
        nn.ReLU(),
        nn.MaxPool1d(kernel_size=2),
        nn.Conv1d(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1,
        ),
        nn.BatchNorm1d(64),
        nn.ReLU(),
        nn.AdaptiveAvgPool1d(4),
    )
    self.fc_classifier = nn.Sequential(
        nn.Linear(64 * 4, 128),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(128, num_classes),
    )

  def forward(self, x):
    x = self.conv_block(x)
    x = x.view(x.size(0), -1)
    x = self.fc_classifier(x)
    return x


model = Traffic1DCNN(num_features, num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.003, weight_decay=1e-4)

# 4. Train Model
print("\n[+] Initiating deep learning training epochs...")
epochs = 25
start_time = time.time()

for epoch in range(1, epochs + 1):
  model.train()
  total_loss = 0
  for batch_x, batch_y in train_loader:
    optimizer.zero_grad()
    outputs = model(batch_x)
    loss = criterion(outputs, batch_y)
    loss.backward()
    optimizer.step()
    total_loss += loss.item()

  if epoch % 5 == 0 or epoch == epochs:
    print(
        f"  Epoch [{epoch:02d}/{epochs:02d}] - Loss:"
        f" {total_loss / len(train_loader):.4f}"
    )

training_time = time.time() - start_time
print(f"[+] 1D-CNN Training finished in {training_time:.2f} seconds.")

# 5. Evaluate Deep Learning Model
model.eval()
with torch.no_grad():
  test_outputs = model(test_tensor_x)
  _, preds = torch.max(test_outputs, 1)

preds_np = preds.numpy()
y_test_np = test_tensor_y.numpy()

cnn_acc = accuracy_score(y_test_np, preds_np) * 100
cnn_f1 = f1_score(y_test_np, preds_np, average="macro") * 100

print("\n" + "=" * 50)
print("     1D-CNN BENCHMARK EVALUATION RESULTS")
print("=" * 50)
print(f" Test Accuracy   : {cnn_acc:.2f}%")
print(f" Macro F1-Score  : {cnn_f1:.2f}%")
print("=" * 50)

# 6. Save Model Weights Artifact
torch.save(model.state_dict(), "traffic_cnn_model.pth")
print("\n[+] Saved trained deep learning weights to 'traffic_cnn_model.pth'!")