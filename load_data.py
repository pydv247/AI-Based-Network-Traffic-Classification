import os
import pandas as pd
from scipy.io import arff

# Recursively find all ARFF files inside the data directory
all_arff_files = []
for root, dirs, files in os.walk("data"):
    for file in files:
        if file.endswith(".arff"):
            all_arff_files.append(os.path.join(root, file))

print(f"Total .arff files found: {len(all_arff_files)}")

# Pick the first Scenario A2 file
a2_files = [f for f in all_arff_files if "Scenario A2" in f]
file_to_load = a2_files[0] if a2_files else all_arff_files[0]

print(f"\nLoading file: {file_to_load} ...")

# Read and convert ARFF to DataFrame
raw_data, meta = arff.loadarff(file_to_load)
df = pd.DataFrame(raw_data)

# Decode byte strings into standard text
for col in df.select_dtypes([object]):
    df[col] = df[col].str.decode("utf-8")

print("\n--- Dataset Summary ---")
print(f"Total Rows: {df.shape[0]}, Total Columns: {df.shape[1]}")

print("\nFirst 3 rows:")
print(df.head(3))

label_col = df.columns[-1]
print(f"\nTarget Column Name: '{label_col}'")
print("\nTraffic category distribution:")
print(df[label_col].value_counts())