import pandas as pd
from collections import Counter

# 1. Load the clean data
input_file = "dataset.csv"
df = pd.read_csv(input_file)

print(f"📊 Loaded {len(df)} sequences.")

# 2. Define the 20 standard amino acids
amino_acids = "ACDEFGHIKLMNPQRSTVWY"

def get_amino_acid_composition(sequence):
    """
    Counts how often each amino acid appears and divides by sequence length.
    Returns a dictionary: {'A': 0.05, 'C': 0.01, ...}
    """
    seq_len = len(sequence)
    if seq_len == 0:
        return {aa: 0 for aa in amino_acids}
    
    # Count specific amino acids
    counts = Counter(sequence)
    
    # Calculate frequency (Count / Length)
    composition = {aa: counts.get(aa, 0) / seq_len for aa in amino_acids}
    
    return composition

# 3. Apply this math to every row
print("⚙️  Calculating features... (This might take a second)")

# This creates a new dataframe of just the math (20 columns)
features = df["Sequence"].apply(lambda seq: pd.Series(get_amino_acid_composition(seq)))

# 4. Combine Features with the Label
# We drop the 'ID' and 'Sequence' because the AI doesn't need them anymore.
final_dataset = pd.concat([features, df["Label"]], axis=1)

# 5. Save
output_file = "final_features.csv"
final_dataset.to_csv(output_file, index=False)

print(f"✅ DONE! Features saved to '{output_file}'")
print(f"   - Rows: {final_dataset.shape[0]}")
print(f"   - Columns: {final_dataset.shape[1]} (20 amino acids + 1 label)")
print("👉 You are ready for AI Training (Day 5).")