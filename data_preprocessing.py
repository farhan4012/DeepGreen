import os
from Bio import SeqIO
import pandas as pd

# Define paths
POSITIVE_FILE = "data/positive_autophagy.fasta"
NEGATIVE_FILE = "data/negative_control.fasta"
OUTPUT_FILE = "dataset.csv"

def parse_fasta(file_path, label, limit=None):
    """
    Reads a FASTA file and returns a list of dictionaries with sequence and label.
    limit: Max number of sequences to read (to balance data).
    """
    data = []
    if not os.path.exists(file_path):
        print(f"❌ Error: File not found at {file_path}")
        return []
        
    sequences = list(SeqIO.parse(file_path, "fasta"))
    
    # If a limit is set, only take that many (e.g., first 100)
    if limit:
        sequences = sequences[:limit]
        
    print(f"📂 Reading {file_path}: Found {len(sequences)} sequences.")
    
    for seq_record in sequences:
        # Get the protein sequence as a string
        seq_str = str(seq_record.seq)
        
        # Basic cleanup: Remove 'X' (unknown amino acids) or very short sequences
        if "X" not in seq_str and len(seq_str) > 50:
            data.append({
                "ID": seq_record.id,
                "Sequence": seq_str,
                "Label": label  # 1 for Autophagy, 0 for Control
            })
            
    return data

# --- MAIN EXECUTION ---

# 1. Load Positive Data (Label = 1)
# We take ALL verified autophagy proteins.
positive_data = parse_fasta(POSITIVE_FILE, label=1)

# 2. Load Negative Data (Label = 0)
# We limit this to matches the positive count (or slightly more) to keep data balanced.
# If you have 40 positives, we'll take 60 negatives.
target_neg_count = len(positive_data) + 20 
negative_data = parse_fasta(NEGATIVE_FILE, label=0, limit=target_neg_count)

# 3. Combine them
all_data = positive_data + negative_data

# 4. Create DataFrame
df = pd.DataFrame(all_data)

# 5. Save to CSV
if not df.empty:
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\n✅ SUCCESS! Dataset saved to '{OUTPUT_FILE}'")
    print(f"📊 Total Samples: {len(df)}")
    print(f"   - Autophagy (1): {len(positive_data)}")
    print(f"   - Control (0): {len(negative_data)}")
    print("👉 Check your folder for 'dataset.csv'.")
else:
    print("\n❌ Error: No data found. Check your FASTA files!")