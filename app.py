import streamlit as st
import pandas as pd
import joblib
from collections import Counter

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="DeepGreen Predictor",
    page_icon="🌿",
    layout="centered"
)


# @st.cache_resource so it doesn't reload the model every time a button is clicked
@st.cache_resource
def load_models():
    model = joblib.load("autophagy_model.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

try:
    model, scaler = load_models()
except FileNotFoundError:
    st.error("Error: Model files not found. Did you run optimize_model.py?")
    st.stop()

# --- 2. THE MATH FUNCTION (Hidden Backend) ---
# must match your Day 4 logic exactly!
def get_amino_acid_composition(sequence):
    amino_acids = "ACDEFGHIKLMNPQRSTVWY"
    seq_len = len(sequence)
    if seq_len == 0:
        return {aa: 0 for aa in amino_acids}
    
    counts = Counter(sequence)
    composition = {aa: counts.get(aa, 0) / seq_len for aa in amino_acids}
    
    # Convert to a DataFrame (1 row) so the model accepts it
    return pd.DataFrame([composition])

# --- 3. THE UI (Frontend) ---
st.title("DeepGreen")
st.subheader("AI-Driven Autophagy Gene Detector")
st.markdown("---")

# Input Box
sequence_input = st.text_area(
    "Paste Protein Sequence (FASTA format or raw sequence):", 
    height=150,
    placeholder="Example: MKLV..."
)

# The  Button
if st.button("Analyze Sequence"):
    if not sequence_input.strip():
        st.warning("Please enter a sequence first.")
    else:
        # A. Cleanup: Remove headers (lines starting with >) and newlines
        clean_seq = "".join([line.strip() for line in sequence_input.splitlines() if not line.startswith(">")])
        clean_seq = clean_seq.upper() # Handle lowercase inputs
        
        # B. Check for valid characters
        valid_chars = set("ACDEFGHIKLMNPQRSTVWY")
        if not set(clean_seq).issubset(valid_chars):
             st.warning("Warning: Your sequence contains invalid characters (like X, B, Z). Results might be less accurate.")

        # C. Feature Engineering
        features = get_amino_acid_composition(clean_seq)
        
        # D. Scaling (CRITICAL step for SVM)
        features_scaled = scaler.transform(features)
        
        # E. Prediction
        prediction = model.predict(features_scaled)[0]
        probability = model.predict_proba(features_scaled)[0]
        
        # F. Display Results
        st.markdown("### Diagnosis:")
        
        if prediction == 1:
            st.success(f"**POSITIVE (Autophagy Protein)**")
            st.metric(label="Confidence Score", value=f"{probability[1]*100:.2f}%")
            st.balloons() # Fun animation
        else:
            st.error(f"**NEGATIVE (Non-Autophagy)**")
            st.metric(label="Confidence Score", value=f"{probability[0]*100:.2f}%")
            
        # Optional: Show the math
        with st.expander("See Biochemical Features"):
            st.dataframe(features)

# Footer
st.markdown("---")
st.caption("Built with Python & Scikit-Learn | © 2026 Farhan Alam")