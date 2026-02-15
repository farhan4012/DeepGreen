# 🌿 DeepGreen: ML driven Autophagy Predictor

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Machine Learning](https://img.shields.io/badge/Model-SVM-green)
![Accuracy](https://img.shields.io/badge/Accuracy-80%25-brightgreen)
![Status](https://img.shields.io/badge/Status-Prototype-orange)

## 🧬 Project Overview
**DeepGreen** is a bioinformatics machine learning tool designed to identify **Autophagy-related (ATG) genes** in plant genomes (*Arabidopsis thaliana*). Autophagy is a critical cellular recycling process that helps plants survive stress (drought, starvation).

Traditional identification of these genes requires expensive lab work. DeepGreen automates this using **Amino Acid Composition (AAC)** analysis and a tuned **Support Vector Machine (SVM)** classifier.

---

## 📊 Key Results
We benchmarked Random Forest, Gradient Boosting, and SVM. The **SVM (RBF Kernel)** outperformed others with high precision, making it a "Zero False Positive" detector for this dataset.

| Metric | Score | Note |
| :--- | :--- | :--- |
| **Accuracy** | **80%** | Overall correctness on unseen test data. |
| **Precision (Autophagy)** | **100%** | When it predicts "Autophagy," it is *always* correct. |
| **Recall (Autophagy)** | **50%** | Conservative prediction (minimizes false alarms). |

---

## 🛠️ Tech Stack
* **Language:** Python
* **Bioinformatics:** Biopython (Sequence parsing)
* **ML Engine:** Scikit-learn (SVM, Random Forest)
* **Data Handling:** Pandas & NumPy
* **Visualization:** Matplotlib & Seaborn

---

## 📂 Project Structure
```text
DeepGreen/
├── data/                   # FASTA files (UniProt sourced)
├── dataset.csv             # Processed numerical dataset
├── feature_extraction.py   # Converts Protein Sequences -> Math (AAC)
├── train_model.py          # Trains the SVM brain
├── visualize_svm.py        # Generates scientific plots
├── autophagy_model.pkl     # The saved AI brain (Ready for deployment)
└── README.md               # You are here