import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# 1. Load Data
df = pd.read_csv("final_features.csv")
X = df.drop("Label", axis=1)
y = df["Label"]

# 2. Split Data (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Scale Data (Crucial for SVM!)
# This makes all numbers comparable (z-scores)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save the scaler so the App can use it later
joblib.dump(scaler, "scaler.pkl")

# --- DEFINING THE CONTENDERS ---

models = {
    "Random Forest (Tuned)": RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42),
    "SVM (Support Vector Machine)": SVC(kernel='rbf', C=1.0, probability=True, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
}

best_model = None
best_accuracy = 0
best_name = ""

print(f"{'Model Name':<30} | {'Accuracy':<10} | {'Cross-Val Mean':<15}")
print("-" * 65)

for name, model in models.items():
    # Train
    model.fit(X_train_scaled, y_train)
    
    # Test on the 20% holdout set
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    
    # Cross-Validation (Double Check)
    # This trains 5 times on different splits to ensure the score isn't luck
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
    cv_mean = cv_scores.mean()
    
    print(f"{name:<30} | {acc:.2%}    | {cv_mean:.2%}")
    
    # Keep the winner
    if acc > best_accuracy:
        best_accuracy = acc
        best_model = model
        best_name = name

print("-" * 65)
print(f"🏆 WINNER: {best_name} with {best_accuracy:.2%} Accuracy")

# Show detailed report for the winner
print(f"\nDetailed Report for {best_name}:")
y_pred_final = best_model.predict(X_test_scaled)
print(classification_report(y_test, y_pred_final))

# Save the best model
joblib.dump(best_model, "autophagy_model.pkl")
print("✅ Best model saved as 'autophagy_model.pkl'")