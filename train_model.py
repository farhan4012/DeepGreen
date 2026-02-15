import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib  # To save the brain for later

# 1. Load the Math Data
df = pd.read_csv("final_features.csv")

# 2. Separate Features (X) and Answers (y)
X = df.drop("Label", axis=1)  # All columns except Label
y = df["Label"]               # Only the Label column

# 3. Split the data (80% for studying, 20% for the exam)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"🧠 Training on {len(X_train)} samples...")
print(f"📝 Testing on {len(X_test)} samples...")

# 4. Initialize the Brain (Random Forest)
# n_estimators=100 means we create 100 little decision trees
model = RandomForestClassifier(n_estimators=100, random_state=42)

# 5. Train!
model.fit(X_train, y_train)

# 6. Evaluate!
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("-" * 30)
print(f"🏆 Model Accuracy: {accuracy * 100:.2f}%")
print("-" * 30)

# 7. Show Details (Confusion Matrix)
# Top-Left: True Negatives | Top-Right: False Positives
# Bottom-Left: False Negatives | Bottom-Right: True Positives
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

print("\nDetailed Report:")
print(classification_report(y_test, y_pred))

# 8. Save the model for the App (Day 8)
joblib.dump(model, "autophagy_model.pkl")
print("\n✅ Model saved as 'autophagy_model.pkl'. Ready for the App!")