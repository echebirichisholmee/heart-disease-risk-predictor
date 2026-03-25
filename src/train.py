
# TRAIN MODEl   
import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

print("Starting training...")
# Load dataset
train = pd.read_csv("data/train.csv", nrows=50000)

# Drop ID column
train = train.drop("id", axis=1)

# Features and target
X = train.drop("Heart Disease", axis=1)
y = train["Heart Disease"]

# Split data
X_train, X_val, y_train, y_val = train_test_split(

    X, y, test_size=0.2, random_state=17
)

# Define model (just BEST one, not all)
model = Pipeline([

    ("scaler", StandardScaler()),

    ("model", LogisticRegression(max_iter=1000, random_state=17))
])
# Train model
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_val)
acc = accuracy_score(y_val, y_pred)

print(f"Validation Accuracy: {acc:.4f}")

#create models folder automatically
os.makedirs("models", exist_ok=True)


# Save model
joblib.dump(model, "models/heart_model.pkl")

print("Model saved successfully!")
print("Training complete")

