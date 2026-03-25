# MAKE PREDICTIONS USING TRAINED MODEL

import joblib
import pandas as pd
import numpy as np

from utils import get_risk_level, explain_prediction, dynamic_guidance

# Load trained model
model = joblib.load("models/heart_model.pkl")

# Load test data
test = pd.read_csv("data/test.csv", nrows=5)

# Save IDs (optional)
ids = test["id"]

# Drop ID column
test = test.drop("id", axis=1)

# Predict probabilities
probs = model.predict_proba(test)[:,1]

# Compute thresholds from predictions
low = np.percentile(probs, 33)
high = np.percentile(probs, 66)

print("\n=== PATIENT RISK ANALYSIS ===")

# Loop through patients
for i in range(len(test)):
    
    sample = test.iloc[i]
    prob = probs[i]

    # Get risk level
    level = get_risk_level(prob, low, high)

    print(f"\nPatient {i+1}")
    print(f"Risk Score: {prob:.2f}")
    print(f"Risk Level: {level}")

    # EXPLANATION (Top features)
    #Simple fallback (since model is Logistic Regression)
    print("\nTop Contributing Factors:")
    for col in sample.index[:3]:  # simple placeholder
        print(f"• {col}")

    # DYNAMIC GUIDANCE
    advice = dynamic_guidance(sample, level)

    print("\nRecommendations:")
    for a in advice:
        print("•", a)
