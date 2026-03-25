# UTILS — Helper functions for prediction system

import pandas as pd

# RISK LEVEL FUNCTION
def get_risk_level(prob, low, high):
    """
    Convert probability into risk category
    """
    if prob < low:
        return "Low Risk"
    elif prob < high:
        return "Moderate Risk"
    else:
        return "High Risk"

# EXPLAIN PREDICTION (TOP FEATURES)
def explain_prediction(sample, feature_importance):
    """
    Returns top contributing features for a patient
    """
    contributions = sample * feature_importance
    top = contributions.sort_values(ascending=False).head(3)

    # Clean feature names
    return [f.replace("_", " ").title() for f in top.index]

# DYNAMIC HEALTH GUIDANCE
def dynamic_guidance(sample, level):
    """
    Generate personalized advice based on patient data
    """
    advice = []

    #Made sure these column names match my dataset exactly

    if "Cholesterol" in sample and sample["Cholesterol"] > 240:
        advice.append("Reduce cholesterol intake")

    if "BP" in sample and sample["BP"] > 130:
        advice.append("Monitor and manage blood pressure")

    if "Max HR" in sample and sample["Max HR"] < 100:
        advice.append("Improve cardiovascular fitness")

    if "ST depression" in sample and sample["ST depression"] > 2:
        advice.append("Consult a doctor about heart stress indicators")

    if "Exercise angina" in sample and sample["Exercise angina"] == 1:
        advice.append("Avoid strenuous activity without medical advice")

    # Add general advice based on risk level
    if level == "High Risk":
        advice.append("Consult a cardiologist immediately")

    elif level == "Moderate Risk":
        advice.append("Schedule routine medical checkups")

    else:
        advice.append("Maintain a healthy lifestyle")

    return advice

