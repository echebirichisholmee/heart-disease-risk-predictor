#Import libraries 
#Pandas is for loading and handling our data 
#Sklearn = ML algorithims and evaluation tools 
from unittest import result

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier 
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from  sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

train =pd.read_csv("train.csv", nrows= 50000) #mytrain file 

#See shape and first few rows 
print(train.head())
print("Train shape:", train.shape)
#Check for missing values 
print(train.isnull().sum())
#Check data type 
print(train.dtypes)
#Check for duplicates 
print("Duplicates:", train.duplicated().sum)
#Basic Statistics 
print(train.describe())
print(train.columns)

#Drop ID column cause keeping it would confuse the model 
train = train.drop("id", axis=1)

#Separate features (X) and target (y)
#X = the medical information what the model uses to predict 
#y = the answer we want to predict(Heart disease: yes or no)
X= train.drop("Heart Disease", axis =1)
y = train["Heart Disease"]

#Split training Data into train + validation 
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size= 0.2, random_state= 17
)
print("Training set size:", X_train.shape)
print("Validation set size:", X_val.shape)

#----------------------------------------------------------------------------------------------
#Define models 
models = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter =1000, random_state= 17))
    ]),
    "Random Forest": Pipeline([
        ("scaler", StandardScaler()),
        ("model", RandomForestClassifier(n_estimators =100, random_state= 17))
    ]),
    "SVM": Pipeline ([
        ("scaler", StandardScaler()),
        ("model", SVC(random_state =17))
    ]),
    "KNN": Pipeline ([
        ("scaler", StandardScaler()),
        ("model", KNeighborsClassifier(n_neighbors =5))

    ]),

}

#------------------------------------------------------------------------------------------------
# Train and evaluate models 
results ={}
print("\MODEL RESULTS\n" + "=" *40)

for name, pipeline in models.items():
    pipeline.fit(X_train, y_train)

    y_pred =pipeline.predict(X_val)
     
    acc= accuracy_score(y_val, y_pred)
    results[name]= acc

    print(f"{name}: {acc:.4f}")

# Select the best model 
best_model_name = max(results, key= results.get)
best_pipeline = models[best_model_name]

print(f"\nBEST MODEL: {best_model_name}")

   
#Evaluation 
y_val_pred= best_pipeline.predict(X_val)
print("\nclassification_report:")
print(classification_report(y_val, y_val_pred))

#----------------------------------------------------------------------------------------------
#CONFUSION MATRIX
#This shows how many predictions the model got wrong or right 
#for each class(Presence/ Absence of heart disease)
#It helps us understand where the model makes mistakes 

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

#Generate the confusion matrix using the true labels and predicted labels
cm = confusion_matrix(y_val, y_val_pred)

#Create a visual display of the confusion matrix 
disp = ConfusionMatrixDisplay(confusion_matrix=cm)

#Plot the confusion matrix 
plt.title("Confusion Matrix")
plt.show()


#-----------------------------------------------------------------------------------------------------------------
#ROC CURVE 
#ROC curve evaluates how well the model separates the two classes 
#(heart disease vs no heart disease)
#If the ROCcurve bends strongly toward the top-left corner the model is good 

from sklearn.metrics import roc_curve, roc_auc_score

#Get probability predictions from the model
#[:, 1] means we take the probability of the positive class(Presence)
y_prob = best_pipeline.predict_proba(X_val)[:, 1]

#Calculate the false positive rate and true positive rate
fpr,tpr, _ = roc_curve(y_val, y_prob)

#Plot the ROC curve 
plt.plot(fpr, tpr, label="ROC CURVE")

#PLot a diagonal line representing random guessing 
plt.plot([0,1], [0,1], linestyles= "--")

#Label the axes 
plt.xlabel("False Positive Rate ")
plt.ylabel("True Positive Rate")

#Add title 
plt.title("ROC Curve")

#Display the plot 
plt.show()

#------------------------------------------------------------------------------------------------------------------
#MODEL ACCURACY COMPARISON 
#This visualizes the accuracy of all the algot=rithims we tested 

#Extract model names from the results dictionary 
names = list(results.keys())

#Extract accuracy scores 
scores = list(results.values())

#Create a bar chart comparing model performance 
plt.bar(names,scores)

#Label the chart 
plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy")

#Display the chart 
plt.show

#-------------------------------------------------------------------------------------------------------------------------------
#Make final predictions on test data 
#There is no accuracy here cause theres no label 

test = pd.read_csv("test.csv")
test_ids = test["id"]
test = test.drop("id", axis= 1)

final_predictions = best_pipeline.predict(test)

print("\nSample Predictions on Test Data:")
print(final_predictions[:10])

#Data driven risk thresholds

import numpy as np 

#get probability for all validation samples
all_probabilities = best_pipeline.predict_proba(X_val)[:,1]
#calculate thresholds using percentiles 
low_threshold =np.percentile(all_probabilities, 33)
high_threshold =np.percentile(all_probabilities, 66)

print(f"\nLow Risk Threshold:{low_threshold:.2f}")
print(f"High Risk Threshold:{high_threshold:.2f}")

#get risk for one patient
#select a sample patient 
sample= X_val.iloc[0]
#convert to 2D format
sample_2d = [sample]

#get probability of heart disease 
prob= best_pipeline.predict_proba(sample_2d)[0][1]
print(f"\nHeart Disease Risk Score: {prob:.2f}")

#assign risk level
if prob < low_threshold:
    level= "Low Risk"
elif prob < high_threshold:
    level = "Moderate Risk"
else: 
    level = "High Risk"
print("Risk Level:", level)

#---------------------------------------------------------------------------------------
#Explain prediction that is why the model made this decision 

#This tells us which medical features influence the model most 
#(examp#le: cholestreol,age,blood pressure )
#So it will visually show which variables influence predictions most 
#get random forest feature importance 
rf_model = models["Random Forest"].named_steps["model"]
importance= rf_model.feature_importances_

feature_importance = pd.Series(importance, index=X.columns)

#calculate contribution of each feature 
contributions= sample * feature_importance

#get top 3 contributing features 
top_contributors = contributions.sort_values(ascending=False).head(3)

print("n\Top ContributingFactors:")
for feature in top_contributors.index:
    print(f"• {feature.replace('_', '').title()}")

#-----------------------------------------------------------------------------------------------
#Provide basic health guidance

def give_health_advice(prediction):

    if prediction== "Presence":
        print("\n ⚠️ Potential Risk of Heart Disesase Detected")
        print("Recommended Actions:")
        print("- Consult a cardiologist")
        print("- Monitor blood pressure regularly")
        print("- Reduce cholesterol intake")
        print("- Exercise regularly")
        print("- Followa heart healthy diet")

    else:
        print("\n ✅ Low Risk of Heart Disease")
        print("Recommended Actions:")
        print("- Maintain healthy diet")
        print("- Continue regular exercise")
        print("- Keep monitoring blood pressure")
        print("- Attend routine medical checkups")
#Example usage
sample_prediction = final_predictions[0]
give_health_advice(sample_prediction)
        
    