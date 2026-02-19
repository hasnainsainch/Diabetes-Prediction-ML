# ==============================
# DIABETES PREDICTION PROJECT
# Complete ML + Web App in One File
# ==============================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# ------------------------------
# PAGE CONFIG
# ------------------------------
st.set_page_config(page_title="Diabetes Prediction System", layout="wide")

st.title("🩺 Diabetes Prediction Using Machine Learning")

# ------------------------------
# LOAD DATA
# ------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("diabetes.csv")
    return df

df = load_data()

st.subheader("Dataset Preview")
st.write(df.head())

# ------------------------------
# DATA PREPROCESSING
# ------------------------------
cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
for col in cols:
    df[col] = df[col].replace(0, df[col].median())

X = df.drop("Outcome", axis=1)
y = df["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ------------------------------
# TRAIN MODELS
# ------------------------------
models = {
    "Logistic Regression": LogisticRegression(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "SVM": SVC()
}

results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    results[name] = acc

# ------------------------------
# MODEL COMPARISON
# ------------------------------
st.subheader("Model Accuracy Comparison")

results_df = pd.DataFrame(results.items(), columns=["Model", "Accuracy"])
st.write(results_df)

fig, ax = plt.subplots()
sns.barplot(x="Model", y="Accuracy", data=results_df, ax=ax)
plt.xticks(rotation=45)
st.pyplot(fig)

# ------------------------------
# BEST MODEL (Random Forest)
# ------------------------------
best_model = RandomForestClassifier()
best_model.fit(X_train, y_train)
y_pred = best_model.predict(X_test)

st.subheader("Confusion Matrix (Random Forest)")

cm = confusion_matrix(y_test, y_pred)
fig2, ax2 = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax2)
st.pyplot(fig2)

st.subheader("Classification Report")
st.text(classification_report(y_test, y_pred))

# ------------------------------
# PREDICTION SECTION
# ------------------------------
st.subheader("🔍 Test the Model")

pregnancies = st.number_input("Pregnancies", 0, 20)
glucose = st.number_input("Glucose Level", 0, 200)
bp = st.number_input("Blood Pressure", 0, 150)
skin = st.number_input("Skin Thickness", 0, 100)
insulin = st.number_input("Insulin", 0, 900)
bmi = st.number_input("BMI", 0.0, 70.0)
dpf = st.number_input("Diabetes Pedigree Function", 0.0, 3.0)
age = st.number_input("Age", 1, 120)

if st.button("Predict"):
    input_data = np.array([[pregnancies, glucose, bp, skin,
                            insulin, bmi, dpf, age]])
    input_data = scaler.transform(input_data)
    prediction = best_model.predict(input_data)

    if prediction[0] == 1:
        st.error("⚠ High Risk of Diabetes")
    else:
        st.success("✅ Low Risk of Diabetes")

# ------------------------------
# FOOTER
# ------------------------------
st.markdown("---")
st.markdown("Developed for BS Data Science Final Year Project")