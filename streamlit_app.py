
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# -----------------------------
# Page Title
# -----------------------------
st.title("Phishing Website Detector Dashboard")

st.write("Machine Learning model to detect phishing websites")

# -----------------------------
# Load Dataset
# -----------------------------
df = pd.read_csv("phishing.csv")

st.subheader("Dataset Preview")
st.dataframe(df.head())

# -----------------------------
# Dataset Info
# -----------------------------
st.subheader("Dataset Shape")
st.write(df.shape)

# -----------------------------
# Class Distribution
# -----------------------------
st.subheader("Website Class Distribution")

fig1, ax1 = plt.subplots()

df["class"].value_counts().plot(
    kind="bar",
    ax=ax1
)

ax1.set_title("Website Types")
ax1.set_xlabel("Class")
ax1.set_ylabel("Count")

st.pyplot(fig1)

# -----------------------------
# Correlation Heatmap
# -----------------------------
st.subheader("Correlation Heatmap")

fig2, ax2 = plt.subplots(figsize=(10,6))

sns.heatmap(
    df.corr(),
    cmap="coolwarm",
    ax=ax2
)

st.pyplot(fig2)

# -----------------------------
# Feature / Target Split
# -----------------------------
X = df.drop("class", axis=1)
y = df["class"]

# -----------------------------
# Train Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# -----------------------------
# Train Model
# -----------------------------
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# -----------------------------
# Predictions
# -----------------------------
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

# -----------------------------
# Accuracy
# -----------------------------
st.subheader("Model Accuracy")

st.write(f"Accuracy: {accuracy:.2f}")

# -----------------------------
# Classification Report
# -----------------------------
st.subheader("Classification Report")

report = classification_report(y_test, y_pred)

st.text(report)

# -----------------------------
# Confusion Matrix
# -----------------------------
st.subheader("Confusion Matrix")

cm = confusion_matrix(y_test, y_pred)

fig3, ax3 = plt.subplots()

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    ax=ax3
)

ax3.set_xlabel("Predicted")
ax3.set_ylabel("Actual")

st.pyplot(fig3)

# -----------------------------
# Feature Importance
# -----------------------------
st.subheader("Feature Importance")

importance = model.feature_importances_

feat_imp = pd.DataFrame({
    "Feature": X.columns,
    "Importance": importance
}).sort_values(by="Importance", ascending=False)

fig4, ax4 = plt.subplots(figsize=(8,6))

sns.barplot(
    x="Importance",
    y="Feature",
    data=feat_imp.head(10),
    ax=ax4
)

st.pyplot(fig4)
