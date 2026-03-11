import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

# Set page configuration
st.set_page_config(page_title="Phishing Dataset Analysis", layout="wide")

# Title
st.title("🔍 Phishing Website Detection")
st.markdown("Explore the phishing dataset and train a machine learning model.")

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("phishing.csv")
    # Drop the 'Index' column if it exists (first column)
    if 'Index' in df.columns:
        df = df.drop(columns=['Index'])
    return df

df = load_data()

# Sidebar
st.sidebar.header("Dataset Overview")
st.sidebar.write(f"**Rows:** {df.shape[0]}")
st.sidebar.write(f"**Columns:** {df.shape[1]}")
st.sidebar.write(f"**Target column:** 'class'")
st.sidebar.write(f"**Class values:** {df['class'].unique()}")

# Display data
st.subheader("Raw Data (first 100 rows)")
st.dataframe(df.head(100))

# Basic statistics
st.subheader("Statistical Summary")
st.write(df.describe())

# Class distribution
st.subheader("Class Distribution")
fig, ax = plt.subplots()
df['class'].value_counts().plot(kind='bar', ax=ax, color=['skyblue', 'salmon'])
ax.set_xlabel("Class")
ax.set_ylabel("Count")
ax.set_title("Distribution of Target Classes")
st.pyplot(fig)

# Correlation heatmap (optional, can be slow for many features)
if st.checkbox("Show correlation heatmap (may take a moment)"):
    st.subheader("Feature Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(df.corr(), annot=False, cmap='coolwarm', ax=ax)
    st.pyplot(fig)

# Machine Learning Section
st.header("🤖 Train a Random Forest Classifier")

# Prepare features and target
X = df.drop(columns=['class'])
y = df['class']

# Split data
test_size = st.slider("Test size (%)", 10, 40, 20, 5) / 100
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)

st.write(f"**Training samples:** {X_train.shape[0]}")
st.write(f"**Testing samples:** {X_test.shape[0]}")

# Optional scaling
scale = st.checkbox("Apply Standard Scaling to features")
if scale:
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    st.info("Features have been standardized.")

# Model parameters
st.sidebar.header("Model Parameters")
n_estimators = st.sidebar.slider("Number of trees", 10, 200, 100, 10)
max_depth = st.sidebar.slider("Max depth", 1, 20, 10, 1)
min_samples_split = st.sidebar.slider("Min samples split", 2, 20, 2, 1)

# Train button
if st.button("Train Model"):
    with st.spinner("Training in progress..."):
        clf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=42,
            n_jobs=-1
        )
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

    st.success(f"Model trained! Accuracy: **{acc:.4f}**")

    # Classification report
    st.subheader("Classification Report")
    report = classification_report(y_test, y_pred, output_dict=True)
    st.dataframe(pd.DataFrame(report).transpose())

    # Confusion matrix
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Legit', 'Phish'], yticklabels=['Legit', 'Phish'])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

    # Feature importance
    if hasattr(clf, 'feature_importances_'):
        st.subheader("Feature Importance")
        importances = clf.feature_importances_
        feature_names = X.columns
        imp_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
        imp_df = imp_df.sort_values('importance', ascending=False).head(20)
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(data=imp_df, x='importance', y='feature', palette='viridis', ax=ax)
        ax.set_title("Top 20 Feature Importances")
        st.pyplot(fig)

st.markdown("---")
st.caption("Built with Streamlit • Dataset: phishing.csv")
