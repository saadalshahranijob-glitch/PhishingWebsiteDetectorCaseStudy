import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# ------------------------------
# Page config
# ------------------------------
st.set_page_config(page_title="Phishing Detection Dashboard", layout="wide")

# ------------------------------
# Session state for model and predictions
# ------------------------------
if 'model' not in st.session_state:
    st.session_state.model = None
if 'y_test' not in st.session_state:
    st.session_state.y_test = None
if 'y_pred' not in st.session_state:
    st.session_state.y_pred = None
if 'X_test' not in st.session_state:
    st.session_state.X_test = None
if 'feature_names' not in st.session_state:
    st.session_state.feature_names = None

# ------------------------------
# Load data
# ------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("phishing.csv")
    if 'Index' in df.columns:
        df = df.drop(columns=['Index'])
    return df

df = load_data()

# ------------------------------
# Main title
# ------------------------------
st.title("🔍 Phishing Website Detection Dashboard")
st.markdown("Explore the dataset, visualize features, and train a Random Forest classifier.")

# ------------------------------
# Create three tabs
# ------------------------------
tab1, tab2, tab3 = st.tabs(["📁 Dataset Overview", "📈 Exploratory Data Analysis", "🤖 Machine Learning"])

# ======================
# TAB 1: Dataset Overview
# ======================
with tab1:
    st.header("Dataset Overview")
    
    colA, colB = st.columns(2)
    with colA:
        st.subheader("First 100 rows")
        st.dataframe(df.head(100))
    with colB:
        st.subheader("Missing Values")
        missing = df.isnull().sum()
        if missing.sum() == 0:
            st.success("✅ No missing values in the dataset.")
        else:
            st.write(missing[missing > 0])
    
    st.subheader("Statistical Summary")
    st.write(df.describe())

# ======================
# TAB 2: Exploratory Data Analysis (styled layout)
# ======================
with tab2:
    st.header("Exploratory Data Analysis")
    st.markdown("**Patterns, quality, and correlations across phishing indicators**")
    
    # Create a copy for EDA to avoid altering the original DataFrame
    eda_df = df.copy()
    
    # Helper function for insights (displayed as captions)
    def add_insight(text):
        st.caption(f"💡 {text}")
    
    # ---------- Row 1: Class Distribution & URL Length Types ----------
    st.subheader("Class Distribution & URL Characteristics")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Class Distribution**")
        fig, ax = plt.subplots(figsize=(5, 3))
        eda_df['class'].value_counts().plot(kind='bar', ax=ax, color=['skyblue', 'salmon'])
        ax.set_xlabel("Class")
        ax.set_ylabel("Count")
        ax.set_title("")
        st.pyplot(fig)
        add_insight("Balance between legitimate and phishing sites.")
    
    with col2:
        st.markdown("**URL Length Type**")
        eda_df['URL_Length_Type'] = 'Normal'
        eda_df.loc[eda_df['LongURL'] == 1, 'URL_Length_Type'] = 'Long'
        eda_df.loc[eda_df['ShortURL'] == 1, 'URL_Length_Type'] = 'Short'
        length_counts = eda_df.groupby(['class', 'URL_Length_Type']).size().unstack(fill_value=0)
        fig, ax = plt.subplots(figsize=(5, 3))
        length_counts.plot(kind='bar', ax=ax, color=['#2ecc71', '#e74c3c', '#3498db'])
        ax.set_xlabel("Class")
        ax.set_ylabel("Count")
        ax.set_title("")
        ax.legend(title="Length")
        st.pyplot(fig)
        add_insight("Phishing often uses unusual URL lengths.")
    
    # ---------- Row 2: Three key binary indicators ----------
    st.subheader("Key Phishing Indicators")
    col3, col4, col5 = st.columns(3)
    
    with col3:
        st.markdown("**'@' Symbol in URL**")
        eda_df['Symbol@_label'] = eda_df['Symbol@'].map({-1: 'No', 1: 'Yes'})
        symbol_counts = eda_df.groupby(['class', 'Symbol@_label']).size().unstack(fill_value=0)
        fig, ax = plt.subplots(figsize=(4, 2.5))
        symbol_counts.plot(kind='bar', ax=ax, color=['#3498db', '#e67e22'])
        ax.set_xlabel("Class")
        ax.set_ylabel("Count")
        ax.set_title("")
        ax.legend(title="Contains '@'")
        st.pyplot(fig)
        add_insight("@ symbol is a red flag.")
    
    with col4:
        st.markdown("**IP Address in URL**")
        eda_df['UsingIP_label'] = eda_df['UsingIP'].map({-1: 'No', 1: 'Yes'})
        ip_counts = eda_df.groupby(['class', 'UsingIP_label']).size().unstack(fill_value=0)
        fig, ax = plt.subplots(figsize=(4, 2.5))
        ip_counts.plot(kind='bar', ax=ax, color=['#3498db', '#e67e22'])
        ax.set_xlabel("Class")
        ax.set_ylabel("Count")
        ax.set_title("")
        ax.legend(title="Uses IP")
        st.pyplot(fig)
        add_insight("IP addresses are rare in legitimate URLs.")
    
    with col5:
        st.markdown("**HTTPS Usage**")
        eda_df['HTTPS_label'] = eda_df['HTTPS'].map({-1: 'No', 1: 'Yes'})
        https_counts = eda_df.groupby(['class', 'HTTPS_label']).size().unstack(fill_value=0)
        fig, ax = plt.subplots(figsize=(4, 2.5))
        https_counts.plot(kind='bar', ax=ax, color=['#3498db', '#e67e22'])
        ax.set_xlabel("Class")
        ax.set_ylabel("Count")
        ax.set_title("")
        ax.legend(title="HTTPS")
        st.pyplot(fig)
        add_insight("Legitimate sites almost always use HTTPS.")
    
    # ---------- Row 3: Correlation Heatmap ----------
    st.subheader("Feature Correlations")
    fig, ax = plt.subplots(figsize=(10, 6))
    numeric_cols = [col for col in df.columns if col != 'class']
    sns.heatmap(df[numeric_cols].corr(), annot=False, cmap='coolwarm', ax=ax)
    ax.set_title("Correlation Matrix of Phishing Features")
    st.pyplot(fig)
    add_insight("Highly correlated features may be redundant; look for strong correlations with the target.")

# ======================
# TAB 3: Machine Learning
# ======================
with tab3:
    st.header("Train a Random Forest Classifier")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        test_size = st.slider("Test size (%)", 10, 40, 20, 5) / 100
    with col2:
        n_estimators = st.slider("Number of trees", 10, 200, 100, 10)
    with col3:
        max_depth = st.slider("Max depth", 1, 20, 10, 1)
    min_samples_split = st.slider("Min samples split", 2, 20, 2, 1)
    
    scale = st.checkbox("Apply Standard Scaling to features")
    train_button = st.button("Train Model")
    
    if train_button:
        X = df.drop(columns=['class'])
        y = df['class']
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        if scale:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
        else:
            X_train = X_train.values
            X_test = X_test.values
        
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
        
        # Store in session state
        st.session_state.model = clf
        st.session_state.y_test = y_test
        st.session_state.y_pred = y_pred
        st.session_state.X_test = X_test
        st.session_state.feature_names = X.columns.tolist()
        
        st.success(f"✅ Model trained! Accuracy: **{acc:.4f}**")
    
    # Show performance results if model exists
    if st.session_state.model is not None:
        st.subheader("📈 Model Performance")
        colA, colB = st.columns(2)
        with colA:
            st.metric("Accuracy", f"{accuracy_score(st.session_state.y_test, st.session_state.y_pred):.4f}")
        with colB:
            st.dataframe(
                pd.DataFrame(classification_report(
                    st.session_state.y_test,
                    st.session_state.y_pred,
                    output_dict=True
                )).transpose()
            )
        
        # Confusion matrix
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(st.session_state.y_test, st.session_state.y_pred)
        fig_cm, ax_cm = plt.subplots(figsize=(4,3))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm,
                    xticklabels=['Legit', 'Phish'], yticklabels=['Legit', 'Phish'])
        ax_cm.set_xlabel("Predicted")
        ax_cm.set_ylabel("Actual")
        st.pyplot(fig_cm)
        
        # Feature importance
        if hasattr(st.session_state.model, 'feature_importances_'):
            st.subheader("Feature Importance")
            importances = st.session_state.model.feature_importances_
            imp_df = pd.DataFrame({
                'feature': st.session_state.feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False).head(20)
            fig_imp, ax_imp = plt.subplots(figsize=(4,3))
            sns.barplot(data=imp_df, x='importance', y='feature', palette='viridis', ax=ax_imp)
            ax_imp.set_title("Top 20 Feature Importances")
            st.pyplot(fig_imp)

st.markdown("---")
st.caption("Built with Streamlit • Dataset: phishing.csv")
