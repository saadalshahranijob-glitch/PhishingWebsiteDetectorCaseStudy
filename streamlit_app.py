import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="Phishing Website Detector",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Phishing Website Detector Dashboard")
st.write("Machine Learning model to detect phishing websites")

# -----------------------------
# Cache data loading
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("phishing.csv")
    # Keep original features for modeling
    original_features = [
        'UsingIP', 'LongURL', 'ShortURL', 'Symbol@', 'Redirecting//',
        'PrefixSuffix-', 'SubDomains', 'HTTPS', 'DomainRegLen', 'Favicon',
        'NonStdPort', 'HTTPSDomainURL', 'RequestURL', 'AnchorURL',
        'LinksInScriptTags', 'ServerFormHandler', 'InfoEmail', 'AbnormalURL',
        'WebsiteForwarding', 'StatusBarCust', 'DisableRightClick',
        'UsingPopupWindow', 'IframeRedirection', 'AgeofDomain',
        'DNSRecording', 'WebsiteTraffic', 'PageRank', 'GoogleIndex',
        'LinksPointingToPage', 'StatsReport'
    ]
    # Add label mapping for visualization
    df['Class_label'] = df['class'].map({1: 'Legitimate', -1: 'Phishing'})
    return df, original_features

df, original_features = load_data()

# -----------------------------
# Cache model loading
# -----------------------------
@st.cache_resource
def load_model():
    # Load the trained Random Forest model (saved from notebook)
    with open('phishing_model.pkl', 'rb') as f:
        model = pickle.load(f)
    # If you used a scaler, load it here (not needed for tree-based models, but kept for consistency)
    # with open('scaler.pkl', 'rb') as f:
    #     scaler = pickle.load(f)
    # return model, scaler
    return model

model = load_model()
# scaler = load_scaler() # if applicable

# -----------------------------
# Sidebar navigation
# -----------------------------
st.sidebar.title("Navigation")
mode = st.sidebar.radio("Go to", ["Exploratory Data Analysis", "Phishing Detector"])

# -----------------------------
# EDA Section
# -----------------------------
if mode == "Exploratory Data Analysis":
    st.header("Dataset Preview")
    st.dataframe(df.head())

    st.subheader("Dataset Shape")
    st.write(df.shape)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Website Class Distribution")
        fig1, ax1 = plt.subplots()
        df["Class_label"].value_counts().plot(kind="bar", ax=ax1)
        ax1.set_title("Website Types")
        ax1.set_xlabel("Class")
        ax1.set_ylabel("Count")
        st.pyplot(fig1)

    with col2:
        st.subheader("Correlation Heatmap (selected features)")
        # Select a subset of features for a cleaner heatmap
        heat_cols = [
            "UsingIP", "LongURL", "ShortURL", "Symbol@", "SubDomains",
            "HTTPS", "DomainRegLen", "WebsiteTraffic", "PageRank", "GoogleIndex", "class"
        ]
        fig2, ax2 = plt.subplots(figsize=(8, 6))
        sns.heatmap(df[heat_cols].corr(), annot=True, cmap="coolwarm", ax=ax2)
        st.pyplot(fig2)

    # Model performance (using precomputed predictions if desired, but we'll compute fresh with cached model)
    st.header("Model Performance (on test set)")

    X = df[original_features]
    y = df['class']

    # Use a fixed split (same as notebook) – but we can just evaluate on whole dataset for simplicity
    # To avoid retraining, we use the already trained model to predict on the full dataset.
    # In practice, you'd want a separate test set, but for demo we'll use all data.
    y_pred = model.predict(X)
    accuracy = accuracy_score(y, y_pred)

    st.metric("Accuracy", f"{accuracy:.2%}")

    st.subheader("Classification Report")
    report = classification_report(y, y_pred, output_dict=True)
    st.dataframe(pd.DataFrame(report).transpose())

    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y, y_pred)
    fig3, ax3 = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax3)
    ax3.set_xlabel("Predicted")
    ax3.set_ylabel("Actual")
    st.pyplot(fig3)

    st.subheader("Feature Importance")
    importance = model.feature_importances_
    feat_imp = pd.DataFrame({
        "Feature": original_features,
        "Importance": importance
    }).sort_values(by="Importance", ascending=False).head(15)

    fig4, ax4 = plt.subplots(figsize=(8, 6))
    sns.barplot(x="Importance", y="Feature", data=feat_imp, ax=ax4)
    st.pyplot(fig4)

# -----------------------------
# Prediction Section
# -----------------------------
else:
    st.header("Phishing Detector")
    st.write("Enter the features of the website below to check if it's phishing or legitimate.")

    # Create input fields for all 30 features
    with st.form("prediction_form"):
        cols = st.columns(3)
        inputs = {}
        for i, feature in enumerate(original_features):
            col = cols[i % 3]
            # Determine appropriate input widget based on feature values
            # Most features are -1,0,1; some may have wider range like WebsiteTraffic, PageRank, etc.
            # We'll use selectbox for binary/ternary, slider for continuous.
            unique_vals = df[feature].unique()
            if set(unique_vals).issubset({-1, 0, 1}):
                # Categorical feature with -1,0,1
                options = {-1: "No / Negative", 0: "Neutral / Unknown", 1: "Yes / Positive"}
                # Map to display names
                display_options = list(options.values())
                selection = col.selectbox(feature, display_options, key=feature)
                # Convert back to numeric
                inv_map = {v: k for k, v in options.items()}
                inputs[feature] = inv_map[selection]
            else:
                # Continuous feature – use number input or slider based on range
                min_val = float(df[feature].min())
                max_val = float(df[feature].max())
                # Use number input with step
                inputs[feature] = col.number_input(
                    feature,
                    min_value=min_val,
                    max_value=max_val,
                    value=0.0,
                    step=1.0,
                    format="%f",
                    key=feature
                )

        submitted = st.form_submit_button("Predict")

    if submitted:
        # Convert inputs to DataFrame with correct column order
        input_df = pd.DataFrame([inputs])[original_features]

        # If you used a scaler, apply it here
        # input_scaled = scaler.transform(input_df)

        prediction = model.predict(input_df)[0]
        proba = model.predict_proba(input_df)[0]
        # model.classes_ might be [-1, 1] – check
        if hasattr(model, "classes_"):
            class_idx = list(model.classes_).index(prediction)
        else:
            class_idx = 0 if prediction == -1 else 1

        confidence = proba[class_idx]

        st.markdown("---")
        if prediction == 1:
            st.error(f"⚠️ This website is likely **legitimate** (confidence: {confidence:.2%})")
        else:
            st.success(f"✅ This website is likely **phishing** (confidence: {confidence:.2%})")

        # Optionally show all features and probabilities
        with st.expander("Show detailed probabilities"):
            proba_df = pd.DataFrame({
                "Class": model.classes_ if hasattr(model, "classes_") else [-1, 1],
                "Probability": proba
            })
            st.dataframe(proba_df)

    # Instructions for saving the model
    st.sidebar.markdown("---")
    st.sidebar.markdown("### How to use")
    st.sidebar.info(
        "To use this app, you need to have the trained model saved as `phishing_model.pkl` in the same directory. "
        "You can export the model from the notebook using:\n"
        "```python\n"
        "import pickle\n"
        "with open('phishing_model.pkl', 'wb') as f:\n"
        "    pickle.dump(rf_model, f)\n"
        "```"
    )
