# 🕵️ Phishing Website Detection Dashboard

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)]([https://your-app.streamlit.app](https://phishingwebsitedetectorcasestudy-gx7hbsgdtsgklztxiehef6.streamlit.app/))  

This project provides an interactive dashboard to explore a phishing dataset and train a Random Forest classifier. The goal is to distinguish between legitimate and phishing websites based on URL and webpage characteristics.

## 📊 Dataset

The dataset (`phishing.csv`) contains **32 features** extracted from website URLs and page attributes, along with a binary target variable:

- **`class`**:  
  - `1` → Legitimate (or possibly phishing?) The dataset uses -1, 0, 1; we interpret 1 as legitimate and -1 as phishing, with 0 as a third class? We'll assume 1: legit, -1: phishing.  
  - The dataset has ~11,000 rows with no missing values.

### Features include:
- `UsingIP` – whether the URL uses an IP address  
- `LongURL`, `ShortURL` – flags for URL length  
- `Symbol@` – presence of '@' in URL  
- `HTTPS` – whether HTTPS is used  
- `DomainRegLen` – domain registration length  
- and many more …

(Full list can be seen in the app's **Dataset Overview** tab.)

## ❓ Problem Statement

Can we automatically detect phishing websites using only URL‑based features?  
The goal is to build a classification model that helps users identify malicious sites before they enter sensitive information.

## 🌐 Live App

The dashboard is deployed on **Streamlit Community Cloud**. Try it here:  
👉 **[Phishing Detection Dashboard](https://your-app.streamlit.app)**  <!-- Replace with your actual link -->

The app is organised into three tabs:

1. **Dataset Overview** – first rows, missing values, summary statistics.
2. **Exploratory Data Analysis** – six key visualisations with insights.
3. **Machine Learning** – train a Random Forest classifier, adjust hyperparameters, and view performance metrics.

## 📈 Exploratory Data Analysis

The EDA tab presents six pre‑defined plots to help understand the data:

| Plot | Insight |
|------|---------|
| **Class Distribution** | Checks whether the dataset is balanced. |
| **URL Length Type** | Phishing sites often use unusually long or short URLs. |
| **'@' Symbol in URL** | The '@' symbol can make a URL look legitimate while redirecting – a known phishing trick. |
| **IP Address in URL** | Legitimate sites rarely use raw IP addresses; a high count among phishing sites would be a strong signal. |
| **HTTPS Usage** | HTTPS is a security standard; phishing sites may omit it. |
| **Correlation Heatmap** | Reveals redundant features and shows which ones correlate most with the target. |

All plots are generated with **matplotlib** and **seaborn**, and each comes with a short insight directly below the chart.

## 🤖 Machine Learning Model

The app uses a **Random Forest Classifier** (`sklearn.ensemble.RandomForestClassifier`). Users can adjust:

- **Test size** (% of data held out)
- **Number of trees** (`n_estimators`)
- **Max depth** of each tree
- **Min samples split** (minimum samples required to split an internal node)
- **Optional feature scaling** (though tree‑based models are scale‑invariant)

After training, the app displays:

- **Accuracy score**
- **Full classification report** (precision, recall, f1‑score per class)
- **Confusion matrix** (heatmap)
- **Top‑20 feature importances** (bar chart)

The model and predictions are stored in **session state**, so you can switch between tabs without losing the trained model.

