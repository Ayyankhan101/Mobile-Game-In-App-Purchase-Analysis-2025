import pandas as pd
import numpy as np
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, RocCurveDisplay, confusion_matrix, ConfusionMatrixDisplay
import shap

# --------------------------------------------------
# 1. Load & cache data
# --------------------------------------------------
@st.cache_data
def load_data():
    # Replace with local path or permanent URL
    url = "mobile_game_inapp_purchases.csv"
    return pd.read_csv(url)

df_raw = load_data()

# --------------------------------------------------
# 2. Basic cleaning & feature engineering
# --------------------------------------------------
def clean(df):
    df = df.copy()
    num = ['Age', 'SessionCount', 'AverageSessionLength', 'FirstPurchaseDaysAfterInstall']
    for c in num:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df['log_spend'] = np.log1p(df['InAppPurchaseAmount'].fillna(0))
    df['is_whale'] = (df['SpendingSegment'] == 'Whale').astype(int)
    return df

df = clean(df_raw)

# --------------------------------------------------
# 3. Train / test split & model (cached)
# --------------------------------------------------
@st.cache_resource
def build_model():
    cat = ['Gender', 'Country', 'Device', 'GameGenre']
    num = ['Age', 'SessionCount', 'AverageSessionLength', 'FirstPurchaseDaysAfterInstall']
    pre = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat),
        ('num', 'passthrough', num)
    ],
    sparse_threshold=0        # <-- forces dense output
)
    clf = Pipeline([
        ('pre', pre),
        ('gb', HistGradientBoostingClassifier(random_state=42))
    ])
    X = df.dropna(subset=['is_whale'])[cat + num]
    y = df.dropna(subset=['is_whale'])['is_whale']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25,
                                                        stratify=y, random_state=42)
    clf.fit(X_train, y_train)
    y_prob = clf.predict_proba(X_test)[:, 1]
    roc = roc_auc_score(y_test, y_prob)
    return clf, X_test, y_test, roc

model, X_test, y_test, roc_auc = build_model()

# --------------------------------------------------
# 4. Streamlit layout
# --------------------------------------------------
st.set_page_config(page_title="Game Revenue Analytics", layout="wide")
st.title("📊 Mobile-Game Revenue & Whale Prediction Dashboard")
st.markdown("---")

# --------------------------------------------------
# 5. Key metrics
# --------------------------------------------------
col1, col2, col3 = st.columns(3)
col1.metric("Total Players", f"{len(df):,}")
col2.metric("Whales", f"{df['is_whale'].sum():,}")
col3.metric("Model ROC-AUC", roc_auc)

st.markdown("---")


# --------------------------------------------------
# 9. Section 4 – SHAP waterfall for a random user
# --------------------------------------------------


st.header("🔍 Explain a Single Prediction")
if st.button("Pick Random User"):
    idx = np.random.choice(X_test.index)
    sample = X_test.loc[[idx]]
    X_dense = model.named_steps['pre'].transform(sample)

    explainer = shap.TreeExplainer(model.named_steps['gb'])
    shap_values = explainer.shap_values(X_dense)

    # shap_values is 1-D for binary classification
    plt.figure(figsize=(8, 6))
    shap.waterfall_plot(
        shap.Explanation(values=shap_values[0],
                         base_values=explainer.expected_value,
                         feature_names=model.named_steps['pre'].get_feature_names_out()),
        max_display=15)
    st.pyplot(plt)



# --------------------------------------------------
# 6. Section 1 – Revenue distribution
# --------------------------------------------------
st.header("💰 Revenue & Segment Overview")
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
sns.histplot(df, x='InAppPurchaseAmount', bins=50, kde=True, ax=ax[0])
ax[0].set_xscale('log')
ax[0].axvline(1000, ls='--', c='red')
ax[0].set_title("Spend Distribution")
order = ['Minnow', 'Dolphin', 'Whale']
sns.countplot(y='SpendingSegment', data=df, order=order, ax=ax[1])
ax[1].set_title("Segment Counts")
st.pyplot(fig)

# --------------------------------------------------
# 7. Section 2 – Behaviour vs segment
# --------------------------------------------------
st.header("🕹️ Session Behaviour vs Segment")
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
sns.boxenplot(x='SpendingSegment', y='AverageSessionLength', data=df, ax=ax[0], order=order)
ax[0].set_title("Avg Session Length by Segment")
sns.boxenplot(x='SpendingSegment', y='SessionCount', data=df, ax=ax[1], order=order)
ax[1].set_title("Session Count by Segment")
st.pyplot(fig)

# --------------------------------------------------
# 8. Section 3 – ROC & Confusion matrix
# --------------------------------------------------
st.header("🎯 Model Performance")
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
RocCurveDisplay.from_estimator(model, X_test, y_test, ax=ax[0])
ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, ax=ax[1], display_labels=['Other', 'Whale'])
st.pyplot(fig)
