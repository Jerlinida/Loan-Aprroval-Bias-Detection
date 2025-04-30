
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from aif360.datasets import StandardDataset
from aif360.algorithms.preprocessing import Reweighing
from aif360.metrics import BinaryLabelDatasetMetric
import matplotlib.pyplot as plt

st.set_page_config(page_title='Loan Approval with Bias Mitigation', layout='centered')

# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv("loan_cleaned_10000_preprocessed.csv")

df = load_data()

# Split features and label
X = df.drop("Loan_Status", axis=1)
y = df["Loan_Status"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Combine X_train and y_train to create AIF360 dataset
train_df = X_train.copy()
train_df['Loan_Status'] = y_train

dataset_orig_train = StandardDataset(
    df=train_df,
    label_name='Loan_Status',
    favorable_classes=[1],
    protected_attribute_names=['Gender'],
    privileged_classes=[[1]]
)

# Reweighing for bias mitigation
RW = Reweighing(
    unprivileged_groups=[{'Gender': 0}],
    privileged_groups=[{'Gender': 1}]
)
dataset_transf_train = RW.fit_transform(dataset_orig_train)

# Prepare data for model training
X_rw = pd.DataFrame(dataset_transf_train.features, columns=X.columns)
y_rw = pd.Series(dataset_transf_train.labels.ravel())

# Train classifier
model = RandomForestClassifier()
model.fit(X_rw, y_rw)

# Streamlit App
st.title("🔍 Loan Approval Prediction with Bias Mitigation")

# Sidebar inputs
gender = st.selectbox("Gender", ["Male", "Female"])
married = st.selectbox("Married", ["Yes", "No"])
education = st.selectbox("Education", ["Graduate", "Not Graduate"])
income = st.number_input("Applicant Income", min_value=0, value=5000)
loan_amt = st.number_input("Loan Amount", min_value=0, value=150)
credit = st.selectbox("Credit History", [1.0, 0.0])

# Encode inputs
gender_val = 1 if gender == "Male" else 0
married_val = 1 if married == "Yes" else 0
education_val = 1 if education == "Graduate" else 0

if st.button("Predict Loan Approval"):
    user_input = pd.DataFrame([{
        'Gender': gender_val,
        'Married': married_val,
        'Education': education_val,
        'ApplicantIncome': income,
        'LoanAmount': loan_amt,
        'Credit_History': credit
    }])

    prediction = model.predict(user_input)[0]
    prediction_proba = model.predict_proba(user_input)[0][1]

    st.subheader("🔔 Prediction Result:")
    if prediction == 1:
        st.success(f"✅ Loan Approved (Confidence: {prediction_proba:.2f})")
    else:
        st.error(f"❌ Loan Rejected (Confidence: {1 - prediction_proba:.2f})")

    # Bias metrics calculation
    metric_before = BinaryLabelDatasetMetric(
        dataset_orig_train,
        privileged_groups=[{'Gender': 1}],
        unprivileged_groups=[{'Gender': 0}]
    )
    spd_before = metric_before.statistical_parity_difference()

    metric_after = BinaryLabelDatasetMetric(
        dataset_transf_train,
        privileged_groups=[{'Gender': 1}],
        unprivileged_groups=[{'Gender': 0}]
    )
    spd_after = metric_after.statistical_parity_difference()

    # Plot SPD comparison
    st.subheader("📊 Statistical Parity Difference (SPD)")
    st.write("Closer to 0 = Less Bias")

    fig, ax = plt.subplots()
    ax.bar(["Before Mitigation", "After Mitigation"], [spd_before, spd_after], color=["red", "green"])
    ax.set_ylabel("SPD")
    ax.set_ylim(-1, 1)
    st.pyplot(fig)

    # Show SPD values
    st.markdown(f"**SPD Before Mitigation:** `{spd_before:.4f}`")
    st.markdown(f"**SPD After Mitigation:** `{spd_after:.4f}`")
