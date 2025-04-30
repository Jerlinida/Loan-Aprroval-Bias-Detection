import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from aif360.datasets import StandardDataset
from aif360.metrics import BinaryLabelDatasetMetric
from aif360.algorithms.preprocessing import Reweighing
import matplotlib.pyplot as plt

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("loan_cleaned.csv")
    df = df.dropna()
    return df

df = load_data()

# Encode categorical features
def preprocess(df):
    df = df.copy()
    label_encoders = {}
    for col in ['Gender', 'Married', 'Education']:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
    df['Loan_Status'] = df['Loan_Status'].map({'Y': 1, 'N': 0})
    return df, label_encoders

df_processed, encoders = preprocess(df)

# Split into train and test
X = df_processed.drop('Loan_Status', axis=1)
y = df_processed['Loan_Status']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Rebuild full dataset for AIF360
df_train = X_train.copy()
df_train['Loan_Status'] = y_train

dataset_orig_train = StandardDataset(df_train,
    label_name='Loan_Status',
    favorable_classes=[1],
    protected_attribute_names=['Gender'],
    privileged_classes=[[1]])

# Apply Reweighing
RW = Reweighing(unprivileged_groups=[{'Gender': 0}], privileged_groups=[{'Gender': 1}])
dataset_transf_train = RW.fit_transform(dataset_orig_train)

# Get balanced data for training
X_rw = pd.DataFrame(dataset_transf_train.features, columns=X_train.columns)
y_rw = pd.Series(dataset_transf_train.labels.ravel())

# Train fair model
model = RandomForestClassifier()
model.fit(X_rw, y_rw)

# Streamlit UI
st.title("Loan Approval Prediction (Bias Mitigated)")

gender = st.selectbox("Gender", ["Male", "Female"])
married = st.selectbox("Married", ["Yes", "No"])
education = st.selectbox("Education", ["Graduate", "Not Graduate"])
income = st.number_input("Applicant Income", min_value=0)
loan_amt = st.number_input("Loan Amount", min_value=0)
credit = st.selectbox("Credit History", [1.0, 0.0])

if st.button("Predict Loan Status"):
    user_input = pd.DataFrame({
        'Gender': [encoders['Gender'].transform([gender])[0]],
        'Married': [encoders['Married'].transform([married])[0]],
        'Education': [encoders['Education'].transform([education])[0]],
        'ApplicantIncome': [income],
        'LoanAmount': [loan_amt],
        'Credit_History': [credit]
    })

    prediction = model.predict(user_input)[0]

    st.subheader("📢 Result:")
    if prediction == 1:
        st.success("✅ Loan Approved")
    else:
        st.error("❌ Loan Rejected")

    # 🔽🔽🔽 ADD THE SPD PLOT BLOCK HERE 🔽🔽🔽

    from aif360.metrics import BinaryLabelDatasetMetric
    import matplotlib.pyplot as plt

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

    st.subheader("📊 Statistical Parity Difference Comparison")
    st.write("**Closer to 0 means less bias**")

    fig, ax = plt.subplots()
    ax.bar(["Before Mitigation", "After Mitigation"], [spd_before, spd_after], color=["red", "green"])
    ax.set_ylabel("SPD")
    ax.set_ylim(-1, 1)
    st.pyplot(fig)

    st.markdown(f"**SPD Before Mitigation:** `{spd_before:.4f}`")
    st.markdown(f"**SPD After Mitigation:** `{spd_after:.4f}`")



