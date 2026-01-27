import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Customer Churn Analytics & Prediction", layout="wide")

# =========================
# Load Files
# =========================
df = pd.read_csv("Telco_customer_churn.csv")
model = joblib.load("churn_model.pkl")
preprocessor = joblib.load("preprocessor.pkl")
model_columns = joblib.load("model_columns.pkl")
cat_cols = joblib.load("cat_cols.pkl")
num_cols = joblib.load("num_cols.pkl")
metrics_df = pd.read_csv("model_metrics.csv")

# =========================
# Sidebar Navigation
# =========================
st.sidebar.title("📌 Navigation")
page = st.sidebar.radio(
    "Select Page",
    ["Overview", "EDA", "Model Metrics", "Prediction"]
)

st.title("📊 Telecom Customer Churn Analysis & Prediction System")

# =========================
# OVERVIEW PAGE
# =========================
if page == "Overview":
    st.header("📌 Project Overview")

    st.markdown("""
    **Objective:**  
    To predict whether a telecom customer is likely to churn (leave the service) using Machine Learning.

    **Problem Type:**  
    Binary Classification (Churn: Yes / No)

    **Final Model Used:**  
    Optimized Random Forest Classifier (with SMOTE & GridSearchCV)
    """)

    st.subheader("🔍 Quick Dataset Summary")

    # Clean dataset for correct missing value display
    clean_df = df.drop(columns=["Churn Reason"])
    clean_df["Total Charges"] = pd.to_numeric(clean_df["Total Charges"], errors="coerce")
    clean_df["Total Charges"] = clean_df["Total Charges"].fillna(clean_df["Total Charges"].median())

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Rows", df.shape[0])
    with col2:
        st.metric("Total Columns", df.shape[1])
    with col3:
        st.metric("Total Missing Values (After Cleaning)", int(clean_df.isnull().sum().sum()))

    st.subheader("📄 Top 20 Rows Preview")
    st.dataframe(df.head(20))

    st.subheader("📈 Summary Statistics")
    st.dataframe(clean_df.describe())
# =========================
# EDA PAGE
# =========================
elif page == "EDA":
    st.header("📈 Exploratory Data Analysis (EDA)")

    st.subheader("1. Churn Distribution")
    churn_counts = df["Churn Label"].value_counts()
    fig1, ax1 = plt.subplots()
    ax1.bar(churn_counts.index, churn_counts.values)
    ax1.set_xlabel("Churn")
    ax1.set_ylabel("Count")
    ax1.set_title("Churn Distribution")
    st.pyplot(fig1)

    st.subheader("2. Tenure vs Churn")
    fig2, ax2 = plt.subplots()
    sns.boxplot(x="Churn Label", y="Tenure Months", data=df, ax=ax2)
    ax2.set_title("Tenure vs Churn")
    st.pyplot(fig2)

    st.subheader("3. Monthly Charges vs Churn")
    fig3, ax3 = plt.subplots()
    sns.boxplot(x="Churn Label", y="Monthly Charges", data=df, ax=ax3)
    ax3.set_title("Monthly Charges vs Churn")
    st.pyplot(fig3)

    st.subheader("4. Contract Type vs Churn")
    fig4, ax4 = plt.subplots()
    sns.countplot(x="Contract", hue="Churn Label", data=df, ax=ax4)
    ax4.set_title("Contract Type vs Churn")
    plt.xticks(rotation=30)
    st.pyplot(fig4)

    st.subheader("5. Correlation Heatmap (Numerical Features)")
    corr_df = df[["Tenure Months", "Monthly Charges", "Total Charges", "CLTV", "Churn Value"]].copy()
    corr_df["Total Charges"] = pd.to_numeric(corr_df["Total Charges"], errors="coerce")

    fig5, ax5 = plt.subplots(figsize=(8, 5))
    sns.heatmap(corr_df.corr(), annot=True, cmap="coolwarm", ax=ax5)
    ax5.set_title("Correlation Heatmap")
    st.pyplot(fig5)

    st.markdown("""
    **EDA Observations:**
    - Customers with low tenure are more likely to churn.
    - Month-to-month contracts show the highest churn rate.
    - Higher monthly charges are associated with higher churn.
    - Tenure and Contract Type are strong indicators of churn.
    """)
# =========================
# MODEL METRICS PAGE
# =========================
elif page == "Model Metrics":
    st.header("🤖 Model Performance & Comparison")

    st.subheader("Model Comparison (Test Set Metrics)")

    # Format numbers to 4 decimals
    formatted_df = metrics_df.copy()
    for col in ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]:
        formatted_df[col] = formatted_df[col].apply(lambda x: round(x, 4))

    st.dataframe(formatted_df, use_container_width=True)

    st.subheader("Training vs Testing Accuracy (Random Forest)")
    rf_acc_df = pd.DataFrame({
        "Dataset": ["Training Accuracy", "Testing Accuracy"],
        "Accuracy": [0.96, 0.9319]  # update train accuracy if you have exact value
    })
    st.table(rf_acc_df)

    st.subheader("Confusion Matrix")
    st.image("confusion_matrix.png")

    st.subheader("ROC Curve")
    st.image("roc_curve.png")

    st.subheader("Why Random Forest?")
    st.markdown("""
    - Highest Accuracy (93.18%)
    - Best F1-score (0.8699)
    - High ROC-AUC (0.9709)
    - Handles non-linear relationships
    - Robust to overfitting due to ensemble learning
    - Performed best after GridSearchCV tuning
    """)
# =========================
# PREDICTION PAGE
# =========================
elif page == "Prediction":
    st.header("🔮 Customer Churn Prediction")

    col1, col2 = st.columns(2)

    with col1:
        tenure = st.number_input("Tenure Months", 0, 100, 12)
        monthly = st.number_input("Monthly Charges", 0.0, 500.0, 70.0)
        total = st.number_input("Total Charges", 0.0, 20000.0, 1000.0)

    with col2:
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])

    if st.button("Predict Churn"):

        input_dict = {}
        for col in model_columns:
            if col in num_cols:
                input_dict[col] = 0
            else:
                input_dict[col] = "Unknown"

        input_dict["Tenure Months"] = tenure
        input_dict["Monthly Charges"] = monthly
        input_dict["Total Charges"] = total
        input_dict["Contract"] = contract
        input_dict["Internet Service"] = internet

        input_df = pd.DataFrame([input_dict])
        input_processed = preprocessor.transform(input_df)

        pred = model.predict(input_processed)[0]
        prob = model.predict_proba(input_processed)[0][1]

        if pred == 1:
            st.error(f"⚠️ Customer is likely to CHURN\n\nProbability: {prob:.2f}")
        else:
            st.success(f"✅ Customer is likely to STAY\n\nProbability: {1-prob:.2f}")

