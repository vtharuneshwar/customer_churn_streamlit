import streamlit as st
import pandas as pd
import joblib

st.set_page_config(layout="wide", page_title="Customer Churn Prediction System")

# =========================
# Load Model & Artifacts
# =========================
model = joblib.load("churn_model.pkl")
preprocessor = joblib.load("preprocessor.pkl")
model_columns = joblib.load("model_columns.pkl")
cat_cols = joblib.load("cat_cols.pkl")
num_cols = joblib.load("num_cols.pkl")

# Load model metrics (for later use)
metrics_df = pd.read_csv("model_metrics.csv")

# =========================
# Sidebar Navigation
# =========================
st.sidebar.title("📌 Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Prediction", "Dataset Overview", "EDA", "Model Performance", "Conclusion"]
)

st.title("📱 Telecom Customer Churn Prediction System")

# =========================
# Prediction Page
# =========================
if page == "Prediction":
    st.header("🔮 Churn Prediction")

    st.write("Enter customer details to predict churn probability.")

    col1, col2 = st.columns(2)

    with col1:
        tenure = st.number_input("Tenure Months", 0, 100, 12)
        monthly = st.number_input("Monthly Charges", 0.0, 500.0, 70.0)
        total = st.number_input("Total Charges", 0.0, 20000.0, 1000.0)

    with col2:
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])

    if st.button("Predict Churn"):

        # Initialize default input
        input_dict = {}
        for col in model_columns:
            if col in num_cols:
                input_dict[col] = 0
            else:
                input_dict[col] = "Unknown"

        # Fill user inputs
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

# =========================
# Dataset Overview Page
# =========================
elif page == "Dataset Overview":
    st.header("📊 Dataset Overview")

    st.subheader("Dataset Information")

    st.markdown("""
    **Dataset Name:** IBM Telco Customer Churn Dataset  
    **Source:** Kaggle / IBM Analytics Community  
    **Problem Type:** Binary Classification (Churn: Yes / No)  
    **Total Records:** 7,043 customers  
    **Original Features:** 33  
    **Final Features after preprocessing:** 23  
    """)

    st.subheader("Target Variable")
    st.write("**Churn** → Whether a customer leaves the service (Yes / No)")

    st.subheader("Missing Values Handling")
    st.markdown("""
    - `Total Charges` column converted to numeric  
    - Missing values filled using median  
    - `Churn Reason` dropped (text-heavy column)  
    """)

    st.subheader("Preprocessing Steps Applied")
    st.markdown("""
    ✔ Removed ID and geographical columns  
    ✔ Converted data types  
    ✔ Handled missing values  
    ✔ Encoded categorical features (One-Hot Encoding)  
    ✔ Scaled numerical features (StandardScaler)  
    ✔ Balanced classes using SMOTE  
    """)

    st.subheader("Class Imbalance (Before SMOTE)")
    st.markdown("""
    Majority class: Non-Churn customers  
    Minority class: Churn customers  

    SMOTE was applied to balance the dataset before training.
    """)

    st.info("Dataset prepared for Machine Learning with proper cleaning, encoding, scaling, and balancing.")
# =========================
# EDA Page
# =========================
elif page == "EDA":
    st.header("📈 Exploratory Data Analysis (EDA)")

    st.subheader("Key Insights from Data Analysis")

    st.markdown("""
    - Customers with **low tenure** are more likely to churn.  
    - **Month-to-month contracts** have the highest churn rate.  
    - Higher **Monthly Charges** increase churn probability.  
    - Customers without **long-term contracts** are more unstable.  
    """)

    st.subheader("Confusion Matrix (Visual Insight)")
    st.image("confusion_matrix.png", caption="Confusion Matrix of Final Random Forest Model")

    st.subheader("ROC Curve")
    st.image("roc_curve.png", caption="ROC Curve of Optimized Random Forest Model")

    st.info("These plots validate the strong classification performance and class separation ability of the model.")

# =========================
# Model Performance Page
# =========================
elif page == "Model Performance":
    st.header("🤖 Model Performance & Comparison")

    st.subheader("Model Comparison Table")
    st.dataframe(metrics_df)

    st.subheader("Why Random Forest was Selected")

    st.markdown("""
    **Random Forest performed best because:**
    - Highest Accuracy (~92.4%)
    - Best F1-Score and ROC-AUC
    - Handles non-linear relationships
    - Robust to noise and overfitting
    - Works well with mixed feature types
    - Performs well after SMOTE balancing
    """)

    st.success("Final Model: Optimized Random Forest Classifier")

# =========================
# Conclusion Page
# =========================
elif page == "Conclusion":
    st.header("📌 Conclusion & Business Insights")

    st.markdown("""
    ### Project Summary
    - Built an end-to-end Machine Learning system for Customer Churn Prediction.
    - Performed data cleaning, EDA, feature engineering, and model comparison.
    - Applied SMOTE and GridSearchCV for optimization.
    - Random Forest achieved ~92.4% accuracy and strong ROC-AUC.
    - Deployed the model using Streamlit for real-time predictions.

    ### Business Impact
    - Identify high-risk customers early.
    - Offer retention strategies such as discounts and contract upgrades.
    - Improve customer satisfaction and reduce revenue loss.

    ### Future Enhancements
    - Add customer lifetime value-based retention cost analysis.
    - Integrate real-time data pipelines.
    - Deploy with database and user authentication.
    """)

    st.balloons()
