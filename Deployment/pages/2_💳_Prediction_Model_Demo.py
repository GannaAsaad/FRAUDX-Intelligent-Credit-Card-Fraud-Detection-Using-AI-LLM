# pages/2_Prediction_Model_Demo.py
import streamlit as st
import pandas as pd
import joblib
from datetime import datetime, date as date_type, time as time_type

# --- Page Configuration ---
st.set_page_config(page_title="Fraud Prediction Demo", layout="centered") # Centered layout is better for simple pages

# --- Load Artifacts ---
@st.cache_resource
def load_artifacts():
    try:
        artifacts_data = {
            "model": joblib.load('xgb_model.joblib'),
            "target_encoder": joblib.load('target_encoder.joblib'),
            "scaler": joblib.load('standard_scaler.joblib'),
            "model_columns": joblib.load('model_columns.joblib'),
            "states_list": joblib.load('states_list.joblib'),
            "state_to_region_map": joblib.load('state_to_region_map.joblib')
        }
        artifacts_data["category_options"] = ['misc_net', 'grocery_pos', 'entertainment', 'gas_transport', 'misc_pos', 'grocery_net', 'shopping_net', 'shopping_pos', 'food_dining', 'personal_care', 'health_fitness', 'travel', 'kids_pets', 'home']
        artifacts_data["job_category_options"] = ['Technology', 'Education', 'Other', 'Healthcare', 'Engineering', 'Business/Finance', 'Creative/Arts', 'Environment/Conservation', 'Legal']
        return artifacts_data
    except FileNotFoundError as e:
        st.error(f"Artifact not found: {e}. Ensure .joblib files are present in the root directory.")
        st.stop()
    except Exception as ex:
        st.error(f"Error loading artifacts: {ex}")
        st.stop()
artifacts = load_artifacts()

# --- Preprocessing Function (remains the same) ---
def preprocess_input(data, loaded_artifacts):
    df = pd.DataFrame([data])
    df['Region'] = df['state'].map(loaded_artifacts['state_to_region_map'])
    df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])
    df["trans_year"] = df["trans_date_trans_time"].dt.year
    df["is_weekend"] = df["trans_date_trans_time"].dt.day_name().apply(lambda x: 1 if x in ["Sunday", "Saturday"] else 0)
    def day_period(x_hour):
        if 0 <= x_hour < 6: return "Night"
        elif 6 <= x_hour < 12: return "Morning"
        elif 12 <= x_hour < 18: return "Afternoon"
        else: return "Evening"
    df["day_period"] = df["trans_date_trans_time"].dt.hour.apply(day_period)
    df["trans_month"] = df["trans_date_trans_time"].dt.month_name()
    df['day_name'] = df['trans_date_trans_time'].dt.day_name()
    num_cols = ['amt', 'city_pop', 'age', 'trans_year']
    high_card_cols = ['merchant', 'category', 'job_category', 'state']
    low_card_cols = ['Region', 'day_period', 'trans_month', 'day_name']
    ready_cols = ['gender', 'is_weekend']
    X_high = loaded_artifacts['target_encoder'].transform(df[high_card_cols])
    X_low = pd.get_dummies(df[low_card_cols], drop_first=True)
    X_num = pd.DataFrame(loaded_artifacts['scaler'].transform(df[num_cols]), columns=num_cols)
    X_ready = df[ready_cols].copy()
    X_ready['gender'] = X_ready['gender'].apply(lambda x: 1 if x == 'M' else 0)
    processed_df = pd.concat([X_num, X_high, X_low, X_ready], axis=1)
    final_df = processed_df.reindex(columns=loaded_artifacts['model_columns'], fill_value=0)
    return final_df

# --- Session State ---
if 'prediction_results' not in st.session_state:
    st.session_state.prediction_results = None

# --- Page Title and Input Form ---
st.title("💳 Simple Fraud Prediction Demo")
st.markdown("Enter the transaction details below to get a fraud prediction.")

with st.form("transaction_analysis_form"):
    # Using columns inside the form for a cleaner layout
    st.subheader("Transaction Details")
    r1c1, r1c2 = st.columns(2)
    with r1c1:
        ui_amount = st.number_input("Amount ($)", 0.0, value=125.50, step=0.01, format="%.2f")
    with r1c2:
        ui_category = st.selectbox("Category", artifacts["category_options"], index=0)
    
    ui_merchant = st.text_input("Merchant", "fraud_Kozey Group")

    st.subheader("Cardholder Details")
    r2c1, r2c2, r2c3 = st.columns(3)
    with r2c1:
        ui_age = st.number_input("Age", 18, 100, value=35)
    with r2c2:
        ui_gender_display = st.radio("Gender", ["Male", "Female"], index=0, horizontal=True)
        ui_gender_backend = 'M' if ui_gender_display == "Male" else 'F'
    with r2c3:
        ui_job_category = st.selectbox("Job Category", artifacts["job_category_options"], index=0)

    st.subheader("Location & Time")
    r3c1, r3c2 = st.columns(2)
    with r3c1:
        ui_state = st.selectbox("State", artifacts['states_list'], index=artifacts['states_list'].index('CA'))
    with r3c2:
        ui_city_pop = st.number_input("City Population", 100, value=50000, step=1000)
    
    r4c1, r4c2 = st.columns(2)
    with r4c1:
        ui_date = st.date_input("Date", value=date_type(2025, 6, 9))
    with r4c2:
        ui_time = st.time_input("Time", value=time_type(14, 6))

    analyze_button = st.form_submit_button("Analyze Transaction")


# --- Prediction and Results Display ---
if analyze_button:
    ui_trans_datetime = datetime.combine(ui_date, ui_time)

    input_data_for_model = {
        'amt': ui_amount, 'category': ui_category, 'merchant': ui_merchant, 'age': ui_age,
        'gender': ui_gender_backend, 'job_category': ui_job_category, 'state': ui_state,
        'city_pop': ui_city_pop, 'trans_date_trans_time': ui_trans_datetime
    }
    
    # Store results in session state
    try:
        processed_df = preprocess_input(input_data_for_model, artifacts)
        prediction_val = artifacts['model'].predict(processed_df)
        prediction_proba_val = artifacts['model'].predict_proba(processed_df)
        
        st.session_state.prediction_results = {
            "is_fraud_prediction": prediction_val[0],
            "probability_fraud": prediction_proba_val[0][1]
        }
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        st.session_state.prediction_results = None

# This block now runs after the button is clicked and results are set
if st.session_state.prediction_results:
    results = st.session_state.prediction_results
    is_fraud = results["is_fraud_prediction"]
    prob_fraud = results["probability_fraud"]

    st.markdown("---")
    st.header("Prediction Result")

    if is_fraud == 1:
        st.error(f"🚨 Prediction: FRAUDULENT Transaction")
        st.write(f"The model is {prob_fraud:.2%} confident that this transaction is fraudulent.")
    else:
        prob_not_fraud = 1 - prob_fraud
        st.success(f"✅ Prediction: NOT a Fraudulent Transaction")
        st.write(f"The model is {prob_not_fraud:.2%} confident that this transaction is legitimate.")
        
    # Reset for next prediction if desired
    # st.session_state.prediction_results = None

# Update the sidebar info to be more generic
st.sidebar.info(
    "**Transaction Analysis:** Input details to assess fraud risk using our predictive model."
)