# pages/1_Documentation.py
import streamlit as st
import pandas as pd
import plotly.express as px

# import pandas as pd # Uncomment if you need to display dataframes
# from PIL import Image # Uncomment if you need to display images
# import matplotlib.pyplot as plt # Uncomment if you need to display plots
# import seaborn as sns # Uncomment if you need to display plots

st.set_page_config(page_title="Project Documentation", layout="wide")

st.markdown("<h1 style='text-align: center;'>📚 Project Documentation & Insights</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-style: italic;'>A comprehensive overview of the Credit Card Fraud Detection project, from inception to implementation.</p>", unsafe_allow_html=True)
st.markdown("---")

# Using more descriptive icons for tabs
tab_related, tab_model = st.tabs([
    "🔎 Literature Review",  "⚙️ Model Performance"
])


with tab_related:
    st.header("📚 Knowledge Gap")
    st.markdown('''

This section highlights the major gaps identified in previous research related to credit card fraud detection:

---

❓ **2.2.1 Missing Data**  
- Most existing studies overlook the impact of missing data, despite its critical effect on model performance and accuracy.  
- The few studies that address it rely on basic techniques such as **deletion** or **mean imputation** [14] [15].  
- More advanced methods like **regression imputation** and **multiple imputation** remain underutilized and could significantly improve data handling.

---

⚙️ **2.2.2 Optimization**  
- Analysis of 16 research papers (see Figure 2) shows the use of optimizers like **Adam** and **Genetic Algorithms**.  
- However, other powerful techniques such as **metaheuristic algorithms** and **evaluation-based optimizations** are rarely explored.  
- These could further enhance model accuracy, efficiency, and generalization.

---

🌐 **2.2.3 Deployment (Web Application)**  
- Limited attention is given to developing real-time, user-friendly fraud detection platforms.  
- Existing tools often lack intuitive interfaces, live monitoring capabilities, and alert systems, which hinders practical implementation.

---

🧠 **2.2.4 Large Language Models (LLMs)**  
- No current studies have utilized **LLMs** for fraud detection tasks.  
- Potential applications include:
  - 🔹 Feature extraction  
  - 🔹 Data augmentation  
  - 🔹 Model explainability  
- LLMs offer a powerful opportunity to improve detection, handle class imbalance, and provide interpretable results.

---
''')

    st.header("Project Contributions & Innovations")
    st.markdown('''
This project makes several key contributions to the field of credit card fraud detection:

🧩 **Web Application Development**  
- A real-time, user-friendly web platform was built to monitor credit card transactions and issue alerts for suspicious activities, enhancing business response to potential fraud.

🤖 **Integration of Large Language Models (LLMs)**  
- An LLM was employed for both anomaly detection and interactive chatbot support.  
- The model analyzes transaction patterns to identify potential fraud and offers users insights and prevention recommendations through a conversational interface.

⚖️ **Handling Data Imbalance**  
- To address the challenge of imbalanced datasets, the project applied resampling techniques such as **SMOTE**, **ADASYN**, and **undersampling**.  
- These techniques were combined with effective feature engineering and advanced machine learning models to improve detection accuracy and robustness.

🚀 **Optimization and Hybrid Modeling**  
- The project utilized **metaheuristic optimization techniques** alongside models like **Random Forest**, **Logistic Regression**, and **Genetic Algorithms**.  
- This hybrid approach enhanced both prediction performance and computational efficiency.
''')


with tab_model:


    st.title("📊 Model Performance Comparison: Before vs After")

    # Dataset selection
    dataset_option = st.selectbox("Select Dataset", ("CCFD", "European"))

    # Load and process datasets separately
    if dataset_option == "CCFD":
        after = pd.read_csv(r'C:\Users\ZeyadaNet\Downloads\model_after_performance_comparison (2).csv')
        before = pd.read_csv(r'C:\Users\ZeyadaNet\Downloads\model_before_performance_comparison (2).csv')

        # Metric options for CCFD
        metrics = [
            'Overall Accuracy', 'Weighted Recall', 'Weighted Precision',
            'Weighted F1-score', 'ROC AUC', 'Recall_0', 'Recall_1',
            'Precision_0', 'Precision_1', 'F1-score_0', 'F1-score_1'
        ]

    elif dataset_option == "European":
        after = pd.read_csv(r'E:\ERU\Level 4\S1\ML\Project\.venv\Graduation\after_renamed_after.csv')
        before = pd.read_csv(r'E:\ERU\Level 4\S1\ML\Project\.venv\Graduation\after_renamed_before.csv')

        # Rename columns for consistency
        rename_dict = {
            'Accuracy': 'Overall Accuracy',
            'Weighted_Recall': 'Weighted Recall',
            'Weighted_Precision': 'Weighted Precision',
            'Weighted_F1': 'Weighted F1-score'
        }
        after.rename(columns=rename_dict, inplace=True)
        before.rename(columns=rename_dict, inplace=True)

        # Metric options for European dataset
        metrics = ['Overall Accuracy', 'Weighted Recall', 'Weighted Precision', 'Weighted F1-score']

    # Ensure consistent types
    after['Model Name'] = after['Model Name'].astype(str)
    before['Model Name'] = before['Model Name'].astype(str)

    # Get all model names
    all_models = sorted(set(after['Model Name'].unique()) | set(before['Model Name'].unique()))

    # User selections
    selected_models = st.multiselect("Select Models to Compare", all_models, default=all_models)
    selected_metric = st.selectbox("Select Metric", metrics)

    # Ensure the selected metric exists in both DataFrames
    if selected_metric not in after.columns or selected_metric not in before.columns:
        st.error(f"Selected metric '{selected_metric}' not found in the data. Please check your CSV files or metric selection.")
        st.stop()

    # Filter both before and after datasets
    filtered_after = after[after['Model Name'].isin(selected_models)][['Model Name', selected_metric]].copy()
    filtered_before = before[before['Model Name'].isin(selected_models)][['Model Name', selected_metric]].copy()

    # Add 'Version' column for plotting
    filtered_after['Version'] = 'After'
    filtered_before['Version'] = 'Before'

    # Concatenate for plotting
    plot_data = pd.concat([filtered_before, filtered_after], ignore_index=True)

    # 📄 Show data tables
    st.subheader("📄 Raw Data")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Before**")
        st.dataframe(filtered_before)
    with col2:
        st.markdown("**After**")
        st.dataframe(filtered_after)

    # 📊 Grouped bar chart with Plotly
    st.subheader("📈 Metric Comparison (Grouped Bar Chart)")
    fig = px.bar(
        plot_data,
        x="Model Name",
        y=selected_metric,
        color="Version",
        barmode="group",
        text=selected_metric,
        title=f"{selected_metric} Comparison on {dataset_option} Dataset"
    )
    fig.update_layout(xaxis_title="Model Name", yaxis_title=selected_metric)
    fig.update_traces(texttemplate='%{text:.3f}', textposition='outside')

    st.plotly_chart(fig, use_container_width=True)


st.markdown("---")
st.caption("This documentation provides a high-level overview. For more granular details, please refer to the project's source code and supplementary reports.")