# pages/1_Documentation.py
import streamlit as st
# import pandas as pd # Uncomment if you need to display dataframes
# from PIL import Image # Uncomment if you need to display images
# import matplotlib.pyplot as plt # Uncomment if you need to display plots
# import seaborn as sns # Uncomment if you need to display plots

st.set_page_config(page_title="Project Documentation", layout="wide")

st.markdown("<h1 style='text-align: center;'>📚 Project Documentation </h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-style: italic;'>A comprehensive overview of the Credit Card Fraud Detection project, from inception to implementation.</p>", unsafe_allow_html=True)
st.markdown("---")

# Using more descriptive icons for tabs
tab_intro, tab_limitations = st.tabs([
    "📌 Objectives", "🔎 Limitations"
])

with tab_intro:
    st.header("Project Objectives")
    st.markdown("""
    This section outlines the foundational aspects of the Credit Card Fraud Detection project.
    It delves into the problem statement, the core objectives we aimed to achieve, and the overall scope of our work.

    - **Effectively addressing class imbalance** by ensuring the minority class—fraudulent transactions—is well-represented and accurately detected within the dataset.
    - **Implementing reliable strategies for handling missing data**, preserving the integrity and consistency of the detection process even in the presence of incomplete information.
    - **Applying feature selection techniques** to identify the most significant variables, thereby reducing dimensionality, eliminating noise, and enhancing both performance and interpretability.
    - **Utilizing advanced optimization algorithms to fine-tune the model**, improving accuracy, reducing false positives, and increasing adaptability to emerging fraud patterns.
    - **Incorporating Large Language Models (LLMs)** to extend detection capabilities by analyzing not only structured transaction data but also unstructured inputs such as customer communications.
    """)


with tab_limitations:
    st.header("Project Limitations")
    st.markdown('''
🔒 **Lack of Real-World Data Access**  
- **Privacy & Security Restrictions:** Financial institutions cannot share real credit card data due to strict privacy laws and data protection policies.  
- **Reliance on Synthetic Data:** Researchers are often forced to use synthetic or anonymized datasets that may not accurately represent real fraud patterns.  
- **Loss of Data Nuance:** Synthetic data lacks the subtle correlations, seasonal behaviors, and evolving techniques seen in actual fraudulent transactions.  

💻 **Computational Complexity**  
- **Massive Data Volume:** Credit card fraud detection deals with millions of transactions daily, requiring high processing capacity.  
- **Advanced Model Demands:** Techniques like deep learning, ensembles, or real-time anomaly detection require powerful compute for training and deployment.  
- **Real-Time Requirements:** Models must make accurate decisions within milliseconds, adding latency constraints.  
- **Resource-Intensive Tasks:** Feature engineering, hyperparameter tuning, and model validation significantly increase computational overhead.  

🧩 **Resource Constraints**  
- **Limited RAM & Storage:** Constrains the ability to handle large datasets or train memory-intensive models.  
- **Slow Training Times:** Lack of sufficient processing power leads to longer model training cycles and delayed experimentation.  
- **Restricted Model Choices:** Computational limits may force simpler models and reduce the scope for trying advanced techniques or fine-tuning.  

💰 **Expensive LLM API Costs**  
- **High Usage Costs:** Frequent API calls to LLMs (e.g., GPT-4, Claude) can result in substantial operational expenses.  
- **Budget Limitations:** Research projects often can't afford extensive LLM usage, limiting how often and in what depth they can be used.  
- **Optimization Pressure:** Forces teams to minimize token usage, carefully engineer prompts, and reduce request frequency—sometimes at the expense of performance.  
''')
