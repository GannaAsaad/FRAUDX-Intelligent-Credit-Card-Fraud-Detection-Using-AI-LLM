# app.py (Home Page)
import streamlit as st
from PIL import Image # For image handling
import os # To check if image files exist

# --- Page Configuration (Recommended) ---
st.set_page_config(
    page_title="Credit Fraud Detection",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Enhanced Header Section ---
header_col1, header_col2, header_col3 = st.columns([1, 3, 1])

with header_col1:
    try:
        uni_logo_path = "assets/uni_logo.png"
        if os.path.exists(uni_logo_path):
            uni_logo = Image.open(uni_logo_path)
            st.image(uni_logo, width=110)
        else:
            st.caption("University Logo Placeholder")
    except Exception as e:
        st.error(f"Error loading university logo: {e}")

with header_col2:
    st.markdown("<h1 style='text-align: center; margin-bottom: 0.1em;'>🚀 Credit Card Fraud Detection</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; font-style: italic; font-size: 1.1em;'>Leveraging Machine Learning and LLMs for Enhanced Financial Security</p>", unsafe_allow_html=True)

with header_col3:
    try:
        college_logo_path = "assets/college_logo.png"
        if os.path.exists(college_logo_path):
            college_logo = Image.open(college_logo_path)
            st.image(college_logo, width=110)
        else:
            st.caption("College Logo Placeholder")
    except Exception as e:
        st.error(f"Error loading college logo: {e}")

st.markdown("---")


# --- Project Team ---
st.header("👥 Project Team")
st.markdown("Meet the dedicated individuals who contributed to this project:")

# LinkedIn Icon SVG (Font Awesome) - You can replace with a PNG/JPG if preferred
LINKEDIN_ICON_SVG = """
<svg aria-hidden="true" focusable="false" data-prefix="fab" data-icon="linkedin-in" class="svg-inline--fa fa-linkedin-in fa-w-14" role="img" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 448 512" style="width: 1em; height: 1em; vertical-align: -0.125em;">
  <path fill="currentColor" d="M100.28 448H7.4V148.9h92.88zM53.79 108.1C24.09 108.1 0 83.5 0 53.8a53.79 53.79 0 0 1 53.79-54.3c29.7 0 53.79 24.7 53.79 54.3a53.79 53.79 0 0 1-53.79 54.3zM447.9 448h-92.68V302.4c0-34.7-.7-79.2-48.29-79.2-48.29 0-55.69 37.7-55.69 76.7V448h-92.78V148.9h89.08v40.8h1.3c12.4-23.5 42.69-48.3 87.88-48.3 94 0 111.28 61.9 111.28 142.3V448z"></path>
</svg>
"""

students_data = [
    {
        "name": "GannaTulla Asaad",
        "id": "ID: 204115",
        "image_path": "assets/students/ganna.jpg",
        "linkedin_url": "https://www.linkedin.com/in/ganna-asaad/" # Replace with actual URL
    },
    {
        "name": "Abdelrahman Mahmoud",
        "id": "ID: 214117",
        "image_path": "assets/students/abdelrahman.jpg",
        "linkedin_url": "https://www.linkedin.com/in/abdelrahmanmahmoud1/" # Replace with actual URL
    },
    {
        "name": "Ahmed Khaled",
        "id": "ID: 214030",
        "image_path": "assets/students/ahmed.jpg",
        "linkedin_url": "https://www.linkedin.com/in/abdelrahmanmahmoud1/"
    },
    {
        "name": "Ali Mohamed",
        "id": "ID: 214022",
        "image_path": "assets/students/ali.jpg",
        "linkedin_url": "https://www.linkedin.com/in/alimohamedxyz/" # Replace with actual URL
    }
]

num_students = len(students_data)
cols_per_row = min(num_students, 4)

if num_students > 0:
    student_cols = st.columns(cols_per_row)
    for i, student in enumerate(students_data):
        col_index = i % cols_per_row
        with student_cols[col_index]:
            # The outer div with text-align: center handles centering for block elements inside
            st.markdown("<div style='text-align: center; margin-bottom: 25px; padding: 10px; border-radius: 8px; background-color: var(--secondary-background-color);'>", unsafe_allow_html=True)
            try:
                student_image = Image.open(student["image_path"])
                # To ensure image itself is centered if its natural width is less than column width:
                # One way is to use st.image with use_container_width=True (already done)
                # Another is to wrap it in a div and use margin: auto, but st.image with use_container_width
                # should expand to fill the column, and then the text-align:center on the parent centers the text below.
                # If image is smaller than 150px and you fix width, then need to center image explicitly.
                # Here, we let use_container_width handle it, aiming for ~150px visual via column balancing.
                st.image(student_image, use_container_width=True, width=150) # width here acts as a max-width if use_container_width is also true, or a suggestion
            except FileNotFoundError:
                placeholder_path = "assets/students/placeholder.png"
                try:
                    placeholder_img = Image.open(placeholder_path)
                    st.image(placeholder_img, use_container_width=True, width=150, caption=f"{student['name']} (Image unavailable)")
                except FileNotFoundError:
                    st.warning(f"Image for {student['name']} not found, and placeholder 'assets/students/placeholder.png' is also missing.")
                    st.markdown(f"<div style='width:100%; max-width:150px; height:150px; border:2px dashed #ddd; display:flex; align-items:center; justify-content:center; margin: 0 auto 10px auto; text-align:center; font-style:italic; color:#888; border-radius: 8px;'>Photo <br> unavailable</div>", unsafe_allow_html=True)

            st.markdown(f"**{student['name']}**", unsafe_allow_html=True) # Already centered by parent div
            st.markdown(f"_{student['id']}_", unsafe_allow_html=True)   # Already centered by parent div

            if student.get("linkedin_url"):
                st.markdown(
                    f'<a href="{student["linkedin_url"]}" target="_blank" style="display:inline-block; margin-top:5px; color:#0A66C2; font-weight:bold; text-decoration:none;">'
                    f'{LINKEDIN_ICON_SVG} LinkedIn'
                    '</a>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown("<div style='height: 1.3em; margin-top: 5px;'></div>", unsafe_allow_html=True) # Placeholder for alignment if no LinkedIn

            st.markdown("</div>", unsafe_allow_html=True)
else:
    st.info("Student information is not available at this time.")

st.markdown("---")

# --- Supervisors ---
st.header("🧑‍🏫 Project Guidance & Supervision")
st.markdown("""
- **Dr. Asmaa Fawzy**
- **AL. Shady A. Abdelmoneim**
""")

st.markdown("---")

# --- Project Overview ---
st.header("📖 Project Overview")
st.markdown("""
Welcome to our comprehensive Credit Card Fraud Detection project!

**The Challenge:** Credit card fraud represents a persistent and rapidly evolving threat in today's digital financial landscape. With global losses projected to reach \\$49.32 billion by 2030, traditional detection methods fall short due to their rigidity, high false positive rates, and reactive nature. The timely and accurate detection of fraudulent transactions is paramount to mitigating financial losses and maintaining trust in digital payment systems.

**Our Objective:** This project is dedicated to the development and rigorous evaluation of advanced machine learning models capable of identifying fraudulent credit card transactions with exceptional accuracy. We explore a comprehensive range of methodologies, from established machine learning algorithms to the innovative integration of Large Language Models (LLMs) for sophisticated analysis and interactive user engagement.

**Application Highlights:**
*   **Comprehensive Documentation:** Access detailed narratives covering our project's foundational concepts, critical literature review, exploratory data analysis (EDA), advanced model architecture, and the strategic implementation of LLMs with Retrieval Augmented Generation (RAG).
*   **Advanced Methodology Showcase:** Explore our multi-faceted approach including data preprocessing, class imbalance handling through Conditional Tabular Generative Adversarial Networks (CTGANs), sophisticated feature selection techniques, and optimization algorithms.
*   **Predictive Model Demonstration:** Engage with an interactive showcase of our fraud detection models including XGBoost, Logistic Regression, Deep Neural Networks, LSTM, and Autoencoders using real transaction datasets.
*   **RAG-Enhanced LLM Interaction:** Utilize our intelligent chat interface to converse with our specialized LLM regarding fraud detection insights, anomaly analysis, and prevention recommendations.
*   **Real-Time Detection Platform:** Experience our user-friendly web application designed for immediate fraud detection and risk assessment.

**Key Innovations:**
*   **AI-Powered Solutions:** Integration of Machine Learning, Deep Learning, and Large Language Models for proactive, real-time fraud detection.
*   **Data Balance Mastery:** Advanced techniques including SMOTE, ADASYN, and synthetic data generation to address class imbalance challenges.
*   **Feature Engineering Excellence:** Comprehensive feature selection through SOM, LASSO, Elastic Net, Random Forest, XGBoost, and Mutual Information scoring.
*   **Optimization Integration:** Metaheuristic algorithms combined with traditional models for enhanced performance and reduced false positives.

**Project Impact:**
*   **Enhanced Security:** Contributing to a more secure financial ecosystem through improved detection accuracy and speed.
*   **Operational Excellence:** Reducing costs and optimizing resource allocation for financial institutions.
*   **Customer Protection:** Safeguarding businesses and individuals from financial losses while maintaining user trust.
*   **Strategic Alignment:** Supporting Egypt Vision 2030 and SDGs for economic growth, innovation, and digital transformation.

We believe this comprehensive application offers a clear and insightful window into our cutting-edge research, innovative methodologies, and the transformative potential of our work in revolutionizing credit card fraud detection.
""")

st.markdown("---")
st.sidebar.info("Navigate through the application sections using the options above.")

# --- Footer ---
st.markdown("---")
st.markdown("<p style='text-align: center; color: #888;'><em>Developed with Streamlit by the Project Team</em></p>", unsafe_allow_html=True)