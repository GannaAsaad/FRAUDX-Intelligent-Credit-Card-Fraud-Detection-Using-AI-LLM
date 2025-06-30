# FRAUDX 🔐  
**Smart. Safe. Secure.**  
AI-Powered Credit Card Fraud Detection System  

## 📌 Overview  
FRAUDX is a full-stack AI system developed to tackle the growing threat of credit card fraud in the digital age. This project combines deep learning, machine learning, synthetic data generation, and Large Language Models (LLMs) to create a real-time fraud detection platform with high accuracy and low false positive rates.

---

## 🚨 Problem  
- Surge in digital payments post-COVID has escalated fraud risk.  
- Traditional rule-based systems are outdated and inaccurate.  
- High false positives undermine customer trust and system efficiency.

---

## 🎯 Objectives  
- Rapid and accurate detection of fraudulent transactions  
- Reduce false positives and operational losses  
- Improve customer trust using cutting-edge AI tools  
- Integrate LLMs for intelligent reasoning and explanation

---

## 🧠 Key Technologies  
| Category              | Tools & Methods |
|-----------------------|-----------------|
| Data Generation for handling imbalance | CTGANs (Conditional Tabular GANs) |
| Machine Learning      | Logistic Regression, XGBoost |
| Deep Learning         | AutoEncoder, LSTM, DNN |
| LLM Integration       | Gemini 1.5 Flash (7B) via LangChain |
| Vector Database       | FAISS (Facebook AI Similarity Search) |
| Front-End             | Streamlit |
| RAG Pipeline          | LangChain + Gemini + FAISS |
| Embedding             | Gemini Embedding-001 |

---

## 🧪 Architecture & Pipeline

1. **Data Preprocessing**: Cleansing and preparing credit card transaction data.
2. **CTGANs**: Used to handle class imbalance and generate synthetic fraud data.
3. **Model Training**:  
   - ML Models: Logistic Regression, XGBoost  
   - DL Models: AutoEncoder, LSTM, DNN  
4. **LLM Pipeline**:  
   - Transform structured transactions into text  
   - Embed using Gemini embeddings  
   - Search using FAISS  
   - Use Gemini via LangChain to classify and reason
5. **Web Interface**: Streamlit app for real-time fraud prediction and insights

---

## 💻 Streamlit Demo

Features:
- User input form for transaction data  
- Real-time fraud classification  
- Explanation of the decision process via LLM  
- Live interaction with LangChain and Gemini model  

> **Note**: Demo requires a valid API key for Gemini and a local or cloud-hosted Streamlit server.

---

## 🔮 Future Work
- Integrate biometric and behavioral data for more robust detection  
- Extend RAG pipeline with multimodal data (e.g., image or voice logs)  
- Deploy at scale using containerized microservices and cloud functions  
- Evaluate LLM finetuning and personalization for specific institutions  

---

## 👩‍💻 Team  
- Gannatullah Gouda  
- Abdelrahman Mahmoud  
- Ahmed Khaled  
- Ali Mohamed Ali  

---

## 📫 Contact  
Email: **gradccfd@gmail.com**

---

## 📄 License  
This project is for academic use only. Contact the team for collaboration or reuse permissions.

