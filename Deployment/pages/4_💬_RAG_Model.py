# pages/3_💬_RAG_Model.py
import streamlit as st
import os
import pandas as pd
import numpy as np
import time
from datetime import datetime
import json
import re
import joblib
from operator import itemgetter

from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI, HarmCategory, HarmBlockThreshold
from langchain.docstore.document import Document
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

# --- Page Configuration ---
st.set_page_config(page_title="Fraudy Chat", layout="wide")

# --- Page Title and Introduction ---
st.markdown("<h1 style='text-align: center;'>💬 Fraudy: Advanced RAG + XGBoost</h1>", unsafe_allow_html=True)
st.markdown("""
<p style='text-align: center; font-size: 1.1em;'>
I'm Fraudy, an assistant for assessing credit card transaction risk. I use Retrieval-Augmented Generation (RAG) to consult a knowledge base of past transactions and a pre-trained <strong>XGBoost model</strong> for a quantitative risk score.
<br>
<strong>Describe a transaction to begin (merchant, amount, category). I understand multiple languages!</strong> Type <code>/reset</code> to start over.
</p>
""", unsafe_allow_html=True)
st.markdown("---")

# --- Configuration ---
EMBEDDING_MODEL_NAME = "models/embedding-001" # Must match build script
LLM_MODEL_NAME = "gemini-1.5-flash-latest"
TOP_K_RETRIEVED = 5
FAISS_INDEX_PATH = r"E:\ERU\Level 4\S1\ML\Project\.venv\Graduation\LLM\final_llm\faiss_vector_store" # DIRECTORY where the pre-built index is stored
LLM_TEMPERATURE = 0.5
XGB_ARTIFACTS_PATH = "xgb_artifacts" # Path to directory with XGBoost model and artifacts

# Define specific categories and jobs for better prompting and XGB compatibility
XGB_CATEGORIES = ['misc_net', 'grocery_pos', 'entertainment', 'gas_transport', 'misc_pos', 'grocery_net', 'shopping_net', 'shopping_pos', 'food_dining', 'personal_care', 'health_fitness', 'travel', 'kids_pets', 'home']
XGB_JOB_CATEGORIES = ['Technology', 'Education', 'Other', 'Healthcare', 'Engineering', 'Business/Finance', 'Creative/Arts', 'Environment/Conservation', 'Legal']

# --- Safety Settings for Gemini ---
safety_settings_gemini = {
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
}

# --- Utility Functions ---
def create_transaction_text_for_query(transaction_data):
    gender_display = "Male" if str(transaction_data.get('gender', '')).upper() == "M" else \
                     ("Female" if str(transaction_data.get('gender', '')).upper() == "F" else "Not Specified")
    age_display = transaction_data.get('age', "Not Specified")
    cc_last4_display = transaction_data.get('cc_num_last4', "XXXX")
    state_display = transaction_data.get('state', "Not Specified")
    return (f"Incoming Transaction: Occurred on {transaction_data.get('trans_date_trans_time', datetime.now()).strftime('%Y-%m-%d %H:%M:%S')}. "
            f"Merchant: {transaction_data.get('merchant', 'N/A')}. "
            f"Category: {transaction_data.get('category', 'N/A')}. Amount: ${float(transaction_data.get('amt', 0)):.2f}. "
            f"Cardholder details (if provided): Last 4 Digits: {cc_last4_display}, Gender: {gender_display}, Age: {age_display}, State: {state_display}.")

# --- XGBoost Model Artifacts & Preprocessing ---
@st.cache_resource
def load_xgb_artifacts(artifacts_path):
    st.sidebar.info("Loading XGBoost artifacts...")
    if not os.path.exists(artifacts_path):
        st.sidebar.error(f"🛑 XGBoost artifacts directory not found: '{artifacts_path}'.")
        st.error(f"CRITICAL ERROR: XGBoost artifacts directory not found at '{artifacts_path}'. Enhanced prediction features will be disabled.")
        return None
    try:
        loaded_artifacts = {
            "model": joblib.load(os.path.join(artifacts_path,'xgb_model.joblib')),
            "target_encoder": joblib.load(os.path.join(artifacts_path,'target_encoder.joblib')),
            "scaler": joblib.load(os.path.join(artifacts_path,'standard_scaler.joblib')),
            "model_columns": joblib.load(os.path.join(artifacts_path,'model_columns.joblib')),
            "states_list": joblib.load(os.path.join(artifacts_path,'states_list.joblib')),
            "state_to_region_map": joblib.load(os.path.join(artifacts_path,'state_to_region_map.joblib'))
        }
        st.sidebar.success("✅ XGBoost artifacts loaded.")
        return loaded_artifacts
    except FileNotFoundError as e:
        st.sidebar.error(f"🛑 XGBoost artifact not found: {e}. Ensure all artifacts are in '{artifacts_path}'.")
        st.error(f"CRITICAL ERROR: XGBoost artifact not found: {e}. Enhanced prediction features will be disabled.")
        return None
    except Exception as e:
        st.sidebar.error(f"🛑 Error loading XGBoost artifacts: {e}")
        st.error(f"CRITICAL ERROR: Could not load XGBoost artifacts: {e}. Enhanced prediction features will be disabled.")
        return None

def preprocess_input_for_xgb(data, xgb_artifacts_dict):
    df = pd.DataFrame([data])
    df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])
    df['Region'] = df['state'].map(xgb_artifacts_dict['state_to_region_map'])
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
    high_card_cols_for_xgb = ['merchant', 'category', 'job_category', 'state']
    low_card_cols_for_xgb = ['Region', 'day_period', 'trans_month', 'day_name']
    ready_cols_for_xgb = ['gender', 'is_weekend']
    df_copy_for_processing = df.copy()
    df_copy_for_processing['gender'] = df_copy_for_processing['gender'].apply(lambda x: 1 if str(x).upper() == 'M' else 0)
    X_num = pd.DataFrame(xgb_artifacts_dict['scaler'].transform(df_copy_for_processing[num_cols]), columns=num_cols)
    X_high = xgb_artifacts_dict['target_encoder'].transform(df_copy_for_processing[high_card_cols_for_xgb])
    X_low = pd.get_dummies(df_copy_for_processing[low_card_cols_for_xgb], drop_first=True)
    X_ready = df_copy_for_processing[ready_cols_for_xgb].copy()
    processed_df = pd.concat([X_num, X_high, X_low, X_ready], axis=1)
    final_df = processed_df.reindex(columns=xgb_artifacts_dict['model_columns'], fill_value=0)
    return final_df

# --- Langchain Setup (Cached & Persistent) ---
@st.cache_resource
def get_embeddings_model():
    st.info(f"Loading embeddings model: {EMBEDDING_MODEL_NAME}")
    try:
        model = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL_NAME, task_type="retrieval_document")
        st.info("Embeddings model loaded.")
        return model
    except Exception as e:
        st.error(f"Failed to load embedding model ({EMBEDDING_MODEL_NAME}): {e}. Check API key and model name.")
        return None

@st.cache_resource
def load_vector_store(_embeddings_model, index_path):
    st.sidebar.info(f"Checking for pre-built Knowledge Base at: {index_path}")
    if not os.path.exists(index_path):
        st.sidebar.error(f"🛑 Index not found at '{index_path}'")
        st.error(f"CRITICAL ERROR: The Knowledge Base index is missing. Please run the `build_knowledge_base.py` script first to create it. Application cannot proceed.")
        return None
    if not _embeddings_model:
        st.error("CRITICAL ERROR: Embeddings model not loaded, cannot load vector store.")
        return None
    try:
        vector_store = FAISS.load_local(index_path, _embeddings_model, allow_dangerous_deserialization=True)
        st.sidebar.success(f"✅ Knowledge Base loaded from {index_path}")
        return vector_store
    except Exception as e:
        st.sidebar.error(f"🛑 Failed to load index: {e}")
        st.error(f"CRITICAL ERROR: Failed to load the FAISS index from {index_path}. It might be corrupted or incompatible. Try deleting the directory and re-running `build_knowledge_base.py`. Error: {e}")
        return None

@st.cache_resource
def get_retriever(_vector_store):
    if not _vector_store: return None
    return _vector_store.as_retriever(search_kwargs={"k": TOP_K_RETRIEVED})

@st.cache_resource
def get_llm(model_name_to_use, temperature_to_use):
    st.info(f"Loading LLM: {model_name_to_use} with temp: {temperature_to_use}")
    try:
        llm = ChatGoogleGenerativeAI(model=model_name_to_use, temperature=temperature_to_use, convert_system_message_to_human=True, safety_settings=safety_settings_gemini)
        st.info("LLM loaded.")
        return llm
    except Exception as e:
        st.error(f"Failed to load LLM ({model_name_to_use}): {e}. Check API key and model name.")
        return None

# --- RAG Prompt Template ---
RAG_PROMPT_TEMPLATE_STR =("""
You are a senior fraud detection analyst. Your task is to evaluate the incoming transaction and determine its fraud risk level.
You have access to:
- Structured transaction details
- An internal fraud risk signal from an XGBoost model
- Similar past transactions retrieved using semantic similarity

Do not mention machine learning, algorithms, model names, or how scores were calculated.
Be clear, specific, and practical in your response. Use the structure below.
---
**Incoming Transaction (User Description):**
{raw_user_input_for_rag}

**Structured Details:**
{structured_transaction_details}

**Internal Risk Signal:**
{xgboost_prediction}

**Retrieved Similar Past Transactions:**
{retrieved_context}
---
**Fraud Risk Analysis**

1. **Risk Assessment**
   Choose one:
   - Low Risk
   - Medium Risk
   - High Risk
   - Very High Risk

2. **Confidence Level**
   Percent between 0–100%. Reflects your confidence based on the information provided.

3. **Reasoning**
   a. Analyze transaction behavior: merchant, category, amount, location, timing, and cardholder context.
   b. Compare with retrieved past transactions: look for fraud-like or legitimate behavior patterns.
   c. Synthesize all of the above to explain why you assigned the chosen risk level.

4. **Immediate Action Recommendation**
   Based on the risk level:
   - **High/Very High Risk**: Recommend immediate card review, contacting the bank, or freezing the account.
   - **Medium Risk**: Suggest verifying the transaction with the cardholder and monitoring closely.
   - **Low Risk**: No action needed.
""")
rag_prompt_template = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE_STR)

def format_retrieved_docs(docs):
    if not docs: return "No similar past transactions found."
    return "\n\n".join([f"--- Past Tx {i+1} (ID: {d.metadata.get('trans_num', 'N/A')}, Outcome: {'FRAUD' if d.metadata.get('is_fraud') == 1 else 'Legit'}) ---\n"
                        f"Details: {d.page_content}\nAmt: ${float(d.metadata.get('amt',0)):.2f}, Cat: {d.metadata.get('category','N/A')}"
                        for i, d in enumerate(docs)])

@st.cache_resource
def get_rag_chain(_retriever, _llm_rag):
    if not _retriever or not _llm_rag: return None
    return (
        {
            "retrieved_context": itemgetter("user_query_for_retrieval") | _retriever | format_retrieved_docs,
            "raw_user_input_for_rag": itemgetter("user_query_for_retrieval"),
            "structured_transaction_details": itemgetter("structured_transaction_details"),
            "xgboost_prediction": itemgetter("xgboost_prediction")
        }
        | rag_prompt_template
        | _llm_rag
        | StrOutputParser()
    )

# --- Transaction Fields Schema ---
TRANSACTION_FIELDS_SCHEMA = {
    "merchant": {"type": "string", "description": "Name of the merchant (e.g., 'Walmart', 'Amazon', 'Kozey Group')"},
    "category": {"type": "string", "description": f"Category of purchase. Accepted values include: {', '.join(XGB_CATEGORIES)}"},
    "amt": {"type": "number", "description": "Transaction amount in USD (e.g., 25.50)."},
    "state": {"type": "string", "description": f"US State (2-letter code). (Optional)"},
    "age": {"type": "integer", "description": "Cardholder age. (Optional)"},
    "gender": {"type": "string", "description": "Cardholder gender ('M' or 'F'). (Optional)"},
    "city_pop": {"type": "integer", "description": "Population of the city. (Optional)"},
    "job_category": {"type": "string", "description": f"Cardholder's job category. Accepted values include: {', '.join(XGB_JOB_CATEGORIES)}. (Optional)"},
}
REQUIRED_FIELDS_FOR_ANALYSIS = ["merchant", "category", "amt"]

# --- Conversational Prompt ---
CONVERSATIONAL_PROMPT_TEMPLATE = """
You are **Fraudy**, a multilingual, empathetic assistant for detecting credit card transaction fraud. You understand natural conversation, extract key transaction details, and combine **retrieval-based insights** with an **XGBoost model** to assess risk accurately.

**Core Principles**
- **Multilingual Support:** Reply in the user's language if confident; otherwise, use English.
- **Transaction Privacy:** Focus on the transaction's **merchant, amount, and category**. For enhanced model accuracy, you may request optional fields: **age, gender, job category, state, and city population**—but never insist on personal data.
- **Explain Clearly:** When analyzing a transaction, explain the risk score and decision rationale from both RAG and XGBoost.
- **Action-Driven:** Always suggest what to do next.

---

**Interaction Process**

1. **Extract Transaction Details**
    - Look for: `{transaction_fields_description_str}`
    - Required for RAG analysis: `{required_fields_list_str}`
    - Output: `JSON_EXTRACTED_DETAILS_BLOCK: <json_object_here>`
    - Confirm with: “Okay, I have a transaction at [Merchant] for [Amount] in [Category]. Is that correct?”

2. **Prepare for Risk Analysis**
    - If you have all of `{required_fields_list_str}`:
        - Check if optional fields for XGBoost are present.
        - If not, say: “I have the essentials. For a more accurate prediction using our model, additional info like age or state helps. Want to add it, or should I continue with defaults?”
        - If user agrees, update JSON. If not, proceed.
        - Confirm analysis: “Great. Shall I analyze the risk now using our knowledge base and model?”
    - If user agrees, respond with: `ACTION_ANALYZE_TRANSACTION`

3. **After Analysis**
    - Present the risk findings from both RAG and XGBoost.
    - End with: “Would you like to check another transaction?”

---
**Current State**
- Extracted Transaction: `{current_transaction_details_json}`
- Chat History (Latest): {chat_history}
- User Input: {user_input}
**Your Reply:**
(Extract and confirm fields. Use `JSON_EXTRACTED_DETAILS_BLOCK`. Speak naturally. Guide toward analysis.)
"""
chat_prompt = ChatPromptTemplate.from_template(CONVERSATIONAL_PROMPT_TEMPLATE)


# --- API Key Check ---
if 'api_key_loaded' not in st.session_state:
    google_api_key_env = os.environ.get('GOOGLE_API_KEY') or st.secrets.get("GOOGLE_API_KEY")
    if google_api_key_env:
        os.environ['GOOGLE_API_KEY'] = google_api_key_env
        st.session_state.api_key_loaded = True
        st.sidebar.success("✅ Google API Key loaded.")
    else:
        st.sidebar.error("🛑 Google API Key not found in Secrets or Environment.")
        st.error("🚨 Google API Key is not configured. Please set it as an environment variable `GOOGLE_API_KEY` or in Streamlit Cloud secrets. The app cannot proceed without it.")
        st.stop()

# --- Initialization Block ---
# Use unique key for this page to avoid conflicts in a multi-page app
if 'rag_xgb_initialized' not in st.session_state:
    with st.spinner("🚀 Initializing AI Systems, Knowledge Base & XGBoost Model..."):
        st.session_state.embeddings_rag_xgb = get_embeddings_model()
        if st.session_state.embeddings_rag_xgb is None: st.stop()
        st.session_state.vector_store_rag_xgb = load_vector_store(st.session_state.embeddings_rag_xgb, FAISS_INDEX_PATH)
        if st.session_state.vector_store_rag_xgb is None: st.stop()
        st.session_state.retriever_rag_xgb = get_retriever(st.session_state.vector_store_rag_xgb)
        if st.session_state.retriever_rag_xgb is None: st.error("Critical: Failed to create retriever."); st.stop()
        st.session_state.llm_chat_rag_xgb = get_llm(LLM_MODEL_NAME, LLM_TEMPERATURE)
        st.session_state.llm_rag_rag_xgb = get_llm(LLM_MODEL_NAME, LLM_TEMPERATURE)
        if not st.session_state.llm_chat_rag_xgb or not st.session_state.llm_rag_rag_xgb: st.error("Critical: Failed to load LLMs."); st.stop()
        st.session_state.xgb_artifacts_rag_xgb = load_xgb_artifacts(XGB_ARTIFACTS_PATH)
        st.session_state.rag_chain_rag_xgb = get_rag_chain(st.session_state.retriever_rag_xgb, st.session_state.llm_rag_rag_xgb)
        if st.session_state.rag_chain_rag_xgb is None: st.error("Critical: Failed to create RAG chain."); st.stop()
        st.session_state.rag_xgb_initialized = True
        st.sidebar.success("✅ AI Systems & XGBoost Model Initialized!")
        st.success("Fraudy is ready!")

# --- Session State Initialization for Chat (using unique keys) ---
if "messages_rag_xgb" not in st.session_state:
    st.session_state.messages_rag_xgb = [{"role": "assistant", "content": "👋 Hello! To assess transaction risk, please tell me the merchant, amount, and category of the purchase."}]
if "current_transaction_details_rag_xgb" not in st.session_state: st.session_state.current_transaction_details_rag_xgb = {}
if "last_analysis_results_rag_xgb" not in st.session_state:
    st.session_state.last_analysis_results_rag_xgb = {"query_text": "N/A", "rag_assessment": "N/A"}

def parse_json_from_response(text_response):
    match = re.search(r"JSON_EXTRACTED_DETAILS_BLOCK:\s*(\{.*?\})", text_response, re.DOTALL | re.IGNORECASE)
    if match:
        json_str = match.group(1)
        try: return json.loads(json_str)
        except json.JSONDecodeError: return None
    return None

# Display chat messages
for msg in st.session_state.messages_rag_xgb:
    with st.chat_message(msg["role"]):
        content_to_display = msg["content"].split("JSON_EXTRACTED_DETAILS_BLOCK:")[0].strip()
        st.markdown(content_to_display)

# --- Main Chat Loop ---
if user_input := st.chat_input("Describe transaction (merchant, amount, category)..."):
    st.session_state.messages_rag_xgb.append({"role": "user", "content": user_input})
    with st.chat_message("user"): st.markdown(user_input)

    if user_input.lower().strip() == "/reset":
        st.session_state.current_transaction_details_rag_xgb = {}
        st.session_state.last_analysis_results_rag_xgb = {k: "N/A" for k in st.session_state.last_analysis_results_rag_xgb}
        response = "🔄 Transaction details cleared. Let's start over!"
        st.session_state.messages_rag_xgb.append({"role": "assistant", "content": response})
        with st.chat_message("assistant"): st.markdown(response)
        st.rerun()

    with st.chat_message("assistant"):
        placeholder = st.empty()
        chat_history_for_prompt = "\n".join([f"{m['role']}: {m['content'].split('JSON_EXTRACTED_DETAILS_BLOCK:')[0].strip()}" for m in st.session_state.messages_rag_xgb[-6:-1]])
        fields_desc_list = [f"- {k}: {v['description']}" for k,v in TRANSACTION_FIELDS_SCHEMA.items()]
        
        prompt_input_dict = {
            "transaction_fields_description_str": "\n".join(fields_desc_list),
            "required_fields_list_str": ", ".join(REQUIRED_FIELDS_FOR_ANALYSIS),
            "current_transaction_details_json": json.dumps(st.session_state.current_transaction_details_rag_xgb, indent=2),
            "chat_history": chat_history_for_prompt, "user_input": user_input
        }
        
        try:
            full_chat_prompt_obj = chat_prompt.invoke(prompt_input_dict)
            with st.spinner("🤔 Fraudy is thinking..."):
                assistant_response_text = st.session_state.llm_chat_rag_xgb.invoke(full_chat_prompt_obj).content
                user_facing_response_full = assistant_response_text.split("JSON_EXTRACTED_DETAILS_BLOCK:")[0].strip()
                placeholder.markdown(user_facing_response_full)

            extracted_json = parse_json_from_response(assistant_response_text)
            if extracted_json:
                for key, value in extracted_json.items():
                    if key in TRANSACTION_FIELDS_SCHEMA:
                         st.session_state.current_transaction_details_rag_xgb[key] = value

            if "ACTION_ANALYZE_TRANSACTION" in assistant_response_text.upper():
                missing_required = [f for f in REQUIRED_FIELDS_FOR_ANALYSIS if f not in st.session_state.current_transaction_details_rag_xgb or not st.session_state.current_transaction_details_rag_xgb.get(f)]
                
                if not missing_required:
                    confirmation_msg = user_facing_response_full.replace("ACTION_ANALYZE_TRANSACTION", "").strip()
                    placeholder.markdown(confirmation_msg)
                    st.session_state.messages_rag_xgb.append({"role": "assistant", "content": confirmation_msg})
                    
                    with st.spinner("🕵️‍♀️ Consulting knowledge base, running XGBoost model, and performing RAG analysis..."):
                        current_tx_data = st.session_state.current_transaction_details_rag_xgb.copy()
                        current_tx_data['trans_date_trans_time'] = datetime.now()
                        structured_text_for_rag = create_transaction_text_for_query(current_tx_data)
                        
                        # --- XGBoost Prediction Logic ---
                        xgb_prediction_text = "XGBoost Model: Not available."
                        if st.session_state.xgb_artifacts_rag_xgb:
                            xgb_input_data = {}
                            defaults = st.session_state.xgb_artifacts_rag_xgb
                            xgb_field_map = {
                                "merchant": "Unknown Merchant", "category": XGB_CATEGORIES[0], "amt": 0.0,
                                "age": 35, "gender": "F", "job_category": XGB_JOB_CATEGORIES[0],
                                "state": defaults['states_list'][0], "city_pop": 50000
                            }
                            for field, default_val in xgb_field_map.items():
                                xgb_input_data[field] = current_tx_data.get(field, default_val)
                            xgb_input_data['trans_date_trans_time'] = current_tx_data['trans_date_trans_time']
                            if xgb_input_data["gender"] not in ['M', 'F']: xgb_input_data["gender"] = "F"
                            
                            try:
                                processed = preprocess_input_for_xgb(xgb_input_data, defaults)
                                pred = defaults['model'].predict(processed)
                                proba = defaults['model'].predict_proba(processed)
                                risk_label = "High Risk" if pred[0] == 1 else "Low Risk"
                                xgb_prediction_text = f"XGBoost Model Prediction: **{risk_label}** (Fraud Probability: {proba[0][1]:.2%})"
                            except Exception as e_xgb:
                                xgb_prediction_text = f"XGBoost Model: Error during prediction - {e_xgb}."
                        
                        rag_payload = {
                            "user_query_for_retrieval": user_input,
                            "structured_transaction_details": structured_text_for_rag,
                            "xgboost_prediction": xgb_prediction_text
                        }
                        rag_assessment = st.session_state.rag_chain_rag_xgb.invoke(rag_payload)
                        
                        analysis_summary = f"**📊 Transaction Risk Assessment Complete:**\n\n{rag_assessment}\n\n---\n*Analyze another transaction? (Type `/reset` for new)*"
                        st.session_state.messages_rag_xgb.append({"role": "assistant", "content": analysis_summary})
                        st.session_state.current_transaction_details_rag_xgb = {} # Clear for next one
                        st.rerun()
                else:
                    polite_error_msg = f"My apologies, I still need these core details: {', '.join(missing_required)}."
                    placeholder.markdown(polite_error_msg)
                    st.session_state.messages_rag_xgb.append({"role": "assistant", "content": polite_error_msg})
            else:
                st.session_state.messages_rag_xgb.append({"role": "assistant", "content": user_facing_response_full})

        except Exception as e_chat:
            st.error(f"🚨 An unexpected error occurred: {e_chat}")
            err_msg_display = "😥 I encountered a technical hiccup. Could you please rephrase or try again?"
            placeholder.markdown(err_msg_display)
            st.session_state.messages_rag_xgb.append({"role": "assistant", "content": err_msg_display})