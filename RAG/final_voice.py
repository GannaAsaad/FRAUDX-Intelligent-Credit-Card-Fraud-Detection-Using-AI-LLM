# fraudguard_gradio_app.py
# A full-featured, user-friendly Gradio App with a helpful persona, real-time feedback,
# voice I/O, and the complete RAG + XGBoost analysis pipeline.

import gradio as gr
import os
import pandas as pd
import numpy as np
from datetime import datetime
import json
import re
import joblib
from operator import itemgetter
import whisper
import pyttsx3
import tempfile
from scipy.io.wavfile import read as read_wav

# Langchain and Google imports
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI, HarmCategory, HarmBlockThreshold
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

# --- Configuration ---
EMBEDDING_MODEL_NAME = "models/embedding-001"
LLM_MODEL_NAME = "gemini-1.5-flash-latest"
TOP_K_RETRIEVED = 5
FAISS_INDEX_PATH = "faiss_vector_store"
LLM_TEMPERATURE = 0.5
XGB_ARTIFACTS_PATH = "xgb_artifacts"
WHISPER_MODEL_NAME = "base"

XGB_CATEGORIES = ['misc_net', 'grocery_pos', 'entertainment', 'gas_transport', 'misc_pos', 'grocery_net', 'shopping_net', 'shopping_pos', 'food_dining', 'personal_care', 'health_fitness', 'travel', 'kids_pets', 'home']
XGB_JOB_CATEGORIES = ['Technology', 'Education', 'Other', 'Healthcare', 'Engineering', 'Business/Finance', 'Creative/Arts', 'Environment/Conservation', 'Legal']

# --- Safety Settings for Gemini ---
safety_settings_gemini = { HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE, HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE, HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE, HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE, }

# --- Utility & Preprocessing Functions ---
def create_transaction_text_for_query(transaction_data):
    gender_display = "Male" if str(transaction_data.get('gender', '')).upper() == "M" else ("Female" if str(transaction_data.get('gender', '')).upper() == "F" else "Not Specified")
    age_display = transaction_data.get('age', "Not Specified")
    return (f"Incoming Transaction: Occurred on {transaction_data.get('trans_date_trans_time', datetime.now()).strftime('%Y-%m-%d %H:%M:%S')}. "
            f"Merchant: {transaction_data.get('merchant', 'N/A')}. "
            f"Category: {transaction_data.get('category', 'N/A')}. Amount: ${float(transaction_data.get('amt', 0)):.2f}. "
            f"Cardholder details (if provided): Gender: {gender_display}, Age: {age_display}, State: {transaction_data.get('state', 'N/A')}.")

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
    return processed_df.reindex(columns=xgb_artifacts_dict['model_columns'], fill_value=0)

def format_retrieved_docs(docs):
    if not docs: return "No similar past transactions found."
    return "\n\n".join([f"--- Past Tx {i+1} (ID: {d.metadata.get('trans_num', 'N/A')}, Outcome: {'FRAUD' if d.metadata.get('is_fraud') == 1 else 'Legit'}) ---\n"
                        f"Details: {d.page_content}\nAmt: ${float(d.metadata.get('amt',0)):.2f}, Cat: {d.metadata.get('category','N/A')}"
                        for i, d in enumerate(docs)])

def parse_json_from_response(text_response):
    match = re.search(r"JSON_EXTRACTED_DETAILS_BLOCK:\s*(\{.*?\})", text_response, re.DOTALL | re.IGNORECASE)
    if match: json_str = match.group(1); return json.loads(json_str)
    return None

# --- User-Friendly Prompt Templates ---
TRANSACTION_FIELDS_SCHEMA = { "merchant": {"type": "string", "description": "Name of the merchant"}, "category": {"type": "string", "description": f"Category of purchase from list: {XGB_CATEGORIES}"}, "amt": {"type": "number", "description": "Transaction amount in USD"}, "state": {"type": "string", "description": "US State (2-letter code)"}, "age": {"type": "integer", "description": "Cardholder age"}, "gender": {"type": "string", "description": "Cardholder gender ('M' or 'F')"}, "city_pop": {"type": "integer", "description": "Population of the city"}, "job_category": {"type": "string", "description": f"Cardholder's job category from list: {XGB_JOB_CATEGORIES}"}, }
REQUIRED_FIELDS_FOR_ANALYSIS = ["merchant", "category", "amt"]
CONVERSATIONAL_PROMPT_TEMPLATE = """You are **Fraudy AI**, a friendly and helpful assistant for detecting credit card transaction fraud. Your personality is empathetic, clear, and reassuring. **Your Goal:** 1. **Extract Details:** Listen carefully to the user and extract transaction details. The required fields are: {required_fields_list_str}. You can also ask for optional details for better accuracy. 2. **Confirm and Clarify:** Once you have the main details, confirm them with the user in a friendly way. For example: "Okay, I have a transaction at [Merchant] for $[Amount]. Is that right?" 3. **Offer Analysis:** If you have the required details, ask for permission to analyze. For example: "I have what I need. Shall I go ahead and analyze the risk?" 4. **Trigger Analysis:** If the user agrees, end your response with the special command: ACTION_ANALYZE_TRANSACTION. **Current State:** - Details I've gathered so far: {current_transaction_details_json} - Previous Chat History: {chat_history} - User's latest message: {user_input}. **Your Friendly Reply:** (Politely interact with the user, extract details, and remember to ask for permission before outputting the ACTION_ANALYZE_TRANSACTION command. Place any extracted details in a `JSON_EXTRACTED_DETAILS_BLOCK`.)"""
chat_prompt = ChatPromptTemplate.from_template(CONVERSATIONAL_PROMPT_TEMPLATE)
RAG_PROMPT_TEMPLATE_STR = """As **Fraudy AI**, your task is to provide a clear, easy-to-understand fraud risk analysis. Avoid jargon. Be reassuring but direct. Here's the information you have: - **User's Description:** {raw_user_input_for_rag} - **Transaction Details:** {structured_transaction_details} - **Internal ML Model Signal:** {xgboost_prediction} - **Similar Past Examples:** {retrieved_context}. --- Please structure your response like this: ### Fraudy Risk Assessment **1. Overall Risk Level:** *(Choose one: **Low Risk**, **Medium Risk**, or **High Risk**.)* **2. Key Reasons for this Assessment:** *(In simple bullet points, explain *why*. For example:)* - The transaction amount is unusually high for this category. - This purchase is in a different state than usual. **3. My Recommendation:** *(Give a clear, actionable next step.)* - **For Low Risk:** "This transaction looks normal. I don't see any cause for concern." - **For Medium Risk:** "This is a bit unusual. It would be a good idea to quickly verify this charge with your bank or on your banking app." - **For High Risk:** "This transaction shows several signs of potential fraud. I strongly recommend you contact your bank immediately to review and possibly freeze your card." """
rag_prompt_template = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE_STR)

# --- Global Model Loading ---
print("🚀 Initializing AI Systems... This may take a moment.")
google_api_key = '' # Add your gemini api key
if not google_api_key:
    raise ValueError("🚨 GOOGLE_API_KEY environment variable not found! Please set it before running the app.")
os.environ['GOOGLE_API_KEY'] = google_api_key
try:
    whisper_model = whisper.load_model(WHISPER_MODEL_NAME)
    tts_engine = pyttsx3.init()
    embeddings_prod = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL_NAME, task_type="retrieval_document")
    vector_store_prod = FAISS.load_local(FAISS_INDEX_PATH, embeddings_prod, allow_dangerous_deserialization=True)
    retriever_prod = vector_store_prod.as_retriever(search_kwargs={"k": TOP_K_RETRIEVED})
    llm_chat_prod = ChatGoogleGenerativeAI(model=LLM_MODEL_NAME, temperature=LLM_TEMPERATURE, convert_system_message_to_human=True, safety_settings=safety_settings_gemini)
    llm_rag_prod = ChatGoogleGenerativeAI(model=LLM_MODEL_NAME, temperature=LLM_TEMPERATURE, convert_system_message_to_human=True, safety_settings=safety_settings_gemini)
    xgb_artifacts = { "model": joblib.load(os.path.join(XGB_ARTIFACTS_PATH,'xgb_model.joblib')), "target_encoder": joblib.load(os.path.join(XGB_ARTIFACTS_PATH,'target_encoder.joblib')), "scaler": joblib.load(os.path.join(XGB_ARTIFACTS_PATH,'standard_scaler.joblib')), "model_columns": joblib.load(os.path.join(XGB_ARTIFACTS_PATH,'model_columns.joblib')), "states_list": joblib.load(os.path.join(XGB_ARTIFACTS_PATH,'states_list.joblib')), "state_to_region_map": joblib.load(os.path.join(XGB_ARTIFACTS_PATH,'state_to_region_map.joblib')) }
    rag_chain_prod = ( {"retrieved_context": itemgetter("user_query_for_retrieval") | retriever_prod | format_retrieved_docs, "raw_user_input_for_rag": itemgetter("user_query_for_retrieval"), "structured_transaction_details": itemgetter("structured_transaction_details"), "xgboost_prediction": itemgetter("xgboost_prediction")} | rag_prompt_template | llm_rag_prod | StrOutputParser() )
    print("✅ All systems initialized successfully.")
except Exception as e:
    print(f"🔥 An error occurred during model loading: {e}")
    raise e

# --- Helper Functions ---
def text_to_speech_file(text):
    clean_text = text.replace("*", "").replace("📊", "").replace("---", ".")
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as fp: temp_filename = fp.name
    try:
        tts_engine.save_to_file(clean_text, temp_filename); tts_engine.runAndWait()
        return read_wav(temp_filename)
    finally:
        if os.path.exists(temp_filename): os.remove(temp_filename)

def silent_audio(): return (22050, np.zeros(100, dtype=np.int16))

def stop_audio_playback(): return gr.update(value=silent_audio()), gr.update(interactive=False)

def handle_user_interaction(audio_input, text_input, chat_history, current_transaction):
    user_input_text = ""
    yield (chat_history, current_transaction, gr.update(interactive=False, placeholder="Processing..."), gr.update(interactive=False), gr.update(interactive=False), gr.update(interactive=False))
    
    if audio_input is not None:
        try:
            yield (chat_history, current_transaction, gr.update(placeholder="Transcribing..."), None, None, None)
            transcription_result = whisper_model.transcribe(audio_input, fp16=False)
            user_input_text = transcription_result["text"].strip()
            chat_history.append([f"🎤 (Voice): *{user_input_text}*", None])
        except Exception as e:
            chat_history.append(["⚠️ Error during transcription. Please try again.", None])
    elif text_input.strip():
        user_input_text = text_input.strip()
        chat_history.append([user_input_text, None])
    else: 
        yield (chat_history, current_transaction, gr.update(interactive=True, placeholder="Or type your transaction here..."), gr.update(interactive=True), gr.update(interactive=True), gr.update(interactive=False, value=silent_audio())); return

    yield (chat_history, current_transaction, None, None, None, None)

    if user_input_text.lower() == "/reset":
        chat_history, current_transaction, _, _, bot_audio = reset_all_state()
        yield (chat_history, current_transaction, gr.update(interactive=True, placeholder="Or type your transaction here..."), gr.update(interactive=True), gr.update(interactive=False), gr.update(value=bot_audio)); return

    try:
        yield (chat_history, current_transaction, gr.update(placeholder="Thinking..."), None, None, None)
        chat_history_for_prompt = "\n".join([f"user: {h[0]}\nassistant: {h[1]}" for h in chat_history[:-1] if h[0] and h[1]])
        prompt_input_dict = { "transaction_fields_description_str": "\n".join([f"- {k}: {v['description']}" for k,v in TRANSACTION_FIELDS_SCHEMA.items()]), "required_fields_list_str": ", ".join(REQUIRED_FIELDS_FOR_ANALYSIS), "current_transaction_details_json": json.dumps(current_transaction, indent=2), "chat_history": chat_history_for_prompt, "user_input": user_input_text }
        assistant_response_text = llm_chat_prod.invoke(chat_prompt.invoke(prompt_input_dict)).content
        extracted_json = parse_json_from_response(assistant_response_text)
        if extracted_json: current_transaction.update(extracted_json)
        user_facing_response = assistant_response_text.split("JSON_EXTRACTED_DETAILS_BLOCK:")[0].strip()
    except Exception as e:
        user_facing_response = "I'm sorry, I encountered an issue connecting to my brain. Please try again."

    if "ACTION_ANALYZE_TRANSACTION" in assistant_response_text.upper():
        confirmation_response = user_facing_response.replace("ACTION_ANALYZE_TRANSACTION", "").strip()
        chat_history[-1][1] = confirmation_response
        yield (chat_history, current_transaction, gr.update(placeholder="Analyzing risk..."), None, gr.update(interactive=True), gr.update(value=text_to_speech_file(confirmation_response)))
        
        try:
            current_transaction['trans_date_trans_time'] = datetime.now()
            xgb_input_data = { "merchant": current_transaction.get("merchant", "Unknown Merchant"), "category": current_transaction.get("category", XGB_CATEGORIES[0]), "amt": float(current_transaction.get("amt", 0.0)), "age": int(current_transaction.get("age", 35)), "gender": str(current_transaction.get("gender", "F")).upper(), "job_category": current_transaction.get("job_category", XGB_JOB_CATEGORIES[0]), "state": current_transaction.get("state", xgb_artifacts['states_list'][0]), "city_pop": int(current_transaction.get("city_pop", 50000)), "trans_date_trans_time": current_transaction['trans_date_trans_time'] }
            if xgb_input_data["gender"] not in ['M', 'F']: xgb_input_data["gender"] = "F"
            processed_xgb = preprocess_input_for_xgb(xgb_input_data, xgb_artifacts)
            pred = xgb_artifacts['model'].predict(processed_xgb)
            proba = xgb_artifacts['model'].predict_proba(processed_xgb)
            risk_label = "High Risk" if pred[0] == 1 else "Low Risk"
            xgb_prediction_text = f"The internal machine learning model flagged this as **{risk_label}** with a {proba[0][1]:.0%} fraud probability."
            rag_payload = {"user_query_for_retrieval": user_input_text, "structured_transaction_details": create_transaction_text_for_query(current_transaction), "xgboost_prediction": xgb_prediction_text}
            final_analysis_response = rag_chain_prod.invoke(rag_payload)
            chat_history.append([None, final_analysis_response])
            bot_response_audio = text_to_speech_file(final_analysis_response)
            current_transaction = {}
        except Exception as e:
            error_text = "I'm sorry, I had trouble completing the full analysis."
            chat_history.append([None, error_text]); bot_response_audio = text_to_speech_file(error_text)
    else:
        chat_history[-1][1] = user_facing_response
        bot_response_audio = text_to_speech_file(user_facing_response)

    stop_btn_update = gr.update(interactive=True) if bot_response_audio[1].any() else gr.update(interactive=False)
    yield (chat_history, current_transaction, gr.update(interactive=True, placeholder="Or type your transaction here..."), gr.update(interactive=True), stop_btn_update, gr.update(value=bot_response_audio))

def reset_all_state():
    initial_message = "Hello! I'm Fraudy, your personal transaction assistant. How can I help you today?"
    initial_audio = text_to_speech_file(initial_message)
    return [[None, initial_message]], {}, {}, gr.update(interactive=False), initial_audio

# --- Gradio UI Definition ---
with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue", secondary_hue="sky"), title="Fraudy AI") as demo:
    initial_history, initial_transaction, _, stop_btn_init, initial_audio = reset_all_state()
    chat_history_state = gr.State(initial_history)
    transaction_state = gr.State(initial_transaction)
    
    gr.Markdown("# 💳 Fraudy AI: Your Voice-Powered Transaction Assistant")
    gr.Markdown("Describe a transaction by speaking or typing. I'll analyze it for potential fraud risk and give you a clear recommendation.")

    with gr.Row():
        with gr.Column(scale=2):
            chatbot = gr.Chatbot(value=initial_history, label="Conversation", bubble_full_width=False, height=550)
            bot_audio_output = gr.Audio(value=initial_audio, autoplay=True, visible=False)
        with gr.Column(scale=1):
            gr.Markdown("### 🎤 Speak or Type Your Request")
            audio_input = gr.Audio(sources=["microphone"], type="filepath", label="Record Your Voice Here")
            text_input = gr.Textbox(placeholder="Or type your transaction here...", label="Text Input", lines=2)
            with gr.Row():
                send_button = gr.Button("✅ Send", variant="primary")
                stop_button = gr.Button("🔇 Stop Speaking", variant="secondary", interactive=False)
            reset_button = gr.Button("🔄 Reset Conversation", variant="stop")

    inputs = [audio_input, text_input, chat_history_state, transaction_state]
    outputs = [chatbot, transaction_state, text_input, send_button, stop_button, bot_audio_output]
    
    action_events = [send_button.click, text_input.submit, audio_input.stop_recording]
    for event in action_events:
        event(fn=handle_user_interaction, inputs=inputs, outputs=outputs, show_progress="hidden").then(lambda: (gr.update(value=None), gr.update(value="")), outputs=[audio_input, text_input])

    stop_button.click(fn=stop_audio_playback, inputs=None, outputs=[bot_audio_output, stop_button], queue=False)
    reset_button.click(fn=reset_all_state, inputs=None, outputs=[chatbot, transaction_state, text_input, stop_button, bot_audio_output], queue=False)

if __name__ == "__main__":
    demo.queue()
    demo.launch(debug=True)