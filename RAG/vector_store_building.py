# build_knowledge_base_gpu_kaggle.py

import pandas as pd
import numpy as np
from datetime import datetime
import time

# --- Import necessary libraries ---
from tqdm import tqdm
from pandarallel import pandarallel
import kagglehub
from kagglehub import KaggleDatasetAdapter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document

# --- Configuration ---
# Local model to use for embeddings
LOCAL_EMBEDDING_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
EMBEDDING_DEVICE = "cuda"  # 'cuda' for NVIDIA GPU, 'cpu' if no GPU

# Kaggle dataset details
KAGGLE_DATASET_OWNER = "abdelrahmanmahmoud22"
KAGGLE_DATASET_NAME = "final-balanced-data"
KAGGLE_FILE_PATH = "synthetic_balanced_data.csv"

# Output path for the built index
FAISS_INDEX_PATH = "faiss_fraud_index_gpu_local_model_kaggle"

# Initialize Pandarallel for CPU-based document creation
pandarallel.initialize(progress_bar=True, verbose=-1)



def load_data_from_kaggle():
    print(f"--> [1/4] Loading data from Kaggle: '{KAGGLE_DATASET_OWNER}/{KAGGLE_DATASET_NAME}'...")
    start_load_time = time.time()
    try:
        df = kagglehub.load_dataset(
            KaggleDatasetAdapter.PANDAS,
            f"{KAGGLE_DATASET_OWNER}/{KAGGLE_DATASET_NAME}",
            KAGGLE_FILE_PATH,
        )
        end_load_time = time.time()
        print(f"    ...Done in {end_load_time - start_load_time:.2f}s. Shape: {df.shape}")
        return df
    except Exception as e:
        print(f"\nCRITICAL ERROR: Failed to load data from Kaggle.")
        print(f"Please ensure:")
        print(f"1. You have run 'pip install kagglehub[pandas-datasets]'")
        print(f"2. Your 'kaggle.json' API token is correctly placed in the ~/.kaggle/ directory.")
        print(f"3. The dataset path is correct: '{KAGGLE_DATASET_OWNER}/{KAGGLE_DATASET_NAME}'")
        print(f"Error details: {e}")
        return None

# --- Data Preprocessing ---
def preprocess_data(df):
    print("--> Preprocessing data...")
    if 'trans_date_trans_time' in df.columns:
        df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'], errors='coerce')
    else:
        df['trans_date_trans_time'] = pd.NaT

    expected_cols_defaults = {
        'merchant': 'Unknown', 'category': 'Unknown', 'amt': 0.0, 'is_fraud': 0,
        'trans_date_trans_time': pd.NaT, 'cc_num': "0000", 'trans_num': "N/A",
        'state': 'Unknown', 'Region': 'Unknown', 'job_category': 'Unknown',
        'city_pop': 100000, 'age': 35, 'gender': "F",
    }
    for col, default_val in expected_cols_defaults.items():
        if col not in df.columns:
            df[col] = default_val

    for col_fill_mode in ['merchant', 'category', 'state', 'Region', 'job_category', 'gender']:
        if col_fill_mode in df.columns and df[col_fill_mode].isnull().any():
            mode_val = df[col_fill_mode].mode()[0] if not df[col_fill_mode].mode().empty else expected_cols_defaults.get(col_fill_mode, 'Unknown')
            df[col_fill_mode] = df[col_fill_mode].fillna(mode_val)
    
    for col_fill_median in ['city_pop', 'age']:
        if col_fill_median in df.columns and df[col_fill_median].isnull().any():
            median_val = df[col_fill_median].median()
            df[col_fill_median] = df[col_fill_median].fillna(median_val if pd.notna(median_val) else expected_cols_defaults.get(col_fill_median, 0))

    if 'amt' in df.columns: df['amt'] = df['amt'].astype(float)
    if 'age' in df.columns: df['age'] = df['age'].astype(int)
    
    if 'gender' in df.columns:
        df['gender'] = df['gender'].astype(str)
        df['gender_str'] = df['gender'].apply(lambda x: "Male" if x.strip().upper().startswith("M") else ("Female"if x.strip().upper().startswith("F") else "Not Specified"))
    else:
        df['gender_str'] = "Not Specified"
    
    print("--> Data preprocessing complete.")
    return df


# --- Function to be parallelized ---
def create_document_from_row(row):
    fraud_status = "which WAS LATER DETERMINED TO BE FRAUDULENT." if row.get('is_fraud', 0) == 1 else "which was determined to be a legitimate transaction."
    date_str = pd.to_datetime(row.get('trans_date_trans_time')).strftime('%Y-%m-%d %H:%M:%S') if pd.notna(row.get('trans_date_trans_time')) else "Date N/A"
    
    page_content = (f"Tx on {date_str} "
                    f"Card ...{str(row.get('cc_num','0000'))[-4:]}. Merchant: {row.get('merchant', 'N/A')}. "
                    f"Cat: {row.get('category', 'N/A')}. Amt: ${row.get('amt', 0):.2f}. Gender: {row.get('gender_str', 'N/A')}, "
                    f"Age: {row.get('age', 'N/A')}. Loc: {row.get('state', 'N/A')} (Pop {row.get('city_pop', 'N/A')}). "
                    f"Region: {row.get('Region', 'N/A')}. Job: {row.get('job_category', 'N/A')}. ID {row.get('trans_num', 'N/A')}, {fraud_status}")
    
    metadata = row.to_dict()
    for k, v in metadata.items():
        if isinstance(v, pd.Timestamp): metadata[k] = str(v)
        elif isinstance(v, pd.NaT.__class__): metadata[k] = None
        elif isinstance(v, (np.integer, np.floating)): metadata[k] = v.item()
        elif isinstance(v, (datetime)): metadata[k] = v.isoformat()
        elif not isinstance(v, (str, int, float, bool, list, dict)) and v is not None:
            metadata[k] = str(v)
            
    return Document(page_content=page_content, metadata=metadata)


def main():
    print("--- Starting Knowledge Base Build Process (Kaggle Source + Local GPU) ---")
    overall_start_time = time.time()
    
    # 1. Load Data from Kaggle and Preprocess (CPU)
    df_raw = load_data_from_kaggle()
    if df_raw is None: return
    df_knowledge = preprocess_data(df_raw)

    # 2. Create Langchain Documents (Parallel CPU)
    print(f"\n--> [2/4] Creating documents from {df_knowledge.shape[0]} records (Parallel CPU)...")
    start_doc_time = time.time()
    documents = df_knowledge.parallel_apply(create_document_from_row, axis=1).tolist()
    end_doc_time = time.time()
    print(f"    ...Done in {end_doc_time - start_doc_time:.2f}s.")

    # 3. Initialize LOCAL Embeddings Model on GPU
    print(f"\n--> [3/4] Initializing local embedding model '{LOCAL_EMBEDDING_MODEL_NAME}' on device '{EMBEDDING_DEVICE}'...")
    # This will download the model from HuggingFace Hub the first time you run it.
    model_kwargs = {'device': EMBEDDING_DEVICE}

    # ### MODIFICATION HERE ###
    # We add 'show_progress_bar': True to the encode_kwargs dictionary.
    encode_kwargs = {
        'normalize_embeddings': True, 
        'batch_size': 256, # Tune batch_size for your VRAM
    }
    # #######################

    embeddings_model = HuggingFaceEmbeddings(
        model_name=LOCAL_EMBEDDING_MODEL_NAME,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    print("    ...Model loaded.")

    # 4. Build FAISS index using the local model and GPU for indexing
    print(f"\n--> [4/4] Building FAISS index from {len(documents)} documents...")
    print("    (Embedding and Indexing will be performed on your local GPU if available).")
    start_index_time = time.time()
    
    # This single call will now show a progress bar for the embedding portion.
    vector_store = FAISS.from_documents(documents, embeddings_model)
    
    end_index_time = time.time()
    # The time printed here includes the embedding time shown in the progress bar.
    print(f"    ...Done building index in {end_index_time - start_index_time:.2f}s.")

    # 5. Save the Vector Store
    print(f"\n--> Saving final FAISS index to: {FAISS_INDEX_PATH}")
    vector_store.save_local(FAISS_INDEX_PATH)
    
    overall_end_time = time.time()
    print("\n--- Knowledge Base Build Process Complete! ---")
    print(f"✅ FAISS index successfully built and saved at '{FAISS_INDEX_PATH}'.")
    print(f"Total time taken: {overall_end_time - overall_start_time:.2f} seconds.")

if __name__ == "__main__":
    main()