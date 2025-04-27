from transformers import GPT2LMHeadModel, GPT2TokenizerFast
import fitz
import pandas as pd
import os
from datasets import Dataset
import joblib
import numpy as np
import get_paper
import re
import time # Added for potential timing if needed later

# Define model paths and names
def load_models():
    MODEL_DIR = './models/'
    VECTORIZER_PATH = os.path.join(MODEL_DIR, 'tfidf_vectorizer.joblib')
    MODEL_PATHS = {
        'LightGBM': os.path.join(MODEL_DIR, 'lightgbm.joblib'),
        #'Logistic Regression': os.path.join(MODEL_DIR, 'logistic_regression.joblib'),
        #'Naive Bayes': os.path.join(MODEL_DIR, 'naive_bayes.joblib'),
        'CatBoost': os.path.join(MODEL_DIR, 'catboost.joblib'),
        'XGBoost': os.path.join(MODEL_DIR, 'xgboost.joblib'),
    }

    # Load the TF-IDF vectorizer
    try:
        tfidf_vectorizer = joblib.load(VECTORIZER_PATH)
        print(f"Vectorizer loaded from {VECTORIZER_PATH}")
    except FileNotFoundError:
        print(f"Error: Vectorizer not found at {VECTORIZER_PATH}. Please run tfidf.py train first.")
        exit()
    except Exception as e:
        print(f"Error loading vectorizer: {e}")
        exit()

    # Load all models
    loaded_models = {}
    for model_name, model_path in MODEL_PATHS.items():
        try:
            loaded_models[model_name] = joblib.load(model_path)
            print(f"Model '{model_name}' loaded from {model_path}")
        except FileNotFoundError:
            print(f"Warning: Model file not found for {model_name} at {model_path}. Skipping this model.")
        except Exception as e:
            print(f"Error loading model {model_name} from {model_path}: {e}")

    if not loaded_models:
        print("Error: No models could be loaded. Exiting.")
        exit()
    return tfidf_vectorizer,loaded_models

def predict_with_all_models(text, vectorizer, models_dict):
    """
    Uses all loaded models to predict probabilities for the given text.

    Args:
        text (str): The input text.
        vectorizer: The loaded TF-IDF vectorizer.
        models_dict (dict): Dictionary containing the loaded model objects.

    Returns:
        dict: A dictionary where keys are model names and values are predicted probabilities (class 1).
              Returns an empty dictionary if the text is empty or None.
    """
    if not text or not text.strip():
        return {} # Return empty dict for empty text

    results = {}
    try:
        # Transform the text using the loaded vectorizer
        X_new_tfidf = vectorizer.transform([text])

        # Predict using each loaded model
        for model_name, model in models_dict.items():
            try:
                # predict_proba returns probabilities for [class 0, class 1]
                prediction_proba = model.predict_proba(X_new_tfidf)
                # We typically want the probability of the positive class (index 1)
                results[model_name] = prediction_proba[0][1]
            except Exception as e:
                print(f"Error predicting with {model_name}: {e}")
                results[model_name] = "Error"
    except Exception as e:
         print(f"Error during TF-IDF transformation or prediction loop: {e}")
         # Return error state for all models if transform fails
         for model_name in models_dict.keys():
             results[model_name] = "Error"

    return results

# Removed old sentence_prediction function as it's replaced by predict_with_all_models

def get_text(path):
    """
    Extracts text from a PDF, splitting it before and after 'References'.
    Handles cases where 'References' might not be found.
    """
    text_before_ref = ""
    text_after_ref = ""
    try:
        pdf = fitz.open(path)
        full_text_list = [page.get_text() for page in pdf]
        full_text = ' '.join(full_text_list)
        pdf.close() # Close the pdf file
        
        # Clean up spacing
        full_text = re.sub(r'\s+', ' ', full_text).strip()

        # Find "References" section (case-insensitive search is more robust)
        ref_match = re.search(r'References|REFERENCES', full_text)

        if ref_match:
            ref_index = ref_match.start()
            ref_keyword_len = len(ref_match.group(0)) # Get actual length of found keyword
            text_before_ref = full_text[:ref_index].strip()
            text_after_ref = full_text[ref_index + ref_keyword_len:].strip()
        else:
            # If "References" not found, consider the whole text as "before references"
            print(f"  - 'References' section not found in {os.path.basename(path)}.")
            text_before_ref = full_text
            text_after_ref = "" # No text after references if keyword not found

    except Exception as e:
        print(f"Error processing PDF {path}: {e}")
        # Return empty strings on error
        text_before_ref = ""
        text_after_ref = ""

    return text_before_ref, text_after_ref
def get_all_pdf_files(root_dir):
    """递归获取目录下所有PDF文件的完整路径"""
    pdf_files = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.lower().endswith('.pdf'):
                pdf_files.append(os.path.join(root, file))
    return pdf_files
def predict_list():
    tfidf_vectorizer,loaded_models = load_models()
    paper_dir = 'batch_generated_papers/'
    pdf_files = get_all_pdf_files(paper_dir)
    if not pdf_files:
        print(f"No PDF files found in '{paper_dir}'.")
        exit()

    print(f"Found {len(pdf_files)} PDF files in '{paper_dir}'. Processing...")

    for pdf_file in pdf_files:
        print(f"\n--- Processing: {pdf_file} ---")

        text_without_ref, text_with_ref = get_text(pdf_file)

        # Predict for text WITHOUT references
        print("\nResults for Text WITHOUT References:")
        if text_without_ref:
            results_without_ref = predict_with_all_models(text_without_ref, tfidf_vectorizer, loaded_models)
            if results_without_ref:
                 for model_name, probability in results_without_ref.items():
                     if isinstance(probability, str): # Handle error case
                         print(f"  {model_name}: {probability}")
                     else:
                         print(f"  {model_name}: {probability:.4f}")
            else:
                 print("  (No text found or error during processing)")
        else:
            print("  (No text before 'References' found or error processing PDF)")

    print("\n--- Processing Complete ---")
def ml_pdf_detect(pdf_path):
    tfidf_vectorizer,loaded_models = load_models()
    text_without_ref, text_with_ref = get_text(pdf_path)
    print("\nResults for Text WITHOUT References:")
    if text_without_ref:
        results_without_ref = predict_with_all_models(text_without_ref, tfidf_vectorizer, loaded_models)
        if results_without_ref:
                for model_name, probability in results_without_ref.items():
                    if isinstance(probability, str): # Handle error case
                        print(f"  {model_name}: {probability}")
                    else:
                        print(f"  {model_name}: {probability:.4f}")
        else:
                print("  (No text found or error during processing)")
    else:
        print("  (No text before 'References' found or error processing PDF)")
    ml_score = results_without_ref['LightGBM']*0.2+results_without_ref['CatBoost']*0.3+results_without_ref['XGBoost']*0.5
    return ml_score

def ml_text_detect(text):
    tfidf_vectorizer,loaded_models = load_models()
    if text:
        results = predict_with_all_models(text, tfidf_vectorizer, loaded_models)
        if results:
            for model_name, probability in results.items():
                if isinstance(probability, str): # Handle error case
                    print(f"  {model_name}: {probability}")
                else:
                    print(f"  {model_name}: {probability:.4f}")
    else:
         print("  (No text found or error during processing)")
    ml_score = results['LightGBM']*0.2+results['CatBoost']*0.3+results['XGBoost']*0.5
    return ml_score

if __name__ == '__main__':
    pdf_path = r'paper\fakep2.pdf'
    ml_pdf_detect(pdf_path)
    #ml_pdf_detect(r'paper\1511.08458v2.pdf')
    #predict_list()


