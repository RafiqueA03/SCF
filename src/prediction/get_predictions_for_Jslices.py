#!/usr/bin/env python3
"""
Test trained color models on new data with minimum occurrence filtering.
"""

import pandas as pd
import pickle
import numpy as np
import os
import glob
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from src.utils.ut import load_language_data

def extract_language_from_path(filepath):
    """Extract language name from model file path."""
    filename = Path(filepath).name
    return filename.replace('_CustomRotation.pkl', '').replace('_', ' ')

def predict_colors(test_csv, model_path, training_data, min_occurrences=4, batch_size=5000):
    """Predict color names for test data with occurrence filtering."""
    
    # Load model and data
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    test_df = pd.read_csv(test_csv)
    X_test = test_df[['CAM16_J_UCS', 'CAM16_a_UCS', 'CAM16_b_UCS']].values
    
    # Setup language data and scaler
    language = extract_language_from_path(model_path)
    evaluator, _ = load_language_data(training_data, language)
    X_train, _ = evaluator.prepare_regression_data()
    
    scaler = StandardScaler()
    X_test_scaled = scaler.fit(X_train).transform(X_test)
    
    # Batch prediction
    predictions = []
    for i in range(0, len(X_test_scaled), batch_size):
        batch = X_test_scaled[i:i + batch_size]
        predictions.append(model.predict(batch))
    
    # Convert to probabilities
    y_pred = np.vstack(predictions)
    y_pred = np.maximum(y_pred, 0)
    y_pred = y_pred / (np.sum(y_pred, axis=1, keepdims=True) + 1e-10)
    
    # Get initial predictions
    pred_indices = np.argmax(y_pred, axis=1)
    pred_names = [evaluator.color_names[i] if i < len(evaluator.color_names) 
                  else f"unknown_{i}" for i in pred_indices]
    confidences = y_pred[np.arange(len(pred_indices)), pred_indices]
    
    # Filter by minimum occurrences
    unique_names, counts = np.unique(pred_names, return_counts=True)
    valid_names = set(unique_names[counts >= min_occurrences])
    fallback = unique_names[np.argmax(counts)] if len(unique_names) > 0 else "invalid"
    
    # Apply filter
    final_names = [name if name in valid_names else fallback for name in pred_names]
    final_confidences = []
    reassignments = 0
    
    for i, (orig_name, final_name) in enumerate(zip(pred_names, final_names)):
        if orig_name != final_name:
            reassignments += 1
            # Find confidence for reassigned name
            fallback_idx = next((j for j, n in enumerate(evaluator.color_names) if n == final_name), None)
            final_confidences.append(y_pred[i, fallback_idx] if fallback_idx else 0.0)
        else:
            final_confidences.append(confidences[i])
    
    # Save results
    output_df = test_df.copy()
    output_df['language'] = language.replace(" ", "_")
    output_df['predicted_color_name'] = final_names
    output_df['confidence'] = final_confidences
    
    output_file = f'results/predictions/{language.replace(" ", "_")}_predictions.csv'
    Path(output_file).parent.mkdir(exist_ok=True)
    output_df.to_csv(output_file, index=False)
    
    return {
        'language': language,
        'predicted_names': final_names,
        'valid_names': list(valid_names),
        'num_valid_names': len(valid_names),
        'reassignments': reassignments,
        'output_file': output_file
    }

def process_all_languages(models_dir="models", test_csv="data/test_data_cam16_ucs.csv", 
                         training_data="data/processed_combined_data.csv", min_occurrences=4):
    """Process all language models."""
    
    model_files = glob.glob(f"{models_dir}/*_CustomRotation.pkl")
    if not model_files:
        print(f"No model files found in {models_dir}")
        return {}
    
    print("Unique predicted colours for:")
    results = {}
    
    for model_path in model_files:
        language = extract_language_from_path(model_path)
        try:
            result = predict_colors(test_csv, model_path, training_data, min_occurrences)
            results[language] = result
            print(f"   - {language}: {result['num_valid_names']}")
        except Exception as e:
            print(f"Error processing {language}: {e}")
    
    return results

def main():
    """Main execution function."""
    results = process_all_languages(
        models_dir="models",
        test_csv="data/test_data_cam16_ucs.csv", 
        training_data="data/processed_combined_data.csv",
        min_occurrences=4
    )
    return results

if __name__ == "__main__":
    main()