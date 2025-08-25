import pandas as pd
import pickle
import numpy as np
import os
import glob
from sklearn.preprocessing import StandardScaler
from src.utils.interface import load_language_data

def extract_language_from_model_path(model_filepath):
    """Extract language name from model file path."""
    filename = os.path.basename(model_filepath)
    language_part = filename.replace('_CustomRotation.pkl', '')
    language = language_part.replace('_', ' ')
    return language

def predict_color_names_for_test_data(test_csv_path, model_filepath, original_data_path, 
                                    batch_size=5000, min_occurrences=4):
    """
    Predict color names for test data using trained model with minimum occurrence constraint only.
    
    Args:
        test_csv_path: Path to test CSV with J', a', b' coordinates
        model_filepath: Path to trained model
        original_data_path: Path to training data
        batch_size: Batch size for predictions
        min_occurrences: Minimum occurrences to consider a color name valid
        
    Returns:
        Dictionary with prediction results
    """
    
    # Load model
    with open(model_filepath, 'rb') as f:
        model = pickle.load(f)

    # Load test data, extract relevant columns and prepare data
    test_df = pd.read_csv(test_csv_path)
    feature_columns = ['CAM16_J_UCS', 'CAM16_a_UCS', 'CAM16_b_UCS']
    X_test = test_df[feature_columns].values
    
    # Extract language from model filepath
    language = extract_language_from_model_path(model_filepath)
    evaluator, lang_info = load_language_data(original_data_path, language)
    
    # Create and fit scaler on the language data
    X_lang, y_lang = evaluator.prepare_regression_data()
    scaler = StandardScaler()
    scaler.fit(X_lang)
    
    # Scale test features
    X_test_scaled = scaler.transform(X_test)
    
    num_samples = len(X_test_scaled)
    num_batches = (num_samples + batch_size - 1) // batch_size
    
    all_predictions = []
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, num_samples)
        batch_features = X_test_scaled[start_idx:end_idx]
        batch_predictions = model.predict(batch_features)
        all_predictions.append(batch_predictions)
    
    predictions = np.vstack(all_predictions)
    
    # Convert predictions to probabilities
    y_pred = np.maximum(predictions, 0)
    y_pred = y_pred / (np.sum(y_pred, axis=1, keepdims=True) + 1e-10)
    
    # Get peak predictions (no evaluation constraints applied)
    final_predictions = np.argmax(y_pred, axis=1)
    
    # Get initial color names and confidences
    initial_predicted_names = []
    initial_confidences = []
    
    for i, pred_idx in enumerate(final_predictions):
        if pred_idx < len(evaluator.color_names):
            initial_predicted_names.append(evaluator.color_names[pred_idx])
            initial_confidences.append(y_pred[i, pred_idx])
        else:
            initial_predicted_names.append(f"unknown_{pred_idx}")
            initial_confidences.append(0.0)
    
    initial_confidences = np.array(initial_confidences)
    
    # Calculate statistics and apply minimum occurrence filter
    unique_names, counts = np.unique(initial_predicted_names, return_counts=True)
    
    # Apply minimum occurrence filter to determine valid names
    valid_mask = counts >= min_occurrences
    valid_names = unique_names[valid_mask]
    valid_counts = counts[valid_mask]
    valid_names_set = set(valid_names)
    
    # Sort by frequency for reporting
    if len(valid_names) > 0:
        sorted_indices = np.argsort(valid_counts)[::-1]
        valid_names = valid_names[sorted_indices]
        valid_counts = valid_counts[sorted_indices]
    
    # Apply filter: reassign rare names to most common valid name or mark as invalid
    final_predicted_names = []
    final_confidences = []
    filter_reassignments = 0
    
    # Get most common valid name as fallback
    fallback_name = valid_names[0] if len(valid_names) > 0 else "invalid"
    
    for i, (name, conf) in enumerate(zip(initial_predicted_names, initial_confidences)):
        if name in valid_names_set:
            # Name is valid, keep it
            final_predicted_names.append(name)
            final_confidences.append(conf)
        else:
            # Name is invalid (< min_occurrences), reassign to fallback
            filter_reassignments += 1
            final_predicted_names.append(fallback_name)
            # Find confidence for fallback name
            fallback_idx = None
            for j, color_name in enumerate(evaluator.color_names):
                if color_name == fallback_name:
                    fallback_idx = j
                    break
            if fallback_idx is not None:
                final_confidences.append(y_pred[i, fallback_idx])
            else:
                final_confidences.append(0.0)
    
    final_confidences = np.array(final_confidences)
    
    # Save results to CSV
    output_df = test_df.copy()
    language_name_for_csv = language.replace(" ", "_")
    output_df['language'] = language_name_for_csv
    output_df['predicted_color_name'] = final_predicted_names
    output_df['confidence'] = final_confidences
    
    output_file = f'results/{language.replace(" ", "_")}_predictions.csv'
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    output_df.to_csv(output_file, index=False)
    
    return {
        'test_data': test_df,
        'predicted_names': final_predicted_names,
        'confidences': final_confidences,
        'unique_names': unique_names.tolist(),
        'valid_names': valid_names.tolist(),
        'valid_counts': valid_counts.tolist(),
        'num_unique_total': len(unique_names),
        'num_valid_names': len(valid_names),
        'total_samples': len(predictions),
        'predictions': predictions,
        'probabilities': y_pred,
        'language': language,
        'filter_reassignments': filter_reassignments,
        'output_file': output_file,
        'min_occurrences': min_occurrences
    }

def process_single_language_test(model_filepath, test_csv_path, original_data_path, 
                                min_occurrences=4):
    """
    Process a single language test with minimum occurrence constraint only.
    
    Args:
        model_filepath: Path to trained model
        test_csv_path: Path to test data CSV
        original_data_path: Path to training data
        min_occurrences: Minimum occurrences for valid color names
        
    Returns:
        Dictionary with test results
    """
    
    # Get predictions
    prediction_results = predict_color_names_for_test_data(
        test_csv_path, model_filepath, original_data_path, min_occurrences=min_occurrences
    )
    
    if prediction_results is None:
        return None
    
    print(f"   - {prediction_results['language']}:{prediction_results['num_valid_names']}")
    
    return prediction_results

def process_all_languages(models_dir="models", test_csv_path="data/test_data_cam16_ucs.csv", 
                         original_data_path="data/processed_combined_data.csv", min_occurrences=4):
    """
    Process all languages by finding all model files in the models directory.
    
    Args:
        models_dir: Directory containing trained model files
        test_csv_path: Path to test data CSV
        original_data_path: Path to training data
        min_occurrences: Minimum occurrences for valid color names
        
    Returns:
        Dictionary with results for all languages
    """
    
    # Find all model files
    model_pattern = os.path.join(models_dir, "*_CustomRotation.pkl")
    model_files = glob.glob(model_pattern)
    
    if not model_files:
        print(f"No model files found in {models_dir}")
        return {}
    
    print(f"Unique predicted colours for:")
    
    # Process each language
    all_results = {}
    successful_languages = []
    failed_languages = []
    
    for model_filepath in model_files:
        language = extract_language_from_model_path(model_filepath)
        
        try:
            results = process_single_language_test(
                model_filepath, test_csv_path, original_data_path, min_occurrences
            )
            
            if results:
                all_results[language] = results
                successful_languages.append(language)
            else:
                failed_languages.append(language)
                
        except Exception as e:
            failed_languages.append(language)
            print(f"Error processing {language}: {e}")
    
    return all_results

# Main execution
if __name__ == "__main__":
    
    # Configuration
    models_directory = "models"
    test_csv = "data/test_data_cam16_ucs.csv"
    original_data = "data/processed_combined_data.csv"
    min_occurrences = 4
    
    # Process all languages
    all_results = process_all_languages(
        models_dir=models_directory,
        test_csv_path=test_csv,
        original_data_path=original_data,
        min_occurrences=min_occurrences
    )