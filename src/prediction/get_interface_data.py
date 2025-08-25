"""
Color Name Interface Generation

Generates interface-ready color predictions using trained cross-linguistic
color naming models with complete vocabulary coverage.
"""

import numpy as np
import pandas as pd
import pickle
import os
import logging
from src.utils import get_interface_grid
from sklearn.preprocessing import StandardScaler
from src.utils.interface import check_rgb_grid_exists, load_existing_rgb_grid, load_language_data, load_valid_names_from_results, save_interface_results, RGB_GRID_FILE, RGB_STEP

def ensure_all_valid_colors_used(prediction_results, evaluator, valid_names_from_eval):
    """Ensure all valid colors from evaluation appear in grid predictions."""
    
    # Get current predicted colors and missing ones
    predicted_colors = set(prediction_results['predicted_names'])
    valid_colors = set(valid_names_from_eval)
    missing_colors = valid_colors - predicted_colors
    
    if not missing_colors:
        return prediction_results  
    
    logging.info(f"Missing {len(missing_colors)} colors: {missing_colors}")
    
    # Convert evaluator.color_names to list for index() method
    color_names_list = evaluator.color_names.tolist() if hasattr(evaluator.color_names, 'tolist') else list(evaluator.color_names)
    
    # For each missing color, find best grid points to reassign
    predictions_copy = prediction_results['predictions'].copy()
    
    for missing_color in missing_colors:
        if missing_color not in color_names_list:
            logging.warning(f"Missing color '{missing_color}' not found in evaluator color names")
            continue
            
        missing_idx = color_names_list.index(missing_color)
        
        # Find grid points with highest probability for this missing color
        color_probs = predictions_copy[:, missing_idx]
        best_candidates = np.argsort(color_probs)[::-1]  # Highest first
        
        # Reassign the best candidate that isn't already using this color
        reassigned = False
        for candidate_idx in best_candidates:
            current_color = prediction_results['predicted_names'][candidate_idx]
            if current_color != missing_color:
                # Count how many times current color is used
                current_color_count = prediction_results['predicted_names'].count(current_color)
                
                # Only reassign if current color has multiple instances
                if current_color_count > 1:
                    prediction_results['predicted_names'][candidate_idx] = missing_color
                    logging.info(f"Reassigned point {candidate_idx} from '{current_color}' to '{missing_color}'")
                    reassigned = True
                    break
        
        if not reassigned:
            logging.warning(f"Could not reassign any point to '{missing_color}' - all candidates are single instances")
    
    # Recalculate statistics
    unique_names, counts = np.unique(prediction_results['predicted_names'], return_counts=True)
    prediction_results['unique_names'] = unique_names.tolist()
    prediction_results['valid_counts'] = counts.tolist()
    prediction_results['num_unique_total'] = len(unique_names)
    prediction_results['num_valid_names'] = len(unique_names)
    
    return prediction_results


def predict_color_names_for_rgb_grid(rgb_grid, cam16ucs, model_filepath, original_data_path, results_csv_path, batch_size=5000):
    """
    Predict color names for RGB grid using trained model with evaluation constraints.
    """
    logging.info(f"Loading model: {os.path.basename(model_filepath)}")
    
    # Load model
    with open(model_filepath, 'rb') as f:
        model = pickle.load(f)
    
    # Prepare features
    valid_mask = ~((cam16ucs[:, 0] == 0) & (cam16ucs[:, 1] == 0) & (cam16ucs[:, 2] == 0))
    valid_rgb = rgb_grid[valid_mask]
    valid_jab = cam16ucs[valid_mask]
    
    # Extract language from model filepath
    language = "American English"
    if "American_English" in model_filepath:
        language = "American English"
    elif "British_English" in model_filepath:
        language = "British English"
    elif "French" in model_filepath:
        language = "French"
    elif "Greek" in model_filepath:
        language = "Greek"
    elif "Himba" in model_filepath:
        language = "Himba"
    
    # Load language data
    evaluator, lang_info = load_language_data(original_data_path, language)
    
    # Create and fit scaler on the language data
    X_lang, y_lang = evaluator.prepare_regression_data()
    scaler = StandardScaler()
    scaler.fit(X_lang)
    
    X_test = valid_jab
    X_test_scaled = scaler.transform(X_test)
    
    # Make predictions in batches
    logging.info(f"Making predictions for {len(X_test_scaled):,} samples...")
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
    
    # Load valid names constraint
    valid_names_from_eval = load_valid_names_from_results(results_csv_path, language, "custom")
    
    # Convert predictions to probabilities
    y_pred = np.maximum(predictions, 0)
    y_pred = y_pred / (np.sum(y_pred, axis=1, keepdims=True) + 1e-10)
    
    # Apply evaluation constraint
    initial_peak_predictions = np.argmax(y_pred, axis=1)
    
    if valid_names_from_eval is not None:
        # Map valid names to indices
        valid_names_constraint = set(valid_names_from_eval)
        valid_indices_constraint = set()
        
        for i, name in enumerate(evaluator.color_names):
            if name in valid_names_constraint:
                valid_indices_constraint.add(i)
        
        # Apply constraint
        constrained_predictions = []
        eval_reassignments = 0
        
        for i, pred_idx in enumerate(initial_peak_predictions):
            if pred_idx in valid_indices_constraint:
                constrained_predictions.append(pred_idx)
            else:
                eval_reassignments += 1
                # Find next best valid prediction
                sorted_indices = np.argsort(y_pred[i, :])[::-1]
                for idx in sorted_indices:
                    if idx in valid_indices_constraint:
                        constrained_predictions.append(idx)
                        break
                else:
                    constrained_predictions.append(pred_idx)  # Fallback
        
        final_predictions = np.array(constrained_predictions)
    else:
        final_predictions = initial_peak_predictions
        eval_reassignments = 0
    
    # Get final color names and confidences
    final_predicted_names = []
    final_confidences = []
    
    for i, pred_idx in enumerate(final_predictions):
        if pred_idx < len(evaluator.color_names):
            final_predicted_names.append(evaluator.color_names[pred_idx])
            final_confidences.append(y_pred[i, pred_idx])
        else:
            final_predicted_names.append(f"unknown_{pred_idx}")
            final_confidences.append(0.0)
    
    final_confidences = np.array(final_confidences)
    
    # Create prediction results
    prediction_results = {
        'valid_rgb': valid_rgb,
        'valid_jab': valid_jab,
        'predicted_names': final_predicted_names,
        'confidences': final_confidences,
        'predictions': predictions,
        'probabilities': y_pred,
        'language': language,
        'eval_reassignments': eval_reassignments,
        'grid_reassignments': 0
    }
    
    # Ensure all valid colors are used
    prediction_results = ensure_all_valid_colors_used(prediction_results, evaluator, valid_names_from_eval)
    
    # Calculate final statistics
    unique_names, counts = np.unique(prediction_results['predicted_names'], return_counts=True)
    
    # Update the results with final statistics
    prediction_results.update({
        'unique_names': unique_names.tolist(),
        'valid_names': unique_names.tolist(),
        'valid_counts': counts.tolist(),
        'num_unique_total': len(unique_names),
        'num_valid_names': len(unique_names),
        'total_samples': len(predictions)
    })
    
    logging.info(f"Final results: {len(unique_names)} color names, {len(final_predictions):,} samples")
    return prediction_results


def process_language_interface(model_filepath, original_data_path, results_csv_path):
    """
    Process a single language to generate interface-ready color predictions.
    """
    
    # Load or generate RGB grid
    if check_rgb_grid_exists():
        logging.info("Loading existing RGB grid...")
        rgb_grid, cam16ucs = load_existing_rgb_grid()
    else:
        logging.info("Generating RGB grid...")
        rgb_grid, cam16ucs, _ = get_interface_grid.generate_rgb_grid(RGB_STEP)
        
        # Save for future use
        temp_df = pd.DataFrame({
            'R_int': rgb_grid[:, 0].astype(int),
            'G_int': rgb_grid[:, 1].astype(int),
            'B_int': rgb_grid[:, 2].astype(int),
            'J_lightness': cam16ucs[:, 0],
            'a_prime': cam16ucs[:, 1],
            'b_prime': cam16ucs[:, 2],
        })
        temp_df.to_csv(RGB_GRID_FILE, index=False)
        logging.info(f"RGB grid saved: {RGB_GRID_FILE}")
    
    # Get predictions
    prediction_results = predict_color_names_for_rgb_grid(
        rgb_grid, cam16ucs, model_filepath, original_data_path, results_csv_path
    )
    
    # Save interface file
    if prediction_results:
        interface_file = save_interface_results(prediction_results, prediction_results['language'])
        return interface_file, prediction_results
    
    return None, None


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    # Configuration
    original_data = "data/processed_combined_data.csv"
    results_csv = "results/language_model_comparison_detailed.csv"
    
    # Language models to process
    models = {
        "American English": "models/American_English_CustomRotation.pkl",
        "British English": "models/British_English_CustomRotation.pkl", 
        "French": "models/French_CustomRotation.pkl",
        "Greek": "models/Greek_CustomRotation.pkl",
        "Himba": "models/Himba_CustomRotation.pkl"
    }
    
    # Process all languages    
    results_summary = []
    
    for language, model_path in models.items():
        if os.path.exists(model_path):
            interface_file, predictions = process_language_interface(
                model_path, original_data, results_csv
            )
            
            if predictions:
                results_summary.append({
                    'language': language,
                    'color_names': predictions['num_valid_names'],
                    'reassignments': predictions['eval_reassignments'],
                    'samples': predictions['total_samples'],
                    'interface_file': os.path.basename(interface_file)
                })
            
        else:
            logging.warning(f"Model file not found for {language}: {model_path}")
    
    # Final summary
    if results_summary:
        print(" Interface files generated:")
        for result in results_summary:
            print(f"    -{result['language']}: {result['color_names']} colors")
    else:
        logging.info("No interface files generated - check your model paths and data")