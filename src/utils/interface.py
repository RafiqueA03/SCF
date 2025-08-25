import os
import pandas as pd
from spincam.language_model_evaluator import LanguageModelEvaluator

# ---- PARAMETERS ----
RGB_STEP = 8
RGB_GRID_FILE = 'data/sRGB_grid_step8_with_Jab.csv'
RESULTS_DIR = 'results'

def language_to_filename(language):
    """Convert language name to filename format (spaces to underscores)."""
    return language.replace(" ", "_")

def check_rgb_grid_exists():
    """Check if RGB grid with CAM16-UCS coordinates already exists."""
    return os.path.exists(RGB_GRID_FILE)

def load_existing_rgb_grid():
    """Load existing RGB grid with CAM16-UCS coordinates."""
    df = pd.read_csv(RGB_GRID_FILE)
    rgb_grid = df[['R_int', 'G_int', 'B_int']].values
    jab_coords = df[['J_lightness', 'a_prime', 'b_prime']].values
    return rgb_grid, jab_coords

def load_language_data(original_data_path, language):
    """
    Load language data, trying language-specific file first, then combined file.
    """
    lang_filename = language_to_filename(language)
    data_dir = os.path.dirname(original_data_path)
    lang_specific_file = os.path.join(data_dir, f"{lang_filename}_processed.csv")
    
    evaluator = LanguageModelEvaluator(test_grid_csv=None)
    
    if os.path.exists(lang_specific_file):
        try:
            # Use the language-specific file instead of combined file
            lang_info = evaluator.load_and_process_language_data(lang_specific_file, language, use_language_filter=False)
            lang_info['used_specific_file'] = True
            return evaluator, lang_info
            
        except Exception:
            print("Language-specific file failed, using combined data")
    
    # Fallback to combined file
    lang_info = evaluator.load_and_process_language_data(original_data_path, language, use_language_filter=True)
    lang_info['used_specific_file'] = False
    return evaluator, lang_info

def load_valid_names_from_results(results_csv_path, language, model_type="custom"):
    """Load valid names list from model evaluation results CSV."""
    results_df = pd.read_csv(results_csv_path)
    
    # Filter by language and model type
    lang_results = results_df[
        (results_df['language'] == language) & 
        (results_df['model_type'] == model_type)
    ]
    
    if len(lang_results) == 0:
        return None
    
    # Get the best model (highest combined_score)
    best_model = lang_results.loc[lang_results['combined_score'].idxmax()]
    
    # Parse valid_names_list
    valid_names_str = best_model['valid_names_list']
    valid_names = eval(valid_names_str)
    
    return valid_names

def save_interface_results(prediction_results, language):
    """Save interface-ready results with only RGB, prediction, and confidence."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    lang_filename = language_to_filename(language)
    interface_file = os.path.join(RESULTS_DIR, f"{lang_filename}_interface.csv")
    
    # Create clean interface DataFrame
    interface_df = pd.DataFrame({
        'R_int': prediction_results['valid_rgb'][:, 0].astype(int),
        'G_int': prediction_results['valid_rgb'][:, 1].astype(int),
        'B_int': prediction_results['valid_rgb'][:, 2].astype(int),
        'predicted_color_name': prediction_results['predicted_names'],
        'confidence': prediction_results['confidences'],
    })
    
    interface_df.to_csv(interface_file, index=False)
    
    return interface_file