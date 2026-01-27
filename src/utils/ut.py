import os
import pandas as pd
from spincam.language_model_evaluator import LanguageModelEvaluator

def language_to_filename(language):
    """Convert language name to filename format (spaces to underscores)."""
    return language.replace(" ", "_")

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