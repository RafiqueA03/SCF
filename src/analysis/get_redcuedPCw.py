import numpy as np
import pandas as pd
import ast
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import src.utils.get_prob as gd


def filter_by_language(df, language_name=None):
    """
    Filter the combined dataset by language.
    Parameters:
    df: pd.DataFrame - Combined dataset
    language_name: str - Name of the language to filter by (e.g., 'American English', 'British English', etc.)
    Returns:
    pd.DataFrame: Filtered dataset containing only the specified language
    """
    filtered_df = df.copy()
    filtered_df = filtered_df[filtered_df['language'] == language_name]
    
    return filtered_df

data_org = pd.read_csv("data/processed_combined_data.csv", encoding='utf-8-sig')
results_file = pd.read_csv("results/language_model_comparison_detailed.csv", encoding='utf-8-sig')
languages = ["American English", "British English", "French", "Greek", "Himba"]

# create dir name PCw_results if it does not exist
if not os.path.exists("results/PCw_results"):
    os.makedirs("results/PCw_results")
for language in languages:
    lang_data = filter_by_language(data_org, language)
    lang_results = filter_by_language(results_file, language)
    lang_results_best = lang_results.loc[lang_results['combined_score'].idxmax()]
    lang_colors = lang_results_best['valid_names_list']
    lang_accuracy = lang_results_best['cv_peak_accuracy']
    # Parse the string representation back into an actual list
    if isinstance(lang_colors, str):
        lang_colors = ast.literal_eval(lang_colors)
    PWc, PCw, Count_W, colour_names, colour_IDs = gd.condProb(lang_data)
    reduced_PCw = PCw[:, np.isin(colour_names, lang_colors)]
    # Save the reduced PCw matrix to a CSV file
    reduced_PCw_df = pd.DataFrame(reduced_PCw, index=colour_IDs, columns=colour_names[np.isin(colour_names, lang_colors)])
    reduced_PCw_df.to_csv(f"results/PCw_results/PCw_{language.replace(' ', '_')}.csv", index_label='color_id', encoding='utf-8-sig')
   

