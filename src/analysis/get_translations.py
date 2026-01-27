#!/usr/bin/env python3
"""
Color term translation analysis using Jensen-Shannon divergence.
"""

import pandas as pd
import numpy as np
from scipy.spatial.distance import jensenshannon
import os
import argparse
from pathlib import Path

LANG_ABBREV = {
    'American English': 'AE', 'British English': 'BE', 'French': 'FR',
    'Greek': 'GR', 'Himba': 'HB'
}

def load_pcw_data(language, input_dir='results'):
    """Load P(color|word) data from CSV file."""
    input_dir = Path(input_dir) / "PCw_results"
    filepath = Path(input_dir) / f'PCw_{language.replace(" ", "_")}.csv'
    pcw_df = pd.read_csv(filepath, index_col=0, encoding='utf-8-sig')
    return pcw_df.values, pcw_df.columns.values

def compute_similarity_matrix(pcw_source, pcw_target):
    """Compute Jensen-Shannon similarity matrix between languages."""
    n_source, n_target = pcw_source.shape[1], pcw_target.shape[1]
    similarity_matrix = np.zeros((n_source, n_target))
    
    for i in range(n_source):
        for j in range(n_target):
            js_div = jensenshannon(pcw_source[:, i], pcw_target[:, j])
            similarity_matrix[i, j] = 1 - js_div
    
    return similarity_matrix

def save_similarity_matrix(sim_matrix, source_lang, target_lang, 
                          source_terms, target_terms, output_dir='results'):
    """Save similarity matrix as CSV."""
    output_path = Path(output_dir) / "translations_similarity_matrices"
    output_path.mkdir(parents=True, exist_ok=True)
    
    df = pd.DataFrame(sim_matrix, index=source_terms, columns=target_terms)
    filename = f"similarity_matrix_{source_lang.replace(' ', '_')}_to_{target_lang.replace(' ', '_')}.csv"
    df.to_csv(output_path / filename, encoding='utf-8-sig')

def create_translation_table(source_lang, languages, all_data, output_dir='results'):
    """Create translation table for source language."""
    source_terms = all_data[source_lang]['terms']
    rows = []
    
    for i, source_term in enumerate(source_terms):
        row = [source_term]
        for target_lang in languages:
            if target_lang != source_lang:
                sim_matrix = all_data[source_lang]['similarities'][target_lang]
                best_idx = np.argmax(sim_matrix[i])
                best_translation = all_data[target_lang]['terms'][best_idx]
                similarity = sim_matrix[i, best_idx]
                row.extend([best_translation, f"{similarity:.4f}"])
        rows.append(row)
    
    # Create headers
    headers = [LANG_ABBREV[source_lang]]
    for target_lang in languages:
        if target_lang != source_lang:
            abbrev = LANG_ABBREV[target_lang]
            headers.extend([abbrev, f"{abbrev}_similarity"])
    
    # Save table
    output_path = Path(output_dir) / "translations/ourmethod"
    output_path.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows, columns=headers)
    df.to_csv(output_path / f"{source_lang.replace(' ', '_')}_translations.csv", 
              index=False, encoding='utf-8-sig')

def main():
    parser = argparse.ArgumentParser(description='Color translation analysis')
    parser.add_argument('--all-pairs', action='store_true', required=True,
                       help='Generate all language translation files')
    parser.add_argument('--input-dir', default='results', help='Input directory')
    parser.add_argument('--output-dir', default='results', help='Output directory')
    args = parser.parse_args()
    
    languages = list(LANG_ABBREV.keys())
    all_data = {}
    
    # Load all language data
    for lang in languages:
        pcw_matrix, color_names = load_pcw_data(lang, args.input_dir)
        all_data[lang] = {'pcw': pcw_matrix, 'terms': color_names, 'similarities': {}}
    
    # Compute similarities for unique pairs
    pairs = [(languages[i], languages[j]) for i in range(len(languages)) 
             for j in range(i+1, len(languages))]
    
    for source_lang, target_lang in pairs:
        sim_matrix = compute_similarity_matrix(
            all_data[source_lang]['pcw'], all_data[target_lang]['pcw'])
        
        save_similarity_matrix(sim_matrix, source_lang, target_lang,
                              all_data[source_lang]['terms'], 
                              all_data[target_lang]['terms'], args.output_dir)
        
        # Store bidirectional similarities
        all_data[source_lang]['similarities'][target_lang] = sim_matrix
        all_data[target_lang]['similarities'][source_lang] = sim_matrix.T
    
    # Generate translation tables
    for source_lang in languages:
        create_translation_table(source_lang, languages, all_data, args.output_dir)

if __name__ == "__main__":
    main()