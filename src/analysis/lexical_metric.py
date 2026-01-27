"""
Lexical Evaluation of Claude Model Translations using chrF and chrF++
"""

import pandas as pd
from sacrebleu import CHRF
import os


def normalize_text(df):
    """Apply text normalization: lowercase, replace underscores, strip, normalize spaces"""
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = (df[col].astype(str)
                       .str.replace('_', ' ', regex=False)
                       .str.lower()
                       .str.strip()
                       .str.replace(r'\s+', ' ', regex=True))
    return df


def calculate_chrf_scores(references, hypotheses, scorer):
    """Calculate chrF scores for reference-hypothesis pairs"""
    scores = []
    for ref, hyp in zip(references, hypotheses):
        if pd.isna(ref) or pd.isna(hyp):
            scores.append(0.0)
        else:
            score = scorer.sentence_score(str(hyp), [str(ref)])
            scores.append(score.score)
    return scores


def process_model(lang_name, lang_abbr, gt_col, targets, model_name, model_path, 
                  chrf_scorer, chrf_plus_scorer, output_dir):
    """Process one source language for a specific model"""
    # Load data
    gt_file = f"results/translations/ourmethod/{lang_name}_with_focal_colours.csv"
    ourmethod_data = pd.read_csv(gt_file, encoding='utf-8-sig')
    
    if model_name == 'claude':
        model_data = pd.read_excel(f"{model_path}/{lang_name}.xlsx")
    else:
        model_data = pd.read_csv(f"{model_path}/{lang_name}.csv", encoding='utf-8-sig')
    
    # Select and normalize columns
    gt_columns = [gt_col] + [t['gt_col'] for t in targets]
    model_columns = [lang_abbr] + [t['model_col'] for t in targets]
    
    ourmethod_selected = normalize_text(ourmethod_data[gt_columns].copy())
    model_selected = normalize_text(model_data[model_columns].copy())
    
    # Create results dataframe
    merged_df = pd.DataFrame()
    
    for i, target in enumerate(targets):
        gt_col_name = gt_columns[i + 1]
        model_col_name = model_columns[i + 1]
        prefix = target['abbr']
        
        merged_df[f'{prefix}_gt'] = ourmethod_selected[gt_col_name]
        merged_df[f'{prefix}_{model_name}'] = model_selected[model_col_name]
        
        # Calculate scores
        merged_df[f'{prefix}_chrF'] = calculate_chrf_scores(
            ourmethod_selected[gt_col_name],
            model_selected[model_col_name],
            chrf_scorer
        )
        
        merged_df[f'{prefix}_chrF_plus'] = calculate_chrf_scores(
            ourmethod_selected[gt_col_name],
            model_selected[model_col_name],
            chrf_plus_scorer
        )
    
    # Save results
    output_file = os.path.join(output_dir, model_name, f"{lang_name}_chrf_chrfplus_{model_name}.csv")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    merged_df.to_csv(output_file, index=False, encoding='utf-8-sig')


def main():
    """Main execution"""
    # Initialize scorers
    chrf_scorer = CHRF(char_order=6, word_order=0, beta=2, lowercase=False, 
                       whitespace=False, eps_smoothing=False)
    chrf_plus_scorer = CHRF(char_order=6, word_order=2, beta=2, lowercase=False, 
                            whitespace=False, eps_smoothing=False)
    
    # Model configurations
    models = {
        'claude': 'results/translations/claude'    }
    
    # Language configurations
    languages = [
        ('American_English', 'AE', 'AE', [
            {'abbr': 'BE', 'gt_col': 'BE', 'model_col': 'BE'},
            {'abbr': 'FR', 'gt_col': 'French_Original', 'model_col': 'FR'},
            {'abbr': 'GR', 'gt_col': 'Greek_Original', 'model_col': 'GR'},
        ]),
        ('British_English', 'BE', 'BE', [
            {'abbr': 'AE', 'gt_col': 'AE', 'model_col': 'AE'},
            {'abbr': 'FR', 'gt_col': 'French_Original', 'model_col': 'FR'},
            {'abbr': 'GR', 'gt_col': 'Greek_Original', 'model_col': 'GR'},
        ]),
        ('French', 'FR', 'French_Original', [
            {'abbr': 'AE', 'gt_col': 'AE', 'model_col': 'AE'},
            {'abbr': 'BE', 'gt_col': 'BE', 'model_col': 'BE'},
            {'abbr': 'GR', 'gt_col': 'Greek_Original', 'model_col': 'GR'},
        ]),
        ('Greek', 'GR', 'Greek_Original', [
            {'abbr': 'AE', 'gt_col': 'AE', 'model_col': 'AE'},
            {'abbr': 'BE', 'gt_col': 'BE', 'model_col': 'BE'},
            {'abbr': 'FR', 'gt_col': 'French_Original', 'model_col': 'FR'},
        ]),
    ]
    
    output_dir = 'results/metrics/lexical'
    
    # Process all languages for all models
    for lang_name, lang_abbr, gt_col, targets in languages:
        
        for model_name, model_path in models.items():
            process_model(lang_name, lang_abbr, gt_col, targets, model_name, 
                         model_path, chrf_scorer, chrf_plus_scorer, output_dir)


if __name__ == "__main__":
    main()