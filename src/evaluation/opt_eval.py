import pandas as pd
import os
from spincam.language_model_evaluator import LanguageModelEvaluator 
import warnings
warnings.filterwarnings('ignore')
import argparse
import logging


def run_language_specific_comparison(data_path, languages, train_data_path=None, test_grid_csv=None, random_seed=42, n_trials=10):
    """Run model comparison for each specified language."""
    logging.info("Languages to analyze: %s", languages)
    
    all_results = []
    summary_data = []
    
    for language in languages:
        logging.info(f"Processing language: {language}")
        try:
            # Initialize evaluator
            evaluator = LanguageModelEvaluator(test_grid_csv=test_grid_csv)
            
            # Load and process language-specific data
            lang_info = evaluator.load_and_process_language_data(data_path, language, train_data_path)
            
            # Prepare regression data
            X, y = evaluator.prepare_regression_data()
            
            # Compare models for this language
            lang_results = evaluator.compare_models_for_language(
                X, y, language, n_trials=n_trials, random_seed=random_seed
            )
            
            # Add to overall results
            for model_name, result in lang_results.items():
                all_results.append(result)
            
            # Create summary for this language
            best_model = max(lang_results.items(), key=lambda x: x[1]['combined_score'])
            summary_data.append({
                'Language': language,
                'Samples': lang_info['samples'],
                'Unique_Color_Names': lang_info['unique_color_names'],
                'Best_Model': best_model[0],
                'Best_Peak_Accuracy': best_model[1]['cv_peak_accuracy'],
                'Best_Bhattacharyya': best_model[1]['cv_bhattacharyya'],
                'Best_Combined_Score': best_model[1]['combined_score'],
                'Best_Valid_Names': best_model[1]['valid_names_count'],
                'Best_Coverage_Pct': best_model[1]['coverage_percentage'],
                'Best_Rotation_Fraction': best_model[1]['rotation_fraction']
            })
            
        except Exception as e:
            logging.error(f"Error processing {language}: {e}")
            continue
    
    # Create comprehensive results DataFrame
    if all_results:
        results_df = pd.DataFrame(all_results)
        summary_df = pd.DataFrame(summary_data)
        
        # Save results
        if not os.path.exists('results'):
            os.makedirs('results')
        
        # Detailed results
        detailed_file = f'results/language_model_comparison_detailed.csv'
        results_df.to_csv(detailed_file, index=False)
        
        # Summary results
        summary_file = f'results/language_model_comparison_summary.csv'
        summary_df.to_csv(summary_file, index=False)
        
        # Print overall summary
        logging.info("Language Model Comparison Summary:")
        logging.info(f"{'Language':<18} {'Unique Colors':<13} {'Best Model':<18} {'Valid Names':<11} {'Peak Acc':<10} {'Peak Bhatt':<10} {'Combined Score':<15}")
        logging.info("-" * 105)
        
        for _, row in summary_df.iterrows():
            logging.info(f"{row['Language']:<18} {row['Unique_Color_Names']:<13} "
                  f"{row['Best_Model']:<18} {row['Best_Valid_Names']:<11} {row['Best_Peak_Accuracy']:<10.4f} {row['Best_Bhattacharyya']:<10.4f} {row['Best_Combined_Score']:<15.4f}")
        
        return results_df, summary_df
    
    else:
        logging.error("No results generated - check your data and paths")
        return None, None

# Usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    RANDOM_SEED = 42
    ALL_LANGUAGES = ['American English', 'British English', 'Greek', 'French', 'Himba']
    
    parser = argparse.ArgumentParser(description="Run language-specific model comparison.")
    parser.add_argument('--random_seed', type=int, default=RANDOM_SEED, help="Random seed for reproducibility")
    parser.add_argument('--languages', nargs='+', default=ALL_LANGUAGES, help="List of languages to analyze")
    args = parser.parse_args()

    data_path = "data/processed_combined_data.csv"
    train_data_path = "data/train_data_with_cam16_ucs.csv"
    test_grid_csv = "data/test_data_cam16_ucs.csv"
        
    # Run comparison
    detailed_results, summary_results = run_language_specific_comparison(
        data_path=data_path, languages=args.languages, train_data_path=train_data_path,
        test_grid_csv=test_grid_csv, random_seed=args.random_seed, n_trials=10)