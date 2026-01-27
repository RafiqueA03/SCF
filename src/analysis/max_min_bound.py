import pandas as pd
import numpy as np
import src.utils.bootstrapRep as bs
import src.utils.get_prob as gd
import argparse
import os


def bhattacharyya_coefficient(p1, p2):
    """Calculate Bhattacharyya coefficient."""
    return np.sum(np.sqrt(np.multiply(p1, p2)))


def run_max_min_bound(df, lang=None):
    """
    Calculate performance bounds using Bhattacharyya coefficients.
    
    Args:
        df: Input dataframe
        lang: Language name for identification
        
    Returns:
        Dictionary containing statistical measures
    """
    PWc, _, _, _, _ = gd.condProb(df)

    num_bootstrap_replicates = 100
    bootstrap_PWc = [None] * num_bootstrap_replicates

    for rep in range(num_bootstrap_replicates):
        bootstrap_PWc[rep] = bs.bootstrapRep(df)

    bhattacharyya_bootstrap = np.zeros(num_bootstrap_replicates)

    for i in range(num_bootstrap_replicates):
        bootstrap_pwc_i = bootstrap_PWc[i]
        
        bhat_coeffs = []
        for col in range(PWc.shape[1]):
            coeff = bhattacharyya_coefficient(PWc[:, col], bootstrap_pwc_i[:, col])
            bhat_coeffs.append(coeff)
        
        bhattacharyya_bootstrap[i] = np.mean(bhat_coeffs)

    num_colors = PWc.shape[1]
    average_color_distribution = np.mean(PWc, axis=1) 
    bhattacharyya_individual = np.zeros(num_colors)

    for i in range(num_colors):
        bhat_coeff = bhattacharyya_coefficient(average_color_distribution, PWc[:, i])
        bhattacharyya_individual[i] = bhat_coeff

    return {
        'Language': lang,
        'Mean_Bhattacharyya_Bootstrap': np.mean(bhattacharyya_bootstrap),
        'Mean_Bhattacharyya_Average': np.mean(bhattacharyya_individual),
    }


def save_results_to_file(results):
    """
    Save analysis results to a text file in the results directory.
    
    Args:
        results: List of result dictionaries
    """
    os.makedirs('results', exist_ok=True)
    
    with open('results/performance_bounds.txt', 'w') as f:
        f.write("Performance Bounds Analysis Results\n")
        f.write("=" * 40 + "\n\n")
        
        f.write("Minimum and maximum bounds:\n")
        for res in results:
            min_val = round(res['Mean_Bhattacharyya_Average'], 2)
            max_val = round(res['Mean_Bhattacharyya_Bootstrap'], 2)
            f.write(f"   - {res['Language']}: min: {min_val:.2f}, max: {max_val:.2f}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run color naming algorithm using ExtraTrees for different languages.")
    parser.add_argument('--language', type=str, 
                       choices=['American English', 'British English', 'Himba', 'French', 'Greek'],
                       nargs='*',
                       help="Language(s) to run the algorithm for.")
    
    args = parser.parse_args()
    
    if not args.language:
        args.language = ["American English", "British English", "French", "Greek", "Himba"]
    
    results = []
    
    for lang in args.language:
        df = pd.read_csv(f'data/{lang.replace(" ", "_")}_processed.csv')
        result = run_max_min_bound(df, lang=lang)
        results.append(result)
    
    save_results_to_file(results)