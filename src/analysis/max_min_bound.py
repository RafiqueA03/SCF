import pandas as pd
import numpy as np
import src.utils.bootstrapRep as bs
import src.utils.get_prob as gd
import argparse


def bhattacharyya_coefficient(p1, p2):
    """Calculate Bhattacharyya coefficient."""
    return np.sum(np.sqrt(np.multiply(p1, p2)))

def bhattacharyya_vector(p1, p2):
    """Calculates the Bhattacharyya coefficient for vectors"""
    return np.sum(np.sqrt(p1 * p2))


def run_max_min_bound(df, lang=None):
    PWc, _, _, _, _ = gd.condProb(df)

    # Call bootstrapRep 100 times
    num_bootstrap_replicates = 100
    bootstrap_PWc = [None] * num_bootstrap_replicates

    for rep in range(num_bootstrap_replicates):
        bootstrap_PWc[rep] = bs.bootstrapRep(df)
        if (rep + 1) % 20 == 0:
            pass

    # Calculate Bhattacharyya coefficient between original PWc and each bootstrap replicate
    bhattacharyya_bootstrap = np.zeros(num_bootstrap_replicates)

    for i in range(num_bootstrap_replicates):
        bootstrap_pwc_i = bootstrap_PWc[i]
        
        bhat_coeffs = []
        for col in range(PWc.shape[1]):  # For each color column
            coeff = bhattacharyya_coefficient(PWc[:, col], bootstrap_pwc_i[:, col])
            bhat_coeffs.append(coeff)
        
        bhattacharyya_bootstrap[i] = np.mean(bhat_coeffs)

    # Extract individual color columns from the full PWc matrix
    num_colors = PWc.shape[1]  # Number of colors (columns)

    # Calculate average color distribution (mean across all colors)
    average_color_distribution = np.mean(PWc, axis=1) 

    # Calculate Bhattacharyya coefficient for each color vs average
    bhattacharyya_individual = np.zeros(num_colors)

    for i in range(num_colors):
        bhat_coeff = bhattacharyya_coefficient(average_color_distribution, PWc[:, i])
        bhattacharyya_individual[i] = bhat_coeff

    return {
        'Language': lang,
        'Mean_Bhattacharyya_Bootstrap': np.mean(bhattacharyya_bootstrap),
        'Std_Bhattacharyya_Bootstrap': np.std(bhattacharyya_bootstrap, ddof=1),
        'Min_Bhattacharyya_Bootstrap': np.min(bhattacharyya_bootstrap),
        'Max_Bhattacharyya_Bootstrap': np.max(bhattacharyya_bootstrap),
        'Mean_Bhattacharyya_Average': np.mean(bhattacharyya_individual),
        'Std_Bhattacharyya_Average': np.std(bhattacharyya_individual, ddof=1),
        'Min_Bhattacharyya_Average': np.min(bhattacharyya_individual),
        'Max_Bhattacharyya_Average': np.max(bhattacharyya_individual)
    }

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run color naming algorithm using ExtraTrees for different languages.")
    parser.add_argument('--language', type=str, 
                       choices=[ 'American English', 'British English', 'Himba', 'French', 'Greek'],
                       nargs='*',
                       help="Language(s) to run the algorithm for.")
    
    args = parser.parse_args()
    
    # Set default languages if none are provided
    if not args.language:
        args.language = ["American English", "British English", "French", "Greek", "Himba"]
    
    results = []
    # Run the algorithm for each specified language
    for lang in args.language:
            df = pd.read_csv(f'data/{lang.replace(" ", "_")}_processed.csv')
            result = run_max_min_bound(df, lang=lang)
            results.append(result)
    print(f"Minimum and maximum bounds:")
    for res in results:
     print(f"   - {res['Language']}: min: {round(res['Mean_Bhattacharyya_Average'],2):.2f}, max: {round(res['Mean_Bhattacharyya_Bootstrap'],2):.2f}")