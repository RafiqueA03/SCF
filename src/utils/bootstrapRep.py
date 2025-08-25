import pandas as pd
import numpy as np
from scipy.stats import multinomial
import src.utils.get_prob as gd

def bootstrapRep(data):
    """
    Creates conditional probabilities PWc of bootstrapped responses with replacement
    
    Args:
        data: DataFrame with columns 'color_name' and 'color_id'
    
    Returns:
        bPWc: Bootstrapped conditional probability matrix P(word|color)
    """
    
    # Get the original conditional probabilities using the same function as main script
    PWc, _, _, _, _ = gd.condProb(data)
    
    # Get unique color IDs to know how many colors we have
    colour_IDs = data['color_id'].unique()
    num_colors = len(colour_IDs)
    
    # Create crosstab to get original counts
    Count_WC = pd.crosstab(data['color_name'], data['color_id'], dropna=False)
    Count_WC = Count_WC.values  # Convert to numpy array
    
    # Create a bootstrapped dataset: for each color, resample the responses
    bCount_WC = np.zeros_like(Count_WC)
    
    for i in range(num_colors):
        # Get the total count for this color
        n_responses = int(np.sum(Count_WC[:, i]))
        if n_responses > 0:
            # Use multinomial to bootstrap sample using original PWc probabilities
            bCount_WC[:, i] = multinomial.rvs(n_responses, PWc[:, i])
    
    # Now convert bootstrapped counts back to conditional probabilities
    # Sum counts for normalization
    bCount_C = np.sum(bCount_WC, axis=0)
    
    # Calculate p(w|c) = count(w,c) / count(c)
    bPWc = np.zeros_like(bCount_WC, dtype=float)
    for i in range(num_colors):
        if bCount_C[i] > 0:
            bPWc[:, i] = bCount_WC[:, i] / bCount_C[i]
    
    return bPWc