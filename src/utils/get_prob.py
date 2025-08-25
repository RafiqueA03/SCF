import numpy as np
import pandas as pd

def condProb(data):
    """
    Computes conditional probabilities as described in:
    Chuang, Stone, Hanrahan (2008) A Probabilistic Model of the Categorical Association Between Colors
    """
    # Sort by color_name for indexing
    data = data.sort_values('color_name', ascending=True)

    # Find unique colour names and colorIDs
    colour_names = np.unique(data['color_name'])
    colour_IDs = np.unique(data['color_id'])

    # Crosstable between colour names and colour samples
    Count_WC = pd.crosstab(data['color_name'], data['color_id']).values

    # Sum over row & column
    Count_W = Count_WC.sum(axis=1)  # sum over columns (words)
    Count_C = Count_WC.sum(axis=0)  # sum over rows (colors)

    # Prepare M
    counter_names = np.arange(len(colour_names))
    counter_IDs = np.arange(len(colour_IDs))
    zero_names = np.zeros_like(counter_names)
    zero_IDs = np.zeros_like(counter_IDs)
    ones_names = 1 + zero_names
    ones_IDs = 1 + zero_IDs

    # p(w|c)=p(w,c)/p(c)
    PWc = Count_WC / (ones_names[:, None] * Count_C)

    # p(c|w)=p(w,c)/p(w)
    PCw = Count_WC.T / (ones_IDs[:, None] * Count_W)

    # check input tables
    if np.any(np.abs(PCw.sum(axis=0) - 1) > 1e-5):
        raise ValueError('Probability function is not normalized to 1')

    if np.any(np.abs(PWc.sum(axis=0) - 1) > 1e-5):
        raise ValueError('Probability distribution is not normalized to 1')

    return PWc, PCw, Count_W, colour_names, colour_IDs