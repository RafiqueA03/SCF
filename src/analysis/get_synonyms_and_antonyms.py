import pandas as pd
import numpy as np
from scipy.spatial.distance import jensenshannon
import os
import colour

def find_color_synonyms_antonyms_cam16(probabilities_df, extratrees_df, top_k=5, delta_e_threshold=1.0):
    """
    Find synonyms and antonyms for color words.
    Synonyms: highest similarity (1-JSD)
    Antonyms: minimum JSD + maximum CAM16-UCS ΔE approach
    
    Parameters:
    - probabilities_df: DataFrame with color names as columns and probability distributions
    - extratrees_df: DataFrame with columns ['J_lightness', 'a_prime', 'b_prime', 'red', 'green', 'blue', 'color_name']
    - top_k: Number of top matches to find (default: 5)
    - delta_e_threshold: Minimum CAM16-UCS ΔE threshold for antonyms (default: 1.0)
    
    Returns:
    - tuple: (synonym_results, antonym_results) as DataFrames
    """
    
    # Create color name to centroid mapping from extratrees data
    color_centroids = {}
    
    # Group by color_name and average the J,a,b coordinates
    for color_name in extratrees_df['color_name'].unique():
        color_data = extratrees_df[extratrees_df['color_name'] == color_name]
        avg_j = color_data['J_lightness'].mean()
        avg_a = color_data['a_prime'].mean() 
        avg_b = color_data['b_prime'].mean()
        color_centroids[color_name.replace('_', ' ')] = (avg_j, avg_a, avg_b)
    
    all_words = probabilities_df.columns.values
    synonym_results = []
    antonym_results = []
    
    for i, target_word in enumerate(all_words):
        
        target_prob = probabilities_df[target_word].values
        target_clean = target_word.replace('_', ' ')
        
        similarities = []
        
        for compare_word in all_words:
            if compare_word == target_word:
                continue
                
            compare_prob = probabilities_df[compare_word].values
            js_div = jensenshannon(target_prob, compare_prob)
            similarity = 1 - js_div
            
            similarities.append({
                'match': compare_word.replace('_', ' '),
                'similarity': similarity,
                'jsd': js_div
            })
        
        # SYNONYMS: Sort by highest similarity (1-JSD)
        similarities_sorted = sorted(similarities, key=lambda x: x['similarity'], reverse=True)
        
        for rank, match_data in enumerate(similarities_sorted[:top_k], 1):
            synonym_results.append({
                'word': target_clean,
                'rank': rank,
                'synonym': match_data['match'],
                'similarity': match_data['similarity']
            })
    
    # Create pairwise similarity matrix
    n_words = len(probabilities_df.columns)
    similarity_matrix = np.zeros((n_words, n_words))
    
    for i, word1 in enumerate(probabilities_df.columns):
        for j, word2 in enumerate(probabilities_df.columns):
            if i == j:
                similarity_matrix[i, j] = 1.0  # Self-similarity
            else:
                prob1 = probabilities_df[word1].values
                prob2 = probabilities_df[word2].values
                js_div = jensenshannon(prob1, prob2)
                similarity_matrix[i, j] = 1 - js_div
    # Create CAM16-UCS ΔE matrix
    words_with_centroids = [word.replace('_', ' ') for word in probabilities_df.columns if word.replace('_', ' ') in color_centroids]
    n_words_with_centroids = len(words_with_centroids)
    cam16_matrix = np.zeros((n_words_with_centroids, n_words_with_centroids))
    
    for i, word1 in enumerate(words_with_centroids):
        for j, word2 in enumerate(words_with_centroids):
            if i == j:
                cam16_matrix[i, j] = 0.0  # Self-distance is 0
            else:
                centroid1 = color_centroids[word1]
                centroid2 = color_centroids[word2]
                jab1 = np.array([centroid1[0], centroid1[1], centroid1[2]])
                jab2 = np.array([centroid2[0], centroid2[1], centroid2[2]])
                delta_e = colour.difference.delta_E(jab1, jab2, method='CAM16-UCS')
                cam16_matrix[i, j] = delta_e
    # Create DataFrames for matrices
    similarity_df = pd.DataFrame(similarity_matrix,
                               index=probabilities_df.columns.str.replace('_', ' '),
                               columns=probabilities_df.columns.str.replace('_', ' '))
    
    cam16_df = pd.DataFrame(cam16_matrix,
                           index=words_with_centroids,
                           columns=words_with_centroids)
    
    # ANTONYMS: Use only CAM16-UCS matrix for selection
    for i, target_word in enumerate(all_words):
        target_clean = target_word.replace('_', ' ')
        if target_clean in words_with_centroids:
            
            # Get all CAM16-UCS ΔE distances from matrix
            antonym_candidates = []
            
            for compare_clean in words_with_centroids:
                if compare_clean == target_clean:
                    continue
                    
                # Get ΔE from the full matrix
                cam16_delta_e = cam16_df.loc[target_clean, compare_clean]
                
                antonym_candidates.append({
                    'word': compare_clean,
                    'cam16_delta_e': cam16_delta_e
                })
            
            # Sort by maximum CAM16-UCS ΔE (most perceptually distant)
            antonym_candidates.sort(key=lambda x: x['cam16_delta_e'], reverse=True)
            
            # Take top_k and apply threshold
            for rank, candidate in enumerate(antonym_candidates[:top_k], 1):
                if candidate['cam16_delta_e'] >= delta_e_threshold:
                    antonym_results.append({
                        'word': target_clean,
                        'rank': rank,
                        'antonym': candidate['word'],
                        'cam16_delta_e': candidate['cam16_delta_e']
                    })
    
    return pd.DataFrame(synonym_results), pd.DataFrame(antonym_results), similarity_df, cam16_df

def process_language_synonyms_antonyms(language_name, top_k=5, delta_e_threshold=1.0):
    """
    Process synonyms and antonyms for one language.
    """
    
    try:
        # Load probability data
        language_file = language_name.replace(' ', '_')
        prob_file = f"PCw_results/PCw_{language_file}.csv"
        probabilities_df = pd.read_csv(prob_file, encoding='utf-8-sig', index_col=0)
        
        extratrees_file = f"data/{language_file}_processed.csv"
        extratrees_df = pd.read_csv(extratrees_file, encoding='utf-8-sig')
        
        # Find synonyms and antonyms
        synonym_table, antonym_table, similarity_matrix, cam16_matrix = find_color_synonyms_antonyms_cam16(
            probabilities_df, 
            extratrees_df, 
            top_k=top_k, 
            delta_e_threshold=delta_e_threshold
        )
        
        # Save matrices
        similarity_file = f"synonyms_antonyms/{language_file}_similarity_matrix.csv"
        cam16_file = f"synonyms_antonyms/{language_file}_cam16_delta_e_matrix.csv"
        
        similarity_matrix.to_csv(similarity_file, encoding='utf-8-sig')
        cam16_matrix.to_csv(cam16_file, encoding='utf-8-sig')
        
        return synonym_table, antonym_table
        
    except Exception as e:
        print(f"  Error: {e}")
        return None, None

def main():
    """
    Process all languages for synonyms and CAM16-UCS based antonym detection.
    """
    
    languages = ['American English', 'British English', 'French', 'Greek', 'Himba']
    top_k = 5
    delta_e_threshold = 1.0  # JND threshold for CAM16-UCS
    
    all_synonym_results = []
    all_antonym_results = []
    
    for language in languages:
        print(f"    -{language}")
        synonym_result, antonym_result = process_language_synonyms_antonyms(language, top_k, delta_e_threshold)
        if synonym_result is not None:
            synonym_result['language'] = language
            all_synonym_results.append(synonym_result)
        if antonym_result is not None and len(antonym_result) > 0:
            antonym_result['language'] = language
            all_antonym_results.append(antonym_result)
    
    # Save combined results
    if all_synonym_results:
        master_synonym_table = pd.concat(all_synonym_results, ignore_index=True)
        master_synonym_file = 'synonyms_antonyms/all_languages_synonyms.csv'
        master_synonym_table.to_csv(master_synonym_file, index=False, encoding='utf-8-sig')
    
    if all_antonym_results:
        master_antonym_table = pd.concat(all_antonym_results, ignore_index=True)
        master_antonym_file = 'synonyms_antonyms/all_languages_antonyms.csv'
        master_antonym_table.to_csv(master_antonym_file, index=False, encoding='utf-8-sig')

if __name__ == "__main__":
    print("Synonyms and Antonyms Analysis completed for:")
    main()
