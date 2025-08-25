import pandas as pd
import numpy as np
from scipy.spatial.distance import jensenshannon
import os
import argparse
import math

def load_PCw_from_file(language):
    """
    Load pre-computed PCw data from CSV file.
    
    Parameters:
    -----------
    language : str
        Language name (e.g., 'Himba', 'British English', etc.)
    
    Returns:
    --------
    tuple: (PCw, Count_W, colour_names)
        PCw: P(color|word) matrix as numpy array
        Count_W: word counts (approximated from PCw)
        colour_names: color names as numpy array
    """
    
    # Construct filename
    language_file = language.replace(' ', '_')
    filename = f'PCw_results/PCw_{language_file}.csv'
    
    # Load the DataFrame
    PCw_df = pd.read_csv(filename, index_col=0, encoding='utf-8-sig')
    
    # Extract components
    PCw = PCw_df.values  # numpy array
    colour_names = PCw_df.columns.values  # color names
    
    # Approximate word counts from PCw
    Count_W = np.sum(PCw, axis=0)
    
    return PCw, Count_W, colour_names

def get_PCw_fast(language):
    """
    Wrapper function that loads PCw data from file.
    """
    try:
        return load_PCw_from_file(language)
    except FileNotFoundError:
        print(f"Pre-computed file not found for {language}. You need to compute and save it first.")
        raise

def compute_symmetric_color_translation(PCw_source, PCw_target, color_names_source, color_names_target, Count_W_source=None, Count_W_target=None):
    """
    Computes color translations using symmetric similarity matrix approach.
    Since JSD is symmetric, we compute one matrix and extract both directions.
    """
    
    # Initialize Jensen-Shannon divergence matrix
    js_divergence_matrix = np.zeros((len(color_names_source), len(color_names_target)))
    
    # Compute Jensen-Shannon divergence for each pair
    for i_source in range(len(color_names_source)):
        for i_target in range(len(color_names_target)):
            # Get probability distributions
            prob_source = PCw_source[:, i_source]  # P(color | word_source)
            prob_target = PCw_target[:, i_target]  # P(color | word_target)
            
            # Compute Jensen-Shannon divergence
            js_div = jensenshannon(prob_source, prob_target)
            js_divergence_matrix[i_source, i_target] = js_div
    
    # Convert divergence to similarity (1 - JS divergence)
    similarity_matrix = 1 - js_divergence_matrix
    
    # Source -> Target: maximum along each row
    best_source2target = np.argmax(similarity_matrix, axis=1)
    # Target -> Source: maximum along each column  
    best_target2source = np.argmax(similarity_matrix, axis=0)
    
    # Count words (assuming underscore separation)
    def count_words(names):
        return [len(name.split('_')) for name in names]
    
    countWords_source = count_words(color_names_source)
    countWords_target = count_words(color_names_target)
    
    # Calculate individual info loss
    def calculate_info_loss_bits(similarity):
        divergence = 1 - similarity if similarity <= 1 else 0
        if divergence < 0.999:  # Avoid log(0)
            return -math.log2(1 - divergence)
        else:
            return float('inf')  # or some large number for very poor translations
    
    # Create translation table: source -> target
    source_similarities = [similarity_matrix[i, best_source2target[i]] for i in range(len(best_source2target))]
    source_js_divergences = [js_divergence_matrix[i, best_source2target[i]] for i in range(len(best_source2target))]
    source_info_losses = [calculate_info_loss_bits(sim) for sim in source_similarities]
    
    translation_table_source2target = pd.DataFrame({
        'source_name': color_names_source,
        'source_word_count': countWords_source,
        'target_index': best_source2target,
        'target_name': [color_names_target[idx] for idx in best_source2target],
        'target_word_count': [countWords_target[idx] for idx in best_source2target],
        'similarity': source_similarities,
        'info_loss_bits': source_info_losses,
        'js_divergence': source_js_divergences
    })
    
    # Create translation table: target -> source
    target_similarities = [similarity_matrix[best_target2source[i], i] for i in range(len(best_target2source))]
    target_js_divergences = [js_divergence_matrix[best_target2source[i], i] for i in range(len(best_target2source))]
    target_info_losses = [calculate_info_loss_bits(sim) for sim in target_similarities]
    
    translation_table_target2source = pd.DataFrame({
        'source_name': color_names_target,
        'source_word_count': countWords_target,
        'target_index': best_target2source,
        'target_name': [color_names_source[idx] for idx in best_target2source],
        'target_word_count': [countWords_source[idx] for idx in best_target2source],
        'similarity': target_similarities,
        'info_loss_bits': target_info_losses,
        'js_divergence': target_js_divergences
    })
    
    # Add count data if provided
    if Count_W_source is not None:
        translation_table_source2target['source_count'] = Count_W_source
    if Count_W_target is not None:
        translation_table_source2target['target_count'] = [Count_W_target[idx] for idx in best_source2target]
        translation_table_target2source['source_count'] = Count_W_target
    if Count_W_source is not None:
        translation_table_target2source['target_count'] = [Count_W_source[idx] for idx in best_target2source]
    
    return {
        'similarity_matrix': similarity_matrix,
        'js_divergence_matrix': js_divergence_matrix,
        'translation_source2target': best_source2target,
        'translation_target2source': best_target2source,
        'translation_table_source2target': translation_table_source2target,
        'translation_table_target2source': translation_table_target2source
    }

def check_translation_symmetry(translation_results, color_names_source, color_names_target):
    """
    Check for asymmetric translations by comparing row maxima with column maxima.
    """
    
    best_source2target = translation_results['translation_source2target']
    best_target2source = translation_results['translation_target2source']
    similarity_matrix = translation_results['similarity_matrix']
    
    # Find symmetric and asymmetric pairs
    symmetric_pairs = []
    source_asymmetric = []
    target_asymmetric = []
    
    # Check source -> target translations for symmetry
    for source_idx, target_idx in enumerate(best_source2target):
        # Check if the reverse translation matches
        if best_target2source[target_idx] == source_idx:
            # Symmetric pair
            symmetric_pairs.append({
                'source_name': color_names_source[source_idx],
                'target_name': color_names_target[target_idx],
                'similarity': similarity_matrix[source_idx, target_idx],
                'source_idx': source_idx,
                'target_idx': target_idx
            })
        else:
            # Asymmetric: source translates to target, but target translates elsewhere
            competing_source_idx = best_target2source[target_idx]
            source_asymmetric.append({
                'source_name': color_names_source[source_idx],
                'target_name': color_names_target[target_idx],
                'target_best_source': color_names_source[competing_source_idx],
                'source_to_target_sim': similarity_matrix[source_idx, target_idx],
                'target_to_competing_sim': similarity_matrix[competing_source_idx, target_idx],
                'source_idx': source_idx,
                'target_idx': target_idx,
                'competing_source_idx': competing_source_idx
            })
    
    # Check target -> source translations for remaining asymmetric cases
    for target_idx, source_idx in enumerate(best_target2source):
        # Only add if not already covered in symmetric pairs
        if best_source2target[source_idx] != target_idx:
            # Check if this asymmetric case wasn't already recorded
            already_recorded = any(asym['target_idx'] == target_idx for asym in source_asymmetric)
            if not already_recorded:
                competing_target_idx = best_source2target[source_idx]
                target_asymmetric.append({
                    'target_name': color_names_target[target_idx],
                    'source_name': color_names_source[source_idx],
                    'source_best_target': color_names_target[competing_target_idx],
                    'target_to_source_sim': similarity_matrix[source_idx, target_idx],
                    'source_to_competing_sim': similarity_matrix[source_idx, competing_target_idx],
                    'target_idx': target_idx,
                    'source_idx': source_idx,
                    'competing_target_idx': competing_target_idx
                })
    
    return (pd.DataFrame(symmetric_pairs), 
            pd.DataFrame(source_asymmetric), 
            pd.DataFrame(target_asymmetric))

def calculate_information_loss(translation_results, symmetric_pairs, source_asymmetric, target_asymmetric, total_source_terms, total_target_terms):
    """
    Calculate information loss in bits.
    Only uses similarities from source->target direction to avoid double-counting.
    """
    
    all_similarities = []
    
    # Get similarities for all source terms from the translation table
    source_to_target_table = translation_results['translation_table_source2target']
    all_similarities = source_to_target_table['similarity'].tolist()
    
    if not all_similarities:
        return {
            'avg_info_loss_bits': 0, 
            'total_translation_loss_bits': 0, 
            'untranslated_loss_bits': 0,
            'total_system_loss_bits': 0,
            'info_efficiency': 0
        }
    
    # Convert similarities to information loss in bits
    info_losses = []
    for similarity in all_similarities:
        # Convert similarity to divergence: JSD = 1 - similarity
        divergence = 1 - similarity if similarity <= 1 else 0
        if divergence < 0.999:  # Avoid log(0)
            bits = -math.log2(1 - divergence)
            info_losses.append(bits)
    
    if info_losses:
        avg_info_loss = sum(info_losses) / len(info_losses)
        total_translation_loss = sum(info_losses)
        
        # Calculate system-wide information efficiency
        total_terms = total_source_terms + total_target_terms
        max_possible_info = math.log2(total_terms) if total_terms > 1 else 1
        translated_terms = len(symmetric_pairs) + (total_source_terms - len(symmetric_pairs))
        untranslated_terms = total_terms - translated_terms
        
        untranslated_loss = untranslated_terms * max_possible_info
        total_system_loss = total_translation_loss + untranslated_loss
        max_system_info = total_terms * max_possible_info
        
        info_efficiency = (1 - (total_system_loss / max_system_info)) * 100 if max_system_info > 0 else 0
        
        return {
            'avg_info_loss_bits': avg_info_loss,
            'total_translation_loss_bits': total_translation_loss,
            'untranslated_loss_bits': untranslated_loss,
            'total_system_loss_bits': total_system_loss,
            'info_efficiency': info_efficiency
        }
    
    return {
        'avg_info_loss_bits': 0, 
        'total_translation_loss_bits': 0, 
        'untranslated_loss_bits': 0,
        'total_system_loss_bits': 0,
        'info_efficiency': 0
    }

def analyze_directional_vocabulary(translation_results, color_names_source, color_names_target):
    """
    Analyze vocabulary usage in each translation direction.
    """
    
    # Source → Target: What target terms are used?
    source_to_target_table = translation_results['translation_table_source2target']
    target_terms_used = set(source_to_target_table['target_name'].tolist())
    
    # Target → Source: What source terms are used?
    target_to_source_table = translation_results['translation_table_target2source']
    source_terms_used = set(target_to_source_table['target_name'].tolist())
    
    # Check for apparent overlap
    apparent_overlap = target_terms_used & source_terms_used
    
    total_source_terms = len(color_names_source)
    total_target_terms = len(color_names_target)
    
    return {
        'source_to_target_vocab': sorted(target_terms_used),
        'target_to_source_vocab': sorted(source_terms_used),
        'source_to_target_count': len(target_terms_used),
        'target_to_source_count': len(source_terms_used),
        'apparent_overlap': sorted(apparent_overlap),
        'true_total_vocab': len(target_terms_used) + len(source_terms_used),
        'target_utilization': len(target_terms_used) / total_target_terms * 100 if total_target_terms > 0 else 0,
        'source_utilization': len(source_terms_used) / total_source_terms * 100 if total_source_terms > 0 else 0
    }

def generate_academic_report(source_language, target_language, translation_results, 
                           symmetric_pairs, source_asymmetric, target_asymmetric,
                           info_loss_analysis, vocabulary_analysis, 
                           color_names_source, color_names_target):
    """
    Generate clean summary report for academic use.
    """
    
    total_source_terms = len(color_names_source)
    total_target_terms = len(color_names_target)
    mutual_count = len(symmetric_pairs)
    mismatch_count = total_source_terms - mutual_count
    
    # Calculate round-trip survival rates
    source_survival_rate = mutual_count / total_source_terms * 100 if total_source_terms > 0 else 0
    target_survival_rate = mutual_count / total_target_terms * 100 if total_target_terms > 0 else 0
    
    # Calculate vocabulary compression
    original_vocab = total_source_terms + total_target_terms
    active_vocab = vocabulary_analysis['true_total_vocab']
    vocab_reduction = original_vocab - active_vocab
    reduction_rate = vocab_reduction / original_vocab * 100 if original_vocab > 0 else 0
    
    # Calculate translation consistency
    total_translations = mutual_count + mismatch_count
    consistency_rate = mutual_count / total_translations * 100 if total_translations > 0 else 0
    
    report = f"""Language Pair: {source_language} ↔ {target_language}
Matrix Size: {total_source_terms} × {total_target_terms}
Symmetric Translations: {mutual_count}
Translation Asymmetries: {mismatch_count}
Consistency Rate: {consistency_rate:.1f}%
{source_language} Round-trip Survivors: {mutual_count}/{total_source_terms} ({source_survival_rate:.1f}%)
{target_language} Round-trip Survivors: {mutual_count}/{total_target_terms} ({target_survival_rate:.1f}%)
{source_language}→{target_language} Vocabulary: {vocabulary_analysis['source_to_target_count']} {target_language.lower()} terms ({vocabulary_analysis['target_utilization']:.1f}%)
{target_language}→{source_language} Vocabulary: {vocabulary_analysis['target_to_source_count']} {source_language.lower()} terms ({vocabulary_analysis['source_utilization']:.1f}%)
Total Active Vocabulary: {active_vocab} terms
Vocabulary Compression: {reduction_rate:.1f}% loss
Average Information Loss: {info_loss_analysis['avg_info_loss_bits']:.3f} bits per translation
Information Efficiency: {info_loss_analysis['info_efficiency']:.1f}%
"""
    return report

def save_translation_results(source_language, target_language, translation_results, 
                           symmetric_pairs, source_asymmetric, target_asymmetric, 
                           academic_report, colour_names_source, colour_names_target):
    """
    Save all results to files in organized format.
    """
    
    # Create output directory
    os.makedirs('translation_symmetric', exist_ok=True)
    
    # FILE 1: Similarity matrix
    similarity_file = f'translation_symmetric/{source_language}_{target_language}_similarity_matrix.csv'
    similarity_df = pd.DataFrame(translation_results['similarity_matrix'], 
                                index=colour_names_source, 
                                columns=colour_names_target)
    similarity_df.to_csv(similarity_file, encoding='utf-8-sig')
    
    # FILE 2: Translation analysis
    combined_file = f'translation_symmetric/{source_language}_{target_language}_translation_analysis.csv'
    
    # Get the translation tables
    source_to_target = translation_results['translation_table_source2target']
    target_to_source = translation_results['translation_table_target2source']
    
    # Determine the maximum number of rows needed
    max_rows = max(len(source_to_target), len(target_to_source))
    
    # Prepare summary statistics
    unique_source_terms_translated = len(source_to_target)
    unique_target_terms_as_translations = len(source_to_target['target_name'].unique())
    unique_target_terms_translated = len(target_to_source)
    unique_source_terms_as_translations = len(target_to_source['target_name'].unique())
    symmetric_count = len(symmetric_pairs)
    
    # Calculate consistency rates
    source_lang_consistency = symmetric_count / unique_source_terms_translated if unique_source_terms_translated > 0 else 0
    target_lang_consistency = symmetric_count / unique_target_terms_translated if unique_target_terms_translated > 0 else 0
    
    # Create rows with both sections in parallel columns
    all_rows = []
    for i in range(max_rows):
        row = [''] * 28
        
        # Add summary statistics to first row
        if i == 0:
            row[20] = unique_source_terms_translated               
            row[21] = unique_target_terms_as_translations          
            row[22] = symmetric_count                              
            row[23] = f"{source_lang_consistency:.3f}"            
            row[24] = unique_target_terms_translated               
            row[25] = unique_source_terms_as_translations          
            row[26] = symmetric_count                              
            row[27] = f"{target_lang_consistency:.3f}"            
        
        # SECTION 1: Source language translations
        if i < len(source_to_target):
            source_row = source_to_target.iloc[i]
            source_name = source_row['source_name']
            target_name = source_row['target_name']
            similarity = source_row['similarity']
            info_loss_bits = source_row['info_loss_bits']
            
            # Check if symmetric
            is_symmetric = False
            competing_term = ''
            competing_similarity = ''
            
            if len(symmetric_pairs) > 0:
                symmetric_match = symmetric_pairs[
                    (symmetric_pairs['source_name'] == source_name) & 
                    (symmetric_pairs['target_name'] == target_name)
                ]
                if not symmetric_match.empty:
                    is_symmetric = True
            
            if not is_symmetric and len(source_asymmetric) > 0:
                asym_match = source_asymmetric[source_asymmetric['source_name'] == source_name]
                if not asym_match.empty:
                    competing_term = asym_match.iloc[0]['target_best_source']
                    competing_similarity = f"{asym_match.iloc[0]['target_to_competing_sim']:.3f}"
            
            row[0] = source_name
            row[1] = target_name
            row[2] = f"{similarity:.3f}"
            row[3] = f"{info_loss_bits:.3f}"
            row[4] = is_symmetric
            row[5] = competing_term
            row[6] = competing_similarity
            row[7] = f'{source_language}_to_{target_language}'
        
        # SECTION 2: Target language translations
        if i < len(target_to_source):
            target_row = target_to_source.iloc[i]
            source_name = target_row['source_name']
            target_name = target_row['target_name']
            similarity = target_row['similarity']
            info_loss_bits = target_row['info_loss_bits']
            
            # Check if symmetric
            is_symmetric = False
            competing_term = ''
            competing_similarity = ''
            
            if len(symmetric_pairs) > 0:
                symmetric_match = symmetric_pairs[
                    (symmetric_pairs['target_name'] == source_name) & 
                    (symmetric_pairs['source_name'] == target_name)
                ]
                if not symmetric_match.empty:
                    is_symmetric = True
            
            if not is_symmetric and len(target_asymmetric) > 0:
                asym_match = target_asymmetric[target_asymmetric['target_name'] == source_name]
                if not asym_match.empty:
                    competing_term = asym_match.iloc[0]['source_best_target']
                    competing_similarity = f"{asym_match.iloc[0]['source_to_competing_sim']:.3f}"
            
            row[10] = source_name
            row[11] = target_name
            row[12] = f"{similarity:.3f}"
            row[13] = f"{info_loss_bits:.3f}"
            row[14] = is_symmetric
            row[15] = competing_term
            row[16] = competing_similarity
            row[17] = f'{target_language}_to_{source_language}'
        
        all_rows.append(row)
    
    # Create DataFrame with proper headers
    headers = [
        'term', 'translation', 'similarity', 'info_loss_bits', 'is_symmetric', 'competing_term', 'competing_similarity', 'direction',
        '', '',
        'term', 'translation', 'similarity', 'info_loss_bits', 'is_symmetric', 'competing_term', 'competing_similarity', 'direction',
        '', '',
        f'{source_language}_terms', f'{target_language}_translated_terms', 'roundtrip_survivors', f'{source_language}_consistency',
        f'{target_language}_terms', f'{source_language}_translated_terms', 'roundtrip_survivors', f'{target_language}_consistency'
    ]
    
    result_df = pd.DataFrame(all_rows, columns=headers)
    result_df.to_csv(combined_file, index=False, encoding='utf-8-sig')
    
    # FILE 3: Academic report
    report_file = f'translation_symmetric/{source_language}_{target_language}_report.txt'
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(academic_report)
    
    print(f"    - {source_language} & {target_language}")

if __name__ == "__main__":
    
    # Argument parsing
    parser = argparse.ArgumentParser(description='Symmetric color translation analysis with information loss metrics.')
    
    parser.add_argument('--source', type=str, 
                        choices=['British English', 'American English', 'Greek', 'French', 'Himba'], 
                        help='Source language')
    parser.add_argument('--target', type=str, 
                        choices=['British English', 'American English', 'Greek', 'French', 'Himba'], 
                        help='Target language')
    parser.add_argument('--all-pairs', action='store_true', 
                        help='Run all language pair combinations')
    
    args = parser.parse_args()
    
    def process_language_pair(source_lang, target_lang):
        """Process a single language pair for translation analysis."""
        
        # Load pre-computed PCw data
        PCw_source, Count_W_source, colour_names_source = get_PCw_fast(source_lang)
        PCw_target, Count_W_target, colour_names_target = get_PCw_fast(target_lang)
        
        # Compute symmetric translation
        translation_results = compute_symmetric_color_translation(
            PCw_source, PCw_target, colour_names_source, colour_names_target, 
            Count_W_source, Count_W_target
        )
        
        # Check translation symmetry
        symmetric_pairs, source_asymmetric, target_asymmetric = check_translation_symmetry(
            translation_results, colour_names_source, colour_names_target
        )
        
        # Calculate information loss
        info_loss_analysis = calculate_information_loss(
            translation_results, symmetric_pairs, source_asymmetric, target_asymmetric,
            len(colour_names_source), len(colour_names_target)
        )
        
        # Analyze directional vocabulary
        vocabulary_analysis = analyze_directional_vocabulary(
            translation_results, colour_names_source, colour_names_target
        )
        
        # Generate academic report
        academic_report = generate_academic_report(
            source_lang, target_lang, translation_results,
            symmetric_pairs, source_asymmetric, target_asymmetric,
            info_loss_analysis, vocabulary_analysis,
            colour_names_source, colour_names_target
        )
        
        # Save all results
        save_translation_results(
            source_lang, target_lang, translation_results,
            symmetric_pairs, source_asymmetric, target_asymmetric,
            academic_report, colour_names_source, colour_names_target
        )
        
        return True
    
    if args.all_pairs:
        # Run all combinations
        pairs = [
            ('American English', 'British English'),
            ('American English', 'French'), 
            ('American English', 'Greek'),
            ('American English', 'Himba'),
            ('British English', 'French'),
            ('British English', 'Greek'), 
            ('British English', 'Himba'),
            ('French', 'Greek'),
            ('French', 'Himba'),
            ('Greek', 'Himba')
        ]
        successful = 0
        failed = 0
        print("Translation completed for: ")
        for i, (source_lang, target_lang) in enumerate(pairs, 1):
            if process_language_pair(source_lang, target_lang):
                successful += 1
            else:
                failed += 1
        
    else:
        # Single pair mode
        if not args.source or not args.target:
            parser.error("--source and --target are required when not using --all-pairs")
        
        source_language = args.source
        target_language = args.target
        
        # Process single pair
        process_language_pair(source_language, target_language)