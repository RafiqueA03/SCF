"""
Get focal colors for each language and add original script columns to translation files
"""

import pandas as pd
import ast
from pathlib import Path


# Global constants
LANGUAGES = {
    'American English': 'AE',
    'British English': 'BE', 
    'French': 'FR',
    'Greek': 'GR',
    'Himba': 'HB'
}

def load_script_mappings(data_dir='data'):
    """Load romanized to original script mappings"""
    mappings = {'French': {}, 'Greek': {}}
    
    script_files = {
        'French': 'french_script_colors.txt',
        'Greek': 'greek_script_colors.txt'
    }
    
    for lang, filename in script_files.items():
        filepath = Path(data_dir) / filename
        with open(filepath, 'r', encoding='utf-8-sig') as f:
            for line in f:
                if ':' in line:
                    romanized, original = line.strip().split(':', 1)
                    mappings[lang][romanized.strip().lower()] = original.strip()
    
    return mappings

def add_original_scripts(data_dir='data', results_dir='results'):
    """Add original script columns to all translation files"""
    
    script_mappings = load_script_mappings(data_dir)
    trans_dir = Path(results_dir) / 'translations/ourmethod'
    
    for lang, code in LANGUAGES.items():
        filename = f"{lang.replace(' ', '_')}_translations.csv"
        filepath = trans_dir / filename
            
        df = pd.read_csv(filepath, encoding='utf-8-sig')
        
        # Add original script columns
        if 'FR' in df.columns:
            df['French_Original'] = df['FR'].str.lower().map(script_mappings['French'])
        if 'GR' in df.columns:
            df['Greek_Original'] = df['GR'].str.lower().map(script_mappings['Greek'])
        
        # Reorder columns
        cols = reorder_columns(df.columns.tolist(), code)
        df = df[cols]
        
        df.to_csv(filepath, index=False, encoding='utf-8-sig')

def reorder_columns(columns, primary_code):
    """Reorder columns with primary language first"""
    cols = []
    
    # Primary language first
    if primary_code in columns:
        cols.append(primary_code)
        if primary_code == 'FR' and 'French_Original' in columns:
            cols.append('French_Original')
        elif primary_code == 'GR' and 'Greek_Original' in columns:
            cols.append('Greek_Original')
    
    # Other languages
    for code in LANGUAGES.values():
        if code != primary_code and code in columns:
            cols.append(code)
            if code == 'FR' and 'French_Original' in columns:
                cols.append('French_Original')
            elif code == 'GR' and 'Greek_Original' in columns:
                cols.append('Greek_Original')
            
            sim_col = f"{code}_similarity"
            if sim_col in columns:
                cols.append(sim_col)
    
    # Add remaining columns
    remaining = [col for col in columns if col not in cols]
    cols.extend(remaining)
    
    return cols

def find_focal_color(pcw_df, color_name, color_id_rgb):
    """Find focal color for given color name"""
    max_pcw = pcw_df[color_name].max()
    tied_indices = pcw_df[pcw_df[color_name] == max_pcw].index
    tied_color_ids = pcw_df.loc[tied_indices, 'color_id'].tolist()
    
    if len(tied_color_ids) == 1:
        color_id = tied_color_ids[0]
        rgb_row = color_id_rgb[color_id_rgb['color_id'] == color_id]
        r, g, b = rgb_row.iloc[0][['R', 'G', 'B']].values
    else:
        # Average RGB for ties
        tied_rgb = color_id_rgb[color_id_rgb['color_id'].isin(tied_color_ids)]
        r, g, b = tied_rgb[['R', 'G', 'B']].mean().round().astype(int)
        color_id = f"avg_of_{len(tied_color_ids)}_colors"
    
    return {'color_id': color_id, 'max_pcw': max_pcw, 'R': r, 'G': g, 'B': b}

def calculate_focal_colors(data_dir='data', results_dir='results'):
    """Calculate focal colors for all languages - extract names from PCw CSV columns"""
    
    colorspace = pd.read_csv(Path(data_dir) / 'train_data_with_cam16_ucs.csv')
    color_id_rgb = colorspace[['color_id', 'R', 'G', 'B']]
    
    focal_colors = []
    
    for lang_name in LANGUAGES.keys():
        pcw_file = Path(results_dir) / 'PCw_results' / f"PCw_{lang_name.replace(' ', '_')}.csv"
        pcw_df = pd.read_csv(pcw_file)
        
        # Extract color names directly from PCw CSV columns (romanized names that match translation files)
        color_names = [col for col in pcw_df.columns if col != 'color_id']
        
        for color_name in color_names:
            focal_data = find_focal_color(pcw_df, color_name, color_id_rgb)
            focal_data.update({'language': lang_name, 'color_name': color_name})
            focal_colors.append(focal_data)
    
    return pd.DataFrame(focal_colors)

def add_focal_columns_to_df(df, rgb_mappings):
    """Add focal color columns after each language column"""
    new_df = df.copy()
    
    for lang_name, lang_code in LANGUAGES.items():
        if lang_code in new_df.columns and lang_name in rgb_mappings:
            mapping = rgb_mappings[lang_name]
            col_pos = new_df.columns.get_loc(lang_code)
            
            for i, rgb_col in enumerate(['focal_R', 'focal_G', 'focal_B']):
                values = new_df[lang_code].map(
                    lambda x: mapping.get(str(x).strip().lower(), {}).get(rgb_col) if pd.notna(x) else None
                )
                new_col_name = f"{lang_code}_{rgb_col}"
                new_df.insert(col_pos + 1 + i, new_col_name, values)
    
    return new_df

def add_focal_colors_to_translations(data_dir='data', results_dir='results'):
    """Add focal color RGB values to all translation files"""
    focal_colors_df = calculate_focal_colors(data_dir, results_dir)
    
    # Create RGB mappings with normalized keys
    rgb_mappings = {}
    for lang in LANGUAGES.keys():
        lang_focal = focal_colors_df[focal_colors_df['language'] == lang]
        rgb_mappings[lang] = {
            str(row['color_name']).strip().lower(): {f'focal_{col}': row[col] for col in ['R', 'G', 'B']}
            for _, row in lang_focal.iterrows()
        }
    
    # Process each translation file
    trans_dir = Path(results_dir) / 'translations/ourmethod'
    
    for lang, _ in LANGUAGES.items():
        filename = f"{lang.replace(' ', '_')}_translations.csv"
        filepath = trans_dir / filename
            
        df = pd.read_csv(filepath, encoding='utf-8-sig')
        updated_df = add_focal_columns_to_df(df, rgb_mappings)

        output_path = trans_dir / f"{lang.replace(' ', '_')}_with_focal_colours.csv"
        updated_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    
    # remove old translation files without focal colors
    for lang, _ in LANGUAGES.items():
        filename = f"{lang.replace(' ', '_')}_translations.csv"
        filepath = trans_dir / filename
        if filepath.exists():
            filepath.unlink() 

# Usage
if __name__ == "__main__":
    add_original_scripts(data_dir='data', results_dir='results')
    add_focal_colors_to_translations(data_dir='data', results_dir='results')