"""
Perceptual Metric Evaluation of Claude Model Translations using CAM16-UCS Distance
"""

import pandas as pd
import colour
import numpy as np
import os


# CAM16-UCS viewing conditions
VIEWING_CONDITIONS = {
    "XYZ_w": np.array([95.047, 100.000, 108.883]) / 100,  # D65 whitepoint (scaled)
    "L_A": 64,
    "Y_b": 20,
    "surround": colour.VIEWING_CONDITIONS_CAM16['Average']
}


def rgb_to_cam16ucs(rgb_data):
    """Convert RGB data to CAM16-UCS"""
    xyz_data = colour.sRGB_to_XYZ(rgb_data / 255)
    return colour.XYZ_to_CAM16UCS(xyz_data, XYZ_w=VIEWING_CONDITIONS["XYZ_w"], 
                                  L_A=VIEWING_CONDITIONS["L_A"], 
                                  Y_b=VIEWING_CONDITIONS["Y_b"],
                                  surround=VIEWING_CONDITIONS["surround"])


def cam16_distance(a, b):
    """Calculate CAM16-UCS Euclidean distance"""
    return np.linalg.norm(a - b, axis=1)


def process_model(lang_name, lang_abbr, source_col, targets, model_name, model_path, output_dir):
    """Process one source language for a specific model"""
    
    # Load data
    gt_file = f"results/translations/ourmethod/{lang_name}_with_focal_colours.csv"
    ourmethod_data = pd.read_csv(gt_file, encoding='utf-8-sig')
    
    if model_name == 'claude':
        model_data = pd.read_excel(f"{model_path}/{lang_name}.xlsx")
    else:
        model_data = pd.read_csv(f"{model_path}/{lang_name}.csv", encoding='utf-8-sig')
    
    # Initialize results
    result_data = {source_col: model_data[source_col].values}
    
    # Calculate distances for each target language
    for target_abbr in targets:
        model_cols = [f'{target_abbr}_Red', f'{target_abbr}_Green', f'{target_abbr}_Blue']
        gt_cols = [f'{target_abbr}_focal_R', f'{target_abbr}_focal_G', f'{target_abbr}_focal_B']
        
        model_cam16 = rgb_to_cam16ucs(model_data[model_cols].values)
        gt_cam16 = rgb_to_cam16ucs(ourmethod_data[gt_cols].values)
        
        distances = cam16_distance(model_cam16, gt_cam16)
        result_data[f'{target_abbr}_ED'] = distances
    
    # Save results
    output_file = os.path.join(output_dir, model_name, f"{lang_name}_cam16ucs_{model_name}.csv")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    pd.DataFrame(result_data).to_csv(output_file, index=False, encoding='utf-8-sig')


def main():
    """Main execution"""
    
    # Configuration
    models = {
        'claude': 'results/translations/claude',
    }
    
    languages = [
        ('American_English', 'AE', 'AE', ['BE', 'FR', 'GR']),
        ('British_English', 'BE', 'BE', ['AE', 'FR', 'GR']),
        ('French', 'FR', 'FR', ['AE', 'BE', 'GR']),
        ('Greek', 'GR', 'GR', ['AE', 'BE', 'FR']),
    ]
    
    output_dir = 'results/metrics/perceptual'
    
    # Process all languages for all models
    for lang_name, lang_abbr, source_col, targets in languages:
        for model_name, model_path in models.items():
            process_model(lang_name, lang_abbr, source_col, targets, 
                         model_name, model_path, output_dir)

if __name__ == "__main__":
    main()