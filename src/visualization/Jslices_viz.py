#!/usr/bin/env python3
"""
Create lightness slice visualizations for color predictions across languages.
"""

import pandas as pd
import os 
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def load_prediction_data(results_dir='results/predictions'):
    """Load all prediction CSV files."""
    data = []
    for file in Path(results_dir).glob('*_predictions.csv'):
        data.append(pd.read_csv(file))
    return data

def reorder_by_language(data, language_order):
    """Reorder data by specified language order."""
    lang_map = {df['language'].iloc[0]: df for df in data}
    return [lang_map[lang] for lang in language_order if lang in lang_map]

def load_color_centroids(language_name):
    """Load RGB centroids for a language."""
    centroid_file = f'data/{language_name}_rgb_centroids.csv'
    if not Path(centroid_file).exists():
        print(f"Missing: {centroid_file}")
        return {}
    
    centroids = pd.read_csv(centroid_file, encoding='utf-8-sig')
    color_map = {}
    for _, row in centroids.iterrows():
        rgb = np.array([row['red'], row['green'], row['blue']])
        if rgb.max() > 1:
            rgb = rgb / 255
        color_map[row['color_name']] = rgb
        color_map[row['color_name'].replace(' ', '_')] = rgb
    return color_map

def plot_lightness_slice(ax, coords, names, confidences, color_map, 
                        lightness_val, x_bounds, y_bounds, margins):
    """Plot a single lightness slice."""
    # Apply margins if specified
    if margins and any(margins.values()):
        x_range = x_bounds[1] - x_bounds[0]
        y_range = y_bounds[1] - y_bounds[0]
        
        x_min = x_bounds[0] + margins.get('left', 0) * x_range
        x_max = x_bounds[1] - margins.get('right', 0) * x_range
        y_min = y_bounds[0] + margins.get('bottom', 0) * y_range
        y_max = y_bounds[1] - margins.get('top', 0) * y_range
        
        mask = ((coords[:, 1] >= x_min) & (coords[:, 1] <= x_max) &
                (coords[:, 2] >= y_min) & (coords[:, 2] <= y_max))
        
        coords = coords[mask]
        names = names[mask]
        confidences = confidences[mask]
    
    if len(coords) == 0:
        ax.axis('off')
        return
    
    # Get colors for visualization
    colors = [color_map.get(name, [0.5, 0.5, 0.5]) for name in names]
    sizes = (confidences + 0.72)**5
    
    ax.scatter(coords[:, 1], coords[:, 2], s=sizes, c=colors, alpha=1.0)
    ax.set_xlim(x_bounds)
    ax.set_ylim(y_bounds)
    ax.set_aspect('equal')
    ax.axis('off')

def create_lightness_plots(data, slice_margins=None):
    """Create combined lightness slice plots for all languages."""
    
    fig, axes = plt.subplots(5, 9, figsize=(12, 6))
    plt.subplots_adjust(hspace=0.1, wspace=0.05)
    
    # Get global coordinate bounds
    all_coords = np.vstack([df[['CAM16_J_UCS', 'CAM16_a_UCS', 'CAM16_b_UCS']].values 
                           for df in data])
    x_bounds = (all_coords[:, 1].min(), all_coords[:, 1].max())
    y_bounds = (all_coords[:, 2].min(), all_coords[:, 2].max())
    
    if slice_margins is None:
        slice_margins = [{}] * 9
    
    for lang_idx, df in enumerate(data[:5]):
        language = df['language'].iloc[0]
        display_name = language.replace('_', ' ')
        
        coords = df[['CAM16_J_UCS', 'CAM16_a_UCS', 'CAM16_b_UCS']].values
        names = df['predicted_color_name'].values
        confidences = df['confidence'].values
        color_map = load_color_centroids(language)
        
        # Get unique lightness levels
        lightness_levels = np.unique(coords[:, 0])
        lightness_levels = lightness_levels[lightness_levels > 1]
        
        for slice_idx in range(9):
            ax = axes[lang_idx, slice_idx]
            
            if slice_idx >= len(lightness_levels):
                ax.axis('off')
                continue
            
            # Filter by lightness level
            lightness_val = lightness_levels[slice_idx]
            mask = coords[:, 0] == lightness_val
            
            if not mask.any():
                ax.axis('off')
                continue
            
            slice_coords = coords[mask]
            slice_names = names[mask]
            slice_confidences = confidences[mask]
            margins = slice_margins[slice_idx] if slice_idx < len(slice_margins) else {}
            
            plot_lightness_slice(ax, slice_coords, slice_names, slice_confidences,
                                color_map, lightness_val, x_bounds, y_bounds, margins)
            
            # Add titles and labels
            if lang_idx == 0:
                title = f'J = {round(lightness_val)}'
                x_pos = 0.25 if round(lightness_val) in [83, 92] else 0.5
                ax.set_title(title, fontsize=9, ha='center', x=x_pos)
            
            if slice_idx == 0:
                ax.text(0.1, 0.5, display_name, transform=ax.transAxes, 
                       rotation=90, va='center', ha='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('plots/all_languages_lightness_slices.pdf', dpi=300)
    plt.close()

def main():
    """Main execution function."""
    Path('plots').mkdir(exist_ok=True)
    
    # Load and reorder data
    data = load_prediction_data('results/predictions')
    language_order = ['American_English', 'British_English', 'Greek', 'French', 'Himba']
    ordered_data = reorder_by_language(data, language_order)
    
    # Define slice margins
    margins = [
        {'left':0.1, 'right':0.1, 'bottom':0, 'top':0},
        {'left':0.1, 'right':0.1, 'bottom':0, 'top':0},
        {'left':0, 'right':0, 'bottom':0, 'top':0.0},
        {'left':0, 'right':0, 'bottom':0, 'top':0},
        {'left':0, 'right':0, 'bottom':0.05, 'top':0},
        {'left':0, 'right':0.1, 'bottom':0, 'top':0},
        {'left':0, 'right':0.05, 'bottom':0.05, 'top':0},
        {'left':0, 'right':0, 'bottom':0, 'top':0},
        {'left':0, 'right':0, 'bottom':0, 'top':0}
    ]
    
    create_lightness_plots(ordered_data, margins)
    print("Visualization saved to plots/all_languages_lightness_slices.pdf")

if __name__ == "__main__":
    main()