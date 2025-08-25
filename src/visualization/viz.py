import pandas as pd
import os 
import matplotlib.pyplot as plt
import numpy as np

def read_results(directory='results'):
    """Read all CSV files in the 'results' directory."""
    results = []
    for filename in os.listdir(directory):
        if filename.endswith('_predictions.csv'):
            file_path = os.path.join(directory, filename)
            df = pd.read_csv(file_path)
            results.append(df)
    return results

def get_unique_color_names(results):
    """Extract unique color names and their frequencies from results."""
    unique_color_names = {}
    
    for df in results:
        if df.empty:
            continue
            
        language = df['language'].iloc[0]
        
        # Load pre-computed centroids 
        centroid_file_path = f'data/{language}_rgb_centroids.csv'
        
        if not os.path.exists(centroid_file_path):
            continue
            
        centroid_file = pd.read_csv(centroid_file_path, encoding='utf-8-sig')
        
        # Create a dictionary for quick lookup of centroids
        centroid_dict = {}
        for _, row in centroid_file.iterrows():
            # Store centroids with both original name and underscore version
            original_name = row['color_name']
            underscore_name = original_name.replace(' ', '_')
            
            centroid_data = {
                'r': row['red'],
                'g': row['green'], 
                'b': row['blue']
            }
            
            # Store under both formats to handle any naming convention
            centroid_dict[original_name] = centroid_data
            centroid_dict[underscore_name] = centroid_data
        
        total_samples = len(df)
        
        # Get color frequencies and use pre-computed centroids
        color_data = {}
        for color_name, group in df.groupby('predicted_color_name'):
            if pd.isna(color_name):
                continue
            
            # Use pre-computed centroid if available, otherwise use default
            if color_name in centroid_dict:
                centroid = centroid_dict[color_name]
            else:
                print(f"Warning: No pre-computed centroid found for '{color_name}' in {language}")
                if all(col in df.columns for col in ['R_int', 'G_int', 'B_int']):
                    centroid = {
                        'r': group['R_int'].mean(),
                        'g': group['G_int'].mean(), 
                        'b': group['B_int'].mean()
                    }
                else:
                    centroid = {'r': 128, 'g': 128, 'b': 128}
                
            color_data[color_name] = {
                'frequency': len(group),
                'relative_frequency': len(group) / total_samples * 100,  
                'centroid': centroid,
                'confidence_std': group['confidence'].std()  
            }
        
        # Sort by frequency in descending order
        sorted_colors = dict(sorted(color_data.items(), 
                                  key=lambda x: x[1]['frequency'], 
                                  reverse=True))
        
        # Store sorted data
        unique_color_names[language] = {
            'frequencies': {k: v['frequency'] for k,v in sorted_colors.items()},
            'relative_frequencies': {k: v['relative_frequency'] for k,v in sorted_colors.items()},
            'centroids': {k: v['centroid'] for k,v in sorted_colors.items()},
            'confidence_std': {k: v['confidence_std'] for k,v in sorted_colors.items()},
            'count': len(sorted_colors)
        }
    
    return unique_color_names

def plot_unique_color_names(unique_color_names):
    """Plot unique color names and their relative frequencies for each language."""
    
    for language, data in unique_color_names.items():
        # Sort colors by relative frequency in descending order
        color_freq = data['relative_frequencies']
        confidence_std = data['confidence_std']
        sorted_items = sorted(color_freq.items(), key=lambda x: x[1], reverse=True)
        colors, percent = zip(*sorted_items)
        
        # Get corresponding centroid colors
        centroids = data['centroids']
        bar_colors = [(centroids[color]['r']/255, 
                      centroids[color]['g']/255, 
                      centroids[color]['b']/255) 
                     for color in colors]
        
        # Get std values (handle NaN)
        std_values = []
        for color in colors:
            if pd.isna(confidence_std[color]) or np.isnan(confidence_std[color]):
                std_values.append(0.0)
            else:
                std_values.append(confidence_std[color])
        
        # Apply square root transform to percent and error bounds
        percent = np.array(percent)
        std = np.array(std_values)
        percent_sqrt = np.sqrt(percent)
        # Calculate lower and upper bounds for error margins
        lower = np.sqrt(np.clip(percent - std, 0, None))
        upper = np.sqrt(np.clip(percent + std, 0, 100)) 
        
        # Error margins (asymmetric)
        err_lower = percent_sqrt - lower
        err_upper = upper - percent_sqrt
        
        # Create the plot
        plt.figure(figsize=(15, 8))
        bars = plt.bar(range(len(colors)), percent_sqrt, color=bar_colors, 
                      yerr=[err_lower, err_upper], capsize=5)
        
        # Customize x-axis with black labels for better visibility
        plt.xticks(range(len(colors)), colors, rotation=45, color='black')
        
        # Add labels and title
        plt.xlabel('Color Names', color='black')
        plt.ylabel('√(relative_frequencies and confidence_std)', color='black')
        plt.title(f'Color Name Distribution in {language}', color='black')
        
        # Add total count in top right
        total_colors = len(colors)
        plt.text(0.95, 0.95, f'Total Unique Colors: {total_colors}',
                transform=plt.gca().transAxes,
                horizontalalignment='right',
                verticalalignment='top',
                bbox=dict(facecolor='white', alpha=0.8),
                color='black')
        
        # Set white background
        plt.gca().set_facecolor('white')
        plt.gca().spines['bottom'].set_color('black')
        plt.gca().spines['top'].set_color('black')
        plt.gca().spines['left'].set_color('black')
        plt.gca().spines['right'].set_color('black')
        plt.gca().tick_params(colors='black')
        
        # Set y-axis to sqrt percentage range
        plt.ylim(0, 10)
        
        # Adjust layout and save
        plt.tight_layout()
        plt.savefig(f'plots/{language}_relative_frequencies.png',  
                   dpi=300, 
                   bbox_inches='tight',
                   facecolor='white')
        plt.close()

def plot_Jslices(data, language=None):
    """Plot J* slices of color data in CAM16-UCS space.
    Args:
        data (pd.DataFrame): Data containing color information.
        language (str): Language identifier for the plot title.
    """
    
    Jab_coords = data[['CAM16_J_UCS', 'CAM16_a_UCS', 'CAM16_b_UCS']].values
    pred_color_names = data['predicted_color_name'].values
    probabilities = data['confidence'].values
    
    # Load pre-computed centroids
    centroid_file_path = f'data/{language}_rgb_centroids.csv'
    if not os.path.exists(centroid_file_path):
        print(f"Error: Centroid file not found: {centroid_file_path}")
        return
    
    centroid_file = pd.read_csv(centroid_file_path, encoding='utf-8-sig')
    
    # Create a dictionary for quick lookup of centroids
    rgb_centroids = {}
    for _, row in centroid_file.iterrows():
        # Store centroids with both original name and underscore version
        original_name = row['color_name']
        underscore_name = original_name.replace(' ', '_')
        
        centroid_data = np.array([row['red'], row['green'], row['blue']])
        
        # Store under both formats to handle any naming convention
        rgb_centroids[original_name] = centroid_data
        rgb_centroids[underscore_name] = centroid_data
    
    # Normalize RGB values to [0,1] range if they're in [0,255] range
    centroid_values = np.array(list(rgb_centroids.values()))
    if np.max(centroid_values) > 1:
        for color_name in rgb_centroids:
            rgb_centroids[color_name] = rgb_centroids[color_name] / 255

    # Create figure with 3x3 subplot grid
    fig = plt.figure(figsize=(15, 12))
    ha = []
    for k in range(9):
        ax = plt.subplot(3, 3, k+1)
        ha.append(ax)
        
        # Adjust position for tighter spacing
        pos = ax.get_position()
        new_pos = [pos.x0 - 0.02,  # Move left
                pos.y0 + 0.01,  # Move up
                pos.width + 0.03,  # Make wider
                pos.height + 0.02]  # Make taller
        ax.set_position(new_pos)

    # Get lightness levels and skip very small values close to 0
    lightness_levels = np.unique(Jab_coords[:, 0])
    lightness_levels = lightness_levels[lightness_levels > 1]

    for i in range(min(9, len(lightness_levels))):
        
        # Find points at this lightness level
        is_level = np.isin(Jab_coords[:, 0], lightness_levels[i])
        if not np.any(is_level):
            continue  # Skip if no points at this level
        
        J_coords = Jab_coords[is_level, :]
        J_probability = probabilities[is_level]
        J_names = pred_color_names[is_level]
        
        # Get RGB colors for this level - with error handling
        J_rgb_centroids = []
        for name in J_names:
            if name in rgb_centroids:
                J_rgb_centroids.append(rgb_centroids[name])
            else:
                print(f"Warning: No pre-computed centroid found for '{name}' in {language}")
                # Use gray as fallback color
                J_rgb_centroids.append(np.array([0.5, 0.5, 0.5]))
        
        J_rgb_centroids = np.array(J_rgb_centroids)
        
        # Set current axes
        plt.sca(ha[i])
        
        # Create scatter plot - using RGB centroids for colors
        scatter = plt.scatter(J_coords[:, 1], J_coords[:, 2], 
                            s=(J_probability + 0.72)**5, 
                            c=J_rgb_centroids, 
                            alpha=1.0)
        
        # Set axis properties
        plt.xlim([-90, 110])
        plt.ylim([-90, 90])
        plt.gca().set_aspect('equal')
        
        # Add title showing J* value
        plt.title(f'J = {round(lightness_levels[i])}')
        plt.axis('off')

    plt.tight_layout()
    plt.savefig(f'plots/{language}_Jslices_plot.png', dpi=300, bbox_inches='tight')

def main():
    """Main execution function for the color visualization script.
    Reads results, processes unique color names, and generates plots."""

    # Ensure the plots directory exists
    os.makedirs('plots', exist_ok=True)
    
    # Read results and check if any files were found
    results = read_results(directory='results')
        
    # Process unique color names
    unique_color_names = get_unique_color_names(results)
    
    # Generate plots
    plot_unique_color_names(unique_color_names)
    
    for data in results:
        language = data['language'].iloc[0]
        plot_Jslices(data, language=language)
    print(f"Saved following plots:")
    for language in unique_color_names.keys():
        print(f"   - {language}_relative_frequencies.png")
        print(f"   - {language}_Jslices_plot.png")

if __name__ == "__main__":
    main()