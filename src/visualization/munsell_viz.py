import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.ndimage import binary_fill_holes
from skimage.measure import find_contours
import cv2

# Load all required data
et_munsell = loadmat('results/BritishEnglish_munsell_results.mat')
munsell_array = loadmat('data/Munsell_Array_330.mat')
munsell_hue = loadmat('data/Munsell_hue_notation.mat')
sturges_boundaries = loadmat('data/SturgesWhitfielColourBoundries.mat')

centroids = et_munsell['centroids']
colourName = et_munsell['colourName']
munsell330 = munsell_array['munsell330']
MunsellHue = munsell_hue['MunsellHue']
ChipsTable = sturges_boundaries['ChipsTable']

# Load RGB centroids from CSV
rgb_centroids_data = pd.read_csv('data/British_English_rgb_centroids.csv')

# Clean color names to match
colourName_clean = [name.strip().replace('_', ' ') for name in colourName]

# Create mapping from color names to RGB centroids
color_to_rgb_map = {}
for i in range(len(rgb_centroids_data)):
    color_name = rgb_centroids_data.iloc[i]['color_name'].strip().replace('_', ' ')
    rgb_values = np.array([
        rgb_centroids_data.iloc[i]['red'],
        rgb_centroids_data.iloc[i]['green'],
        rgb_centroids_data.iloc[i]['blue']
    ])
    color_to_rgb_map[color_name] = rgb_values

# Replace each Munsell chip color with its predicted color name centroid
centroid_colors = np.zeros((330, 3))
for i in range(330):
    predicted_color = colourName_clean[i]
    if predicted_color in color_to_rgb_map:
        centroid_colors[i, :] = color_to_rgb_map[predicted_color]
    else:
        centroid_colors[i, :] = centroids[i, :]
        print(f'Color "{predicted_color}" not found in centroids, using original')

# Separate chromatic and achromatic using centroid colors
chromatic_mask = munsell330[:, 3] > 0
achromatic_mask = munsell330[:, 3] == 0

chromatic_indices = np.where(chromatic_mask)[0]
achromatic_indices = np.where(achromatic_mask)[0]

centroid_colors_chromatic = centroid_colors[chromatic_indices, :]
centroid_colors_achromatic = centroid_colors[achromatic_indices, :]

# Create integrated visualization
fig, ax = plt.subplots(1, 1, figsize=(14, 4))

# Create the chromatic array
if len(chromatic_indices) == 320:
    c = centroid_colors_chromatic.reshape((40, 8, 3), order='F') 
    c_rotated = np.rot90(c, k=-1, axes=(0, 1))
    
    # Convert to uint8
    c_uint8 = np.clip(c_rotated, 0, 255).astype(np.uint8)
    
    # Resize by factor of 20
    im_chromatic = cv2.resize(c_uint8, (c_uint8.shape[1]*20, c_uint8.shape[0]*20), 
                             interpolation=cv2.INTER_NEAREST)
    
    # Flip along second dimension (flip left-right)
    im_chromatic = np.fliplr(im_chromatic)
    
    # Create the achromatic strip
    num_achromatic = len(achromatic_indices)
    cA = centroid_colors_achromatic.reshape((num_achromatic, 1, 3))
    cA_uint8 = np.clip(cA, 0, 255).astype(np.uint8)
    
    # Resize achromatic to match chromatic height
    im_achromatic = cv2.resize(cA_uint8, (20, im_chromatic.shape[0]), 
                              interpolation=cv2.INTER_NEAREST)
    
    # Create gap
    gap_width = 20
    gap_section = np.ones((im_chromatic.shape[0], gap_width, 3), dtype=np.uint8) * 255
    
    # Combine horizontally
    combined_image = np.concatenate([im_achromatic, gap_section, im_chromatic], axis=1)
    
    # Display image
    ax.imshow(combined_image, aspect='auto')
    #ax.grid(True, alpha=0.3)
    
    # Set up ticks and labels
    achromatic_width = im_achromatic.shape[1]
    gap_width_actual = gap_section.shape[1]
    chromatic_width = im_chromatic.shape[1]
    
    # X-axis labels
    all_ticks = []
    all_labels = []
    
    # Achromatic center
    all_ticks.append(achromatic_width / 2)
    all_labels.append('N')
    
    # Chromatic hue labels
    chromatic_start = achromatic_width + gap_width_actual
    hue_column_width = chromatic_width / 40
    
    for i in range(40):
        hue_center = chromatic_start + (i + 0.5) * hue_column_width
        all_ticks.append(hue_center)
        hue_label = MunsellHue[0, i][0]
        all_labels.append(hue_label)
    
    ax.set_xticks(all_ticks)
    ax.set_xticklabels(all_labels, rotation=90)
    ax.tick_params(length=0)
    ax.set_ylabel('Value', fontsize=12)
    
    num_ticks_y = 9
    y_ticks = np.linspace(0, im_chromatic.shape[0]-1, num_ticks_y)
    uV = [9, 8, 7, 6, 5, 4, 3, 2, 1]
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(uV)
    
    # Add boundaries for achromatic color regions based on actual predictions
    achromatic_height = im_achromatic.shape[0]
    
    # Get the predicted color names for achromatic chips and their value levels
    achromatic_predictions = []
    achromatic_values = []
    
    for idx in achromatic_indices:
        predicted_color = colourName_clean[idx]
        value_level = munsell330[idx, 5]  # Assuming value is in column 5
        achromatic_predictions.append(predicted_color)
        achromatic_values.append(value_level)
    
    # Create a mapping of value to predicted color
    value_color_pairs = list(zip(achromatic_values, achromatic_predictions))
    value_color_pairs.sort(key=lambda x: x[0])  # Sort by value
    
    # Group consecutive chips with the same predicted color
    color_regions = []
    if value_color_pairs:
        current_color = value_color_pairs[0][1]
        start_value = value_color_pairs[0][0]
        
        for i in range(1, len(value_color_pairs)):
            val, color = value_color_pairs[i]
            
            # If color changes, end current region and start new one
            if color != current_color:
                color_regions.append({
                    "name": current_color,
                    "value_range": (start_value, value_color_pairs[i-1][0])
                })
                current_color = color
                start_value = val
        
        # Add the last region
        color_regions.append({
            "name": current_color,
            "value_range": (start_value, value_color_pairs[-1][0])
        })
    
    # Draw boundaries for each color region
    if color_regions and len(value_color_pairs) > 1:
        # Convert value ranges to pixel coordinates
        min_value = min(achromatic_values)
        max_value = max(achromatic_values)
        value_range = max_value - min_value
        
        for region in color_regions:
            min_val, max_val = region["value_range"]
            
            # Convert to pixel coordinates (remember display: higher values at bottom)
            # Normalize values to 0-1 range, then scale to pixel height
            if value_range > 0:
                y_top = achromatic_height * (1 - (max_val - min_value) / value_range)
                y_bottom = achromatic_height * (1 - (min_val - min_value) / value_range)
            else:
                y_top = 0
                y_bottom = achromatic_height
            
            # Draw rectangle boundary for this color region
            # Top horizontal line
            ax.plot([0, achromatic_width], [y_top, y_top], 
                   'k-', linewidth=2, solid_capstyle='round')
            # Bottom horizontal line  
            ax.plot([0, achromatic_width], [y_bottom, y_bottom], 
                   'k-', linewidth=2, solid_capstyle='round')
            # Left vertical line
            ax.plot([0, 0], [y_top, y_bottom], 
                   'k-', linewidth=2, solid_capstyle='round')
            # Right vertical line
            ax.plot([achromatic_width, achromatic_width], [y_top, y_bottom], 
                   'k-', linewidth=2, solid_capstyle='round')
    
    # Add S&W boundaries for chromatic section
    boundary_offset = achromatic_width + gap_width_actual
    
    for i in range(ChipsTable.shape[2]):
        ib = ChipsTable[:, :, i].copy()
        if ib.size > 0 and ib.shape[0] > 1 and ib.shape[1] > 1:
            # Remove first row and first column (MATLAB: ib(:,1) = []; ib(1,:) = [];)
            ib = ib[1:, 1:]
            
            if ib.size > 0:
                # Resize to match the scaling factor used for the main image
                ic = cv2.resize(ib.astype(np.uint8), (ib.shape[1]*20, ib.shape[0]*20), 
                               interpolation=cv2.INTER_NEAREST)
                
                # Fill holes and find contours
                Ifill = binary_fill_holes(ic > 0).astype(np.uint8)
                contours = find_contours(Ifill, 0.5)
                
                for contour in contours:
                    # Ensure contour is closed and plot with proper offset
                    # Close the contour if it's not already closed
                    if not np.array_equal(contour[0], contour[-1]):
                        contour = np.vstack([contour, contour[0]])
                    
                    # Plot boundary with offset, ensuring full visibility
                    ax.plot(contour[:, 1] + boundary_offset, contour[:, 0], 
                           'k-', linewidth=2, solid_capstyle='round', solid_joinstyle='round')
    
    # Remove spines to match MATLAB appearance
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

plt.tight_layout()
plt.savefig(f'plots/Munsell_array.png', dpi=300, bbox_inches='tight')

