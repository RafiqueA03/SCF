import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.ndimage import binary_fill_holes
from skimage.measure import find_contours
import cv2

# Configuration
plt.rcParams['font.size'] = 9
FIGURE_SIZE = (14, 4)
OUTPUT_PATH = 'plots/munsell_array.pdf'
DPI = 300

# Load all required data
et_munsell = loadmat('results/BritishEnglish_munsell_results.mat')
munsell_array = loadmat('data/Munsell_Array_330.mat')
munsell_hue = loadmat('data/Munsell_hue_notation.mat')
sturges_boundaries = loadmat('data/SturgesWhitfielColourBoundries.mat')

# Extract data from loaded files
centroids = et_munsell['centroids']
colourName = et_munsell['colourName']
munsell330 = munsell_array['munsell330']
MunsellHue = munsell_hue['MunsellHue']
ChipsTable = sturges_boundaries['ChipsTable']

# Load RGB centroids from CSV
rgb_centroids_data = pd.read_csv('data/British_English_rgb_centroids.csv')

# Clean color names for matching
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

# Separate chromatic and achromatic colors
chromatic_mask = munsell330[:, 3] > 0
achromatic_mask = munsell330[:, 3] == 0

chromatic_indices = np.where(chromatic_mask)[0]
achromatic_indices = np.where(achromatic_mask)[0]

centroid_colors_chromatic = centroid_colors[chromatic_indices, :]
centroid_colors_achromatic = centroid_colors[achromatic_indices, :]

# Create the visualization
fig, ax = plt.subplots(1, 1, figsize=FIGURE_SIZE)

# Create the chromatic array
if len(chromatic_indices) == 320:
    # Reshape to 40x8x3 using MATLAB-style column-major order
    c = centroid_colors_chromatic.reshape((40, 8, 3), order='F')
    
    # Rotate -90 degrees (clockwise by 90 degrees)
    c_rotated = np.rot90(c, k=-1, axes=(0, 1))
    
    # Convert to uint8 and resize
    c_uint8 = np.clip(c_rotated, 0, 255).astype(np.uint8)
    im_chromatic = cv2.resize(c_uint8, (c_uint8.shape[1]*20, c_uint8.shape[0]*20), 
                             interpolation=cv2.INTER_NEAREST)
    im_chromatic = np.fliplr(im_chromatic)  # Flip left-right
    
    # Create the achromatic strip
    num_achromatic = len(achromatic_indices)
    cA = centroid_colors_achromatic.reshape((num_achromatic, 1, 3))
    cA_uint8 = np.clip(cA, 0, 255).astype(np.uint8)
    
    # Resize achromatic to match chromatic height
    im_achromatic = cv2.resize(cA_uint8, (20, im_chromatic.shape[0]), 
                              interpolation=cv2.INTER_NEAREST)
    
    # Create gap between achromatic and chromatic sections
    gap_width = 20
    gap_section = np.ones((im_chromatic.shape[0], gap_width, 3), dtype=np.uint8) * 255
    
    # Combine all sections horizontally
    combined_image = np.concatenate([im_achromatic, gap_section, im_chromatic], axis=1)
    
    # Display the combined image
    ax.imshow(combined_image, aspect='auto')
    
    # Setup axis dimensions
    achromatic_width = im_achromatic.shape[1]
    gap_width_actual = gap_section.shape[1]
    chromatic_width = im_chromatic.shape[1]
    
    # Setup X-axis labels
    all_ticks = []
    all_labels = []
    
    # Achromatic center label
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
    
    # Apply X-axis labels with selective rotation
    ax.set_xticks(all_ticks)
    labels = ax.set_xticklabels(all_labels)
    
    # Rotate all labels except 'N' (achromatic)
    for i, label in enumerate(labels):
        if i == 0:  # First label is 'N'
            label.set_rotation(0)
        else:
            label.set_rotation(90)
    
    ax.tick_params(length=0)
    ax.set_ylabel('Value', fontsize=9)
    
    # Setup Y-axis labels
    num_ticks_y = 9
    y_ticks = np.linspace(0, im_chromatic.shape[0]-1, num_ticks_y)
    uV = [9, 8, 7, 6, 5, 4, 3, 2, 1]  # Munsell values from 9 (top) to 1 (bottom)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(uV)
    
    # Add rectangular boxes for achromatic color regions
    achromatic_height = im_achromatic.shape[0]
    
    # Define color regions: (start_value, end_value)
    color_boxes = [
        (1.5, 2.5),    # Black region
        (3.5, 6.5),    # Grey region  
        (8, 9)         # White region
    ]
    
    for start_val, end_val in color_boxes:
        # Convert Munsell values to pixel coordinates
        y_top = achromatic_height * (1 - (end_val - 1) / 8)
        y_bottom = achromatic_height * (1 - (start_val - 1) / 8)
        
        # Draw complete rectangular box
        ax.plot([0, achromatic_width], [y_top, y_top], 'k-', linewidth=2, solid_capstyle='round')
        ax.plot([0, achromatic_width], [y_bottom, y_bottom], 'k-', linewidth=2, solid_capstyle='round')
        ax.plot([0, 0], [y_top, y_bottom], 'k-', linewidth=2, solid_capstyle='round')
        ax.plot([achromatic_width, achromatic_width], [y_top, y_bottom], 'k-', linewidth=2, solid_capstyle='round')
    
    # Add Sturges & Whitfield boundaries for chromatic section
    boundary_offset = achromatic_width + gap_width_actual
    
    for i in range(ChipsTable.shape[2]):
        ib = ChipsTable[:, :, i].copy()
        if ib.size > 0 and ib.shape[0] > 1 and ib.shape[1] > 1:
            # Remove first row and column (MATLAB equivalent)
            ib = ib[1:, 1:]
            
            if ib.size > 0:
                # Resize to match scaling factor
                ic = cv2.resize(ib.astype(np.uint8), (ib.shape[1]*20, ib.shape[0]*20), 
                               interpolation=cv2.INTER_NEAREST)
                
                # Fill holes and find contours
                Ifill = binary_fill_holes(ic > 0).astype(np.uint8)
                contours = find_contours(Ifill, 0.5)
                
                for contour in contours:
                    # Close contour if not already closed
                    if not np.array_equal(contour[0], contour[-1]):
                        contour = np.vstack([contour, contour[0]])
                    
                    # Plot boundary with offset
                    ax.plot(contour[:, 1] + boundary_offset, contour[:, 0], 
                           'k-', linewidth=2, solid_capstyle='round', solid_joinstyle='round')
    
    # Remove all spines for clean publication appearance
    for spine in ax.spines.values():
        spine.set_visible(False)

# Save the figure
plt.tight_layout()
plt.savefig(OUTPUT_PATH, dpi=DPI, bbox_inches='tight')