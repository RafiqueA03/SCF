import numpy as np
import matplotlib.pyplot as plt
import colour
from colour.appearance.cam16 import XYZ_to_CAM16, CAM16_to_XYZ, InductionFactors_CAM16, CAM_Specification_CAM16
from colour.models import RGB_COLOURSPACE_sRGB
from colour import JMh_CAM16_to_CAM16UCS, CAM16UCS_to_JMh_CAM16
import pandas as pd
import logging

# ---- PARAMETERS ----
SPACING = 2.0          # CAM16UCS grid spacing (ΔE units)
A_RANGE = 70           # a' coordinate range
B_RANGE = 70           # b' coordinate range  
J_VALUES = np.array([20, 30, 40, 50, 60, 70, 80, 90])  # Lightness levels for uniform grid

# ---- VIEWING CONDITIONS sRGB ----
VIEWING_CONDITIONS = {
    "XYZ_w": np.array([95.047, 100.000, 108.883]),  # D65 whitepoint
    "L_A": 64,
    "Y_b": 20,
    "surround": colour.VIEWING_CONDITIONS_CAM16['Average']
}

def map_gamut_slice(Jp_ucs, a_range=70, b_range=70, spacing=2.0):
    """
    Create uniform CAM16UCS grid and filter for sRGB-compatible colors.
    
    Args:
        Jp_ucs: Target lightness level in CAM16UCS (J')
        a_range, b_range: CAM16UCS coordinate ranges to test
        spacing: Grid spacing in CAM16UCS units
        
    Returns:
        tuple: (CAM16UCS points, RGB colors) for points within sRGB gamut
    """
    # Create uniform CAM16UCS grid coordinates
    a_coords = np.arange(-a_range, a_range + spacing, spacing)
    b_coords = np.arange(-b_range, b_range + spacing, spacing)
    
    total_points = len(a_coords) * len(b_coords)
    
    valid_points = []
    valid_colors = []
    conversion_failures = 0
    
    for a in a_coords:
        for b in b_coords:
            try:
                # Start with uniform CAM16UCS coordinates (pure UCS)
                jab_point = np.array([Jp_ucs, a, b])
                
                # Convert CAM16UCS → JMh → XYZ → RGB to test sRGB validity
                jmh = CAM16UCS_to_JMh_CAM16(jab_point)
                J, M, h = jmh
                
                # Skip invalid CAM16 values
                if not (J > 0 and M >= 0 and np.isfinite([J, M, h]).all()):
                    continue
                
                # Convert to XYZ - FIX: Use M for colorfulness, not C for chroma
                cam_spec = CAM_Specification_CAM16(J=J, M=M, h=h)
                xyz = CAM16_to_XYZ(cam_spec, **VIEWING_CONDITIONS)
                
                # XYZ → RGB (no clipping)
                xyz_norm = xyz / 100.0
                rgb = colour.XYZ_to_RGB(xyz_norm, colourspace=RGB_COLOURSPACE_sRGB)
                
                # Keep only points that map to valid sRGB colors
                if np.all((rgb >= 0) & (rgb <= 1)):
                    valid_points.append(jab_point)
                    valid_colors.append(rgb)
                    
            except (ValueError, RuntimeWarning, Exception):
                conversion_failures += 1
                continue
    
    return np.array(valid_points), np.array(valid_colors)


def getCAM16UCS_samples():
    """Main execution function with pure CAM16-UCS coordinates."""
    logging.info("Generating CAM16-UCS grid samples...")
    
        
    # Process all lightness slices
    slices = {}
    rgb_colors = {}
    total_samples = 0
    
    # Collect all data for CSV export
    all_data = []
    
    for Jp_ucs in J_VALUES:
        points, colors = map_gamut_slice(Jp_ucs, A_RANGE, B_RANGE, SPACING)
        
        slices[Jp_ucs] = points
        rgb_colors[Jp_ucs] = colors
        total_samples += len(points)
        
        # Add to CSV data
        for i in range(len(points)):
            J_prime, a_prime, b_prime = points[i]
            r, g, b = colors[i]
            all_data.append([J_prime, a_prime, b_prime, r, g, b])

    # Convert to arrays
    all_slices = np.vstack([points for points in slices.values() if len(points) > 0])
    logging.info(f"Generated {len(all_slices)} total CAM16-UCS points valid for sRGB gamut.")
    all_rgb_colors = np.vstack([colors for colors in rgb_colors.values() if len(colors) > 0])
    
    # Save to CSV with proper UCS coordinates
    df = pd.DataFrame(all_data, columns=['J_prime', 'a_prime', 'b_prime', 'R', 'G', 'B'])
    df.to_csv('data/sRGB_gamut_CAM16UCS_test_grid.csv', index=False)
    logging.info("Saved CAM16-UCS test grid to data/sRGB_gamut_CAM16UCS_test_grid.csv")
    
    return all_slices, all_rgb_colors, total_samples

# Execute the function
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    all_slices, all_rgb_colors, total_samples = getCAM16UCS_samples()