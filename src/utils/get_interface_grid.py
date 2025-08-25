import numpy as np
import colour
from colour.models import RGB_COLOURSPACE_sRGB

VIEWING_CONDITIONS = {
    "XYZ_w": np.array([95.047, 100.000, 108.883]),  # D65
    "L_A": 64,
    "Y_b": 20,
    "surround": colour.VIEWING_CONDITIONS_CAM16['Average']
}

def generate_rgb_grid(step_size=8):
    """
    Generate uniform RGB grid with specified step size.
    
    Args:
        step_size: Step size for RGB values (default: 8)
        
    Returns:
        tuple: (RGB array [N x 3], total_samples)
    """
    
    # Generate RGB coordinate arrays
    rgb_coords = np.arange(0, 256, step_size)
    
    # Ensure we include 255 if it's not already there
    if rgb_coords[-1] != 255:
        rgb_coords = np.append(rgb_coords, 255)
    
    # Create 3D grid
    R, G, B = np.meshgrid(rgb_coords, rgb_coords, rgb_coords, indexing='ij')
    
    # Flatten to get all combinations
    rgb_grid = np.column_stack([R.flatten(), G.flatten(), B.flatten()])
    # convert to xyz 
    xyz_grid = colour.RGB_to_XYZ(rgb_grid/ 255.0, RGB_COLOURSPACE_sRGB)
    xyz_w_scaled = VIEWING_CONDITIONS["XYZ_w"] / 100  # Scale white point to [0,1]
    cam16ucs = colour.XYZ_to_CAM16UCS(xyz_grid, XYZ_w=xyz_w_scaled, L_A=VIEWING_CONDITIONS["L_A"], Y_b=VIEWING_CONDITIONS["Y_b"])
    
    total_samples = len(rgb_grid)
    
    return rgb_grid, cam16ucs, total_samples