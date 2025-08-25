# Standard library imports
import logging
import os

# Third-party imports
import numpy as np
import pandas as pd
import scipy.io as sio
import colour
from sklearn.preprocessing import StandardScaler
from spincam.rotated_extra_trees_regressor import RotatedExtraTreesRegressor
from spincam.language_model_evaluator import LanguageModelEvaluator

# Set random seed for reproducibility
np.random.seed(42) 
temp_evaluator = LanguageModelEvaluator(test_grid_csv=None)
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# Define viewing conditions for CAM16-UCS conversion
VIEWING_CONDITIONS = {
    "XYZ_w": np.array([95.047, 100.000, 108.883]),  # D65
    "L_A": 64,
    "Y_b": 20,
    "surround": colour.VIEWING_CONDITIONS_CAM16['Average']
}

def extract_last_word(color_name):
    """Extract the last word after splitting by underscore"""
    if pd.isna(color_name):
        return color_name
    words = str(color_name).split('_')
    return words[-1]  # Get the last word

def prepare_single_word_filtered_data():
    """
    Load and prepare British English data with single-word filtering applied before training.
    Returns the training data in the format needed for the model.
    """
    
    # Load processed data
    df_main = pd.read_csv('data/British_English_processed.csv', encoding='utf-8-sig')
    
    logging.info(f"Original color names: {len(np.unique(df_main['color_name']))}")
    
    # Apply single-word filtering BEFORE training
    df_main['lastWord'] = df_main['color_name'].apply(extract_last_word)
    df_main['color_name'] = df_main['lastWord']
    
    logging.info(f"After single-word filtering: {len(np.unique(df_main['color_name']))}")
    
    # Get unique colors based on color_id and CAM16-UCS coordinates
    unique_colors = df_main.groupby(['color_id', 'J_lightness', 'a_prime', 'b_prime']).size().reset_index().rename(columns={0: 'count'})
    
    # Get sorted color names
    sData = df_main.sort_values(by='color_name')
    color_names = np.sort(sData['color_name'].unique())
    
    # Compute conditional probabilities P(W|c)
    Count_WC = pd.crosstab(sData['color_name'], sData['color_id'])
    Count_C = Count_WC.sum(axis=0)
    ones_names = np.ones(len(color_names))
    PWc = Count_WC / np.outer(ones_names, Count_C)
    
    # Create color_id to P(W|c) mapping
    color_id_to_pwc = {}
    for i, color_id in enumerate(Count_WC.columns):
        color_id_to_pwc[color_id] = PWc.iloc[:, i].values
    
    # Prepare regression data
    X = unique_colors[['J_lightness', 'a_prime', 'b_prime']].values
    y = np.zeros((len(unique_colors), len(color_names)))
    
    for i, color_id in enumerate(unique_colors['color_id']):
        if color_id in color_id_to_pwc:
            y[i, :] = color_id_to_pwc[color_id]
    
    # Get training color ids for reference
    training_color_ids = unique_colors['color_id'].values
    
    logging.info(f"Prepared training data: {X.shape[0]} samples, {len(color_names)} color names")
    
    return X, y, color_names, PWc, training_color_ids

def train_and_predict_munsell():
    """
    Train model with single-word filtering and predict Munsell colors.
    """
    # Prepare filtered training data
    X_train, y_train, color_names, sqrtPWc, training_color_ids = prepare_single_word_filtered_data()
    
    logging.info(f"Training on {X_train.shape[0]} samples, {len(color_names)} color names")
    
    # Scale the training features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Optimal parameters from hyperparameter optimization
    optimal_params = {
        'n_estimators': 367, 
        'max_depth': 13, 
        'min_samples_split': 3, 
        'min_samples_leaf': 1, 
        'max_features': 0.7216968971838151, 
        'rotation_fraction': 0.6148538589793427,
        'random_state': 42,
        'n_jobs': -1
    }
    
    # Create and train the model
    logging.info("Training RotatedExtraTrees model with optimal parameters...")
    model = RotatedExtraTreesRegressor(**optimal_params)
    model.fit(X_train_scaled, y_train)
    
    # Load Munsell data
    munsell_mat = sio.loadmat('data/Munsell_Array_330.mat')
    munsell330 = munsell_mat['munsell330']
    
    # Convert Munsell Lab coordinates to CAM16UCS
    lab_coords = munsell330[:, 6:9]  # L*, a*, b*
    
    # Convert Lab to XYZ using D65 illuminant
    xyz_coords = colour.Lab_to_XYZ(lab_coords, illuminant=colour.CCS_ILLUMINANTS['CIE 1931 2 Degree Standard Observer']['D65'])
    # XYZ_to_CAM16UCS expects XYZ values in [0,1] range, and XYZ_w also in [0,1]
    xyz_w_scaled = VIEWING_CONDITIONS["XYZ_w"] / 100  # Scale white point to [0,1]
    cam16ucs = colour.XYZ_to_CAM16UCS(xyz_coords, XYZ_w=xyz_w_scaled, L_A=VIEWING_CONDITIONS["L_A"], Y_b=VIEWING_CONDITIONS["Y_b"])
    
    # The CAM16UCS coordinates are in format [J', a', b']
    test_coords = cam16ucs
    
    # Scale the test coordinates using the same scaler
    test_coords_scaled = scaler.transform(test_coords)
    
    # Make predictions using the trained model
    logging.info("Making predictions on 330 Munsell chips...")
    predictions = model.predict(test_coords_scaled)
    
    # Ensure non-negative and normalize to get proper probabilities
    predictions = np.maximum(predictions, 0)  # Ensure non-negative
    predictions = predictions / (np.sum(predictions, axis=1, keepdims=True) + 1e-10)
    
    # Get predicted color names and confidences
    predicted_indices = np.argmax(predictions, axis=1)
    confidences = np.max(predictions, axis=1)
    predicted_names = [color_names[idx] for idx in predicted_indices]
    
    # Clean up names (replace underscores with spaces)
    predicted_names = [name.replace('_', ' ') for name in predicted_names]
    
    logging.info(f"Predicted {len(np.unique(predicted_names))} unique color names")
    logging.info(f"Unique names: {np.unique(predicted_names)}")
    
    # Debug: Print first few predictions
    for i in range(5):
        logging.info(f"Munsell chip {i}: '{predicted_names[i]}' (confidence: {confidences[i]:.3f})")
    
    # Save results in MATLAB format
    save_matlab_results(predicted_names, xyz_coords, munsell330)
    
def save_matlab_results(predicted_names, xyz_coords, munsell330):
    """
    Save results in MATLAB format matching the reference code.
    """
    rgb_from_xyz = colour.XYZ_to_sRGB(xyz_coords)
    rgb_from_xyz = np.clip(rgb_from_xyz * 255, 0, 255)
    
    # For centroid (330x9 matrix): use all 330 rows
    centroid = np.zeros((len(munsell330), 9))
    centroid[:, :munsell330.shape[1]] = munsell330  # All 330 rows
    # Add actual RGB columns (4:6) for all 330
    centroid[:, 4:7] = rgb_from_xyz
    
    colourName = predicted_names  # All 330 predictions
    
    
    # Save for MATLAB
    sio.savemat('results/BritishEnglish_munsell_results.mat', {
        'centroids': rgb_from_xyz,
        'centroid': centroid, 
        'colourName': colourName
    })
    
    logging.info("Results saved to val_results/BritishEnglish_munsell_results.mat")

if __name__ == "__main__":
    logging.info("Training British English model and predicting Munsell colors...")
    
    try:
        train_and_predict_munsell()
        logging.info("Training and prediction completed successfully!")
        
    except Exception as e:
        logging.error(f"Error during training/prediction: {e}")
        raise