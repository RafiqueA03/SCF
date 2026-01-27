import logging
import os
import pickle

import numpy as np
import scipy.io as sio
import colour
from sklearn.preprocessing import StandardScaler
from src.utils.ut import load_language_data

np.random.seed(42) 
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

VIEWING_CONDITIONS = {
    "XYZ_w": np.array([95.047, 100.000, 108.883]),
    "L_A": 64,
    "Y_b": 20,
    "surround": colour.VIEWING_CONDITIONS_CAM16['Average']
}


def load_pretrained_model_and_predict_munsell():
    """Load pre-trained British English model and predict Munsell colors."""
    model_filepath = "models/British_English_CustomRotation.pkl"
    original_data_path = "data/processed_combined_data.csv"
    
    if not os.path.exists(model_filepath):
        raise FileNotFoundError(f"Pre-trained model not found at {model_filepath}")
    
    with open(model_filepath, 'rb') as f:
        model = pickle.load(f)
    
    language = "British English"
    evaluator, lang_info = load_language_data(original_data_path, language)
    
    X_train, y_train = evaluator.prepare_regression_data()
    scaler = StandardScaler()
    scaler.fit(X_train)
    
    color_names = evaluator.color_names
    
    munsell_mat = sio.loadmat('data/Munsell_Array_330.mat')
    munsell330 = munsell_mat['munsell330']
    
    lab_coords = munsell330[:, 6:9]
    xyz_coords = colour.Lab_to_XYZ(lab_coords, illuminant=colour.CCS_ILLUMINANTS['CIE 1931 2 Degree Standard Observer']['D65'])
    xyz_w_scaled = VIEWING_CONDITIONS["XYZ_w"] / 100
    cam16ucs = colour.XYZ_to_CAM16UCS(xyz_coords, XYZ_w=xyz_w_scaled, L_A=VIEWING_CONDITIONS["L_A"], Y_b=VIEWING_CONDITIONS["Y_b"])
    
    test_coords_scaled = scaler.transform(cam16ucs)
    predictions = model.predict(test_coords_scaled)
    
    y_pred = np.maximum(predictions, 0)
    y_pred = y_pred / (np.sum(y_pred, axis=1, keepdims=True) + 1e-10)
    
    predicted_indices = np.argmax(y_pred, axis=1)
    confidences = np.max(y_pred, axis=1)
    predicted_names = [color_names[idx] for idx in predicted_indices]
    display_names = [name.split('_')[-1] for name in predicted_names]
    
    save_matlab_results(display_names, xyz_coords, munsell330)
    save_unique_names_to_file(display_names)
    
    return {
        'predicted_names': predicted_names,
        'display_names': display_names,
        'confidences': confidences,
        'unique_names': np.unique(display_names),
        'num_unique': len(np.unique(display_names)),
        'model_vocabulary_size': len(color_names)
    }


def save_matlab_results(predicted_names, xyz_coords, munsell330):
    """Save results in MATLAB format."""
    rgb_from_xyz = colour.XYZ_to_sRGB(xyz_coords)
    rgb_from_xyz = np.clip(rgb_from_xyz * 255, 0, 255)
    
    centroid = np.zeros((len(munsell330), 9))
    centroid[:, :munsell330.shape[1]] = munsell330
    centroid[:, 4:7] = rgb_from_xyz
    
    os.makedirs('results', exist_ok=True)
    sio.savemat('results/BritishEnglish_munsell_results.mat', {
        'centroids': rgb_from_xyz,
        'centroid': centroid, 
        'colourName': predicted_names
    })


def save_unique_names_to_file(display_names):
    """Save unique predicted color names to text file."""
    unique_names = sorted(np.unique(display_names))
    
    with open('results/munsell_names.txt', 'w') as f:
        f.write("Unique Color Names Predicted for Munsell 330 Chips\n")
        f.write(f"Total unique names: {len(unique_names)}\n\n")
        for name in unique_names:
            f.write(f"{name}\n")


if __name__ == "__main__":
    try:
        results = load_pretrained_model_and_predict_munsell()
        logging.info("Prediction completed successfully!")
        logging.info(f"Summary: {results['num_unique']} unique color names predicted")
        
    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        raise