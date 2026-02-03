import numpy as np
import pandas as pd
import optuna
import os
import pickle
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from spincam.rotated_trees_regressor import RotatedTreesRegressor
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

class LanguageModelEvaluator:
    def __init__(self, test_grid_csv=None):
        """Initialize the language-specific model evaluator."""
        self.test_grid_csv = test_grid_csv
        self.data = None
        self.color_names = None
        self.unique_colors = None
        self.scaler = StandardScaler()
    
    def load_and_process_language_data(self, csv_path, language_name, train_data_path=None, use_language_filter=True):
        """Load and process data for a specific language."""
        
        full_data = pd.read_csv(csv_path, encoding='utf-8-sig', low_memory=False)
        
        # Only load train_data if path is provided
        if train_data_path is not None:
            train_data = pd.read_csv(train_data_path)
        else:
            train_data = None
        
        # Filter for specific language (skip if using language-specific file)
        if use_language_filter:
            self.data = full_data[full_data['language'] == language_name].copy()
        else:
            self.data = full_data.copy()
        
        if train_data is not None:
            cam16_data = train_data[['color_id', 'CAM16_J_UCS', 'CAM16_a_UCS', 'CAM16_b_UCS']].copy()
            
            # Rename columns to match original function's expected names
            cam16_data = cam16_data.rename(columns={
                'CAM16_J_UCS': 'J_lightness',
                'CAM16_a_UCS': 'a_prime', 
                'CAM16_b_UCS': 'b_prime'
            })
            
            # Merge with main data on color_id
            self.data = self.data.merge(cam16_data, on='color_id', how='left')
            # save self.data to a CSV file for debugging
            lang_file_name = language_name.replace(' ', '_')
            self.data.to_csv(f"data/{lang_file_name}_processed.csv", index=False)
            
            # Check if merge was successful
            missing_cam16 = self.data[['J_lightness', 'a_prime', 'b_prime']].isnull().any(axis=1).sum()
            if missing_cam16 > 0:
                logging.warning(f"Warning: {missing_cam16} rows missing CAM16-UCS values after merge")
            else:
                logging.info("Successfully merged CAM16-UCS coordinates from train_data")
        else:
            # Check if CAM16-UCS coordinates already exist in the data
            required_cols = ['J_lightness', 'a_prime', 'b_prime']
            missing_cols = [col for col in required_cols if col not in self.data.columns]
            
            if missing_cols:
                logging.warning(f"Missing CAM16-UCS coordinates and no train_data provided: {missing_cols}")
        
        # Get unique color names count
        unique_color_names_count = self.data['color_name'].nunique()
        logging.info(f"Unique color names in {language_name}: {unique_color_names_count}")
        
        # Create unique_colors based on existing color_id and merged CAM16-UCS values
        self.unique_colors = self.data.groupby(['color_id','J_lightness','a_prime','b_prime']).size().reset_index().rename(columns={0:'count'})
        
        # Get sorted color names
        sData = self.data.sort_values(by='color_name')
        self.color_names = np.sort(sData['color_name'].unique())
        
        # Compute conditional probabilities P(W|c)
        Count_WC = pd.crosstab(sData['color_name'], sData['color_id'])
        Count_C = Count_WC.sum(axis=0)
        ones_names = np.ones(len(self.color_names))
        PWc = Count_WC / np.outer(ones_names, Count_C)
        
        # Create color_id to P(W|c) mapping
        self.color_id_to_pwc = {}
        for i, color_id in enumerate(Count_WC.columns):
            self.color_id_to_pwc[color_id] = PWc.iloc[:, i].values
            
        return {
            'language': language_name,
            'samples': len(self.data),
            'unique_color_names': unique_color_names_count,
            'unique_color_coordinates': len(self.unique_colors)
        }
        
    def prepare_regression_data(self):
        """Prepare regression data."""
        X = self.unique_colors[['J_lightness', 'a_prime', 'b_prime']].values
        y = np.zeros((len(self.unique_colors), len(self.color_names)))
        
        for i, color_id in enumerate(self.unique_colors['color_id']):
            if color_id in self.color_id_to_pwc:
                y[i, :] = self.color_id_to_pwc[color_id]
        
        return X, y
    
    def bhattacharyya_coefficient(self, p1, p2):
        """Calculate Bhattacharyya coefficient."""
        return np.sum(np.sqrt(np.multiply(p1, p2)))
    
    def compute_metrics(self, y_true, y_pred):
        """Compute peak accuracy, bhattacharyya distance, and other metrics."""
        # Get overall frequencies for tie-breaking
        overall_frequencies = pd.Series(self.data['color_name']).value_counts()
        freq_array = np.zeros(len(self.color_names))
        for i, name in enumerate(self.color_names):
            if name in overall_frequencies.index:
                freq_array[i] = overall_frequencies[name]
        
        # Peak accuracy with frequency-based tie breaking
        correct_predictions = 0
        for i in range(len(y_true)):
            obs_peak_indices = np.where(y_true[i, :] == np.max(y_true[i, :]))[0]
            pred_peak_indices = np.where(y_pred[i, :] == np.max(y_pred[i, :]))[0]
            
            if len(obs_peak_indices) > 1:
                obs_peak = max(obs_peak_indices, key=lambda x: freq_array[x])
            else:
                obs_peak = obs_peak_indices[0]
                
            if len(pred_peak_indices) > 1:
                pred_peak = max(pred_peak_indices, key=lambda x: freq_array[x])
            else:
                pred_peak = pred_peak_indices[0]
            
            if obs_peak == pred_peak:
                correct_predictions += 1
        
        peak_accuracy = correct_predictions / len(y_true)
        
        # Bhattacharyya coefficient
        bhatt_scores = []
        for i in range(len(y_true)):
            bhatt = self.bhattacharyya_coefficient(y_true[i, :], y_pred[i, :])
            bhatt_scores.append(bhatt)
        
        return {
            'peak_accuracy': peak_accuracy,
            'bhattacharyya': np.mean(bhatt_scores)
        }
    
    def evaluate_with_cv(self, model, X, y, random_seed=42):
        """3-fold cross-validation evaluation."""
        kfold = KFold(n_splits=3, shuffle=True, random_state=random_seed)
        fold_results = []
        
        for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(X)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            y_pred = np.maximum(y_pred, 0)
            y_pred = y_pred / (np.sum(y_pred, axis=1, keepdims=True) + 1e-10)
            
            fold_metrics = self.compute_metrics(y_val, y_pred)
            fold_results.append(fold_metrics)
        
        avg_results = {}
        for metric in fold_results[0].keys():
            values = [fold[metric] for fold in fold_results]
            avg_results[f'cv_{metric}'] = np.mean(values)
            avg_results[f'cv_{metric}_std'] = np.std(values)
        
        return avg_results
    
    def count_valid_names_on_test_grid(self, model, X_train, y_train):
        """Count valid names (appearing >=4 times) on test grid."""
        if self.test_grid_csv is None:
            return 0, []
            
        try:
            test_grid = pd.read_csv(self.test_grid_csv)
            cam16_grid = test_grid.iloc[:, 1:4].values
            
            model.fit(X_train, y_train)
            
            cam16_grid_scaled = self.scaler.transform(cam16_grid)
            y_pred = model.predict(cam16_grid_scaled)
            y_pred = np.maximum(y_pred, 0)
            y_pred = y_pred / (np.sum(y_pred, axis=1, keepdims=True) + 1e-10)
            
            peak_predictions = np.argmax(y_pred, axis=1)
            unique_predicted, counts = np.unique(peak_predictions, return_counts=True)
            
            valid_mask = counts >= 4
            valid_names_indices = unique_predicted[valid_mask]
            valid_names_count = np.sum(valid_mask)
            
            # Get the actual color names that appear >=4 times
            valid_color_names = [self.color_names[idx] for idx in valid_names_indices if idx < len(self.color_names)]
            
            return valid_names_count, valid_color_names
            
        except Exception as e:
            print(f"Error in test grid evaluation: {e}")
            return 0, []
    
    def compare_models_for_language(self, X, y, language_name, n_trials=10, random_seed=42):
        """Compare all tree-based models for a specific language."""
        logging.info(f"\n\n Comparing Models for {language_name}...\n")
        
        X_scaled = self.scaler.fit_transform(X)
        
        models = {
            'Extra Trees': {
                'type': 'sklearn',
                'class': ExtraTreesRegressor
            },
            'Random Forest': {
                'type': 'sklearn',
                'class': RandomForestRegressor
            },
            'Custom Unrotated': {
                'type': 'custom',
                'class': RotatedTreesRegressor,
                'rotation_fraction': 0.0
            },
            'Custom Rotation': {
                'type': 'custom',
                'class': RotatedTreesRegressor,
                'rotation_fraction': 0.667
            }
        }
        
        results = {}
        
        for model_name, model_config in models.items():
            print(f"\n Testing {model_name}...")
            
            def objective(trial):
                n_estimators = trial.suggest_int('n_estimators', 200, 400)
                max_depth = trial.suggest_int('max_depth', 10, 25)
                min_samples_split = trial.suggest_int('min_samples_split', 2, 8)
                min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 4)
                max_features = trial.suggest_float('max_features', 0.6, 1.0)
                
                params = {
                    'n_estimators': n_estimators,
                    'max_depth': max_depth,
                    'min_samples_split': min_samples_split,
                    'min_samples_leaf': min_samples_leaf,
                    'max_features': max_features,
                    'random_state': random_seed,
                    'n_jobs': -1
                }
                
                if model_config['type'] == 'sklearn':
                    model = model_config['class'](**params)
                else:
                    if model_name == 'Custom Rotation':
                        rotation_fraction = trial.suggest_float('rotation_fraction', 0.3, 0.9)
                    else:
                        rotation_fraction = model_config['rotation_fraction']
                    
                    params['rotation_fraction'] = rotation_fraction
                    model = model_config['class'](**params)
                
                cv_results = self.evaluate_with_cv(model, X_scaled, y, random_seed)
                combined_score = cv_results['cv_peak_accuracy'] + cv_results['cv_bhattacharyya']
                return -combined_score
            
            # Optimize hyperparameters
            study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=random_seed))
            study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
            
            best_params = study.best_params
            
            # Train final model and evaluate
            if model_config['type'] == 'sklearn':
                final_model = model_config['class'](
                    **{k: v for k, v in best_params.items() if k != 'rotation_fraction'},
                    random_state=random_seed,
                    n_jobs=-1
                )
            else:
                if model_name == 'Custom Rotation':
                    final_rotation_fraction = best_params.get('rotation_fraction', 0.667)
                else:
                    final_rotation_fraction = model_config['rotation_fraction']
                
                final_model = model_config['class'](
                    **{k: v for k, v in best_params.items() if k != 'rotation_fraction'},
                    rotation_fraction=final_rotation_fraction,
                    random_state=random_seed,
                    n_jobs=-1
                )
            
            cv_results = self.evaluate_with_cv(final_model, X_scaled, y, random_seed)
            valid_names_count, valid_names_list = self.count_valid_names_on_test_grid(final_model, X_scaled, y)
            
            results[model_name] = {
                'language': language_name,
                'model_name': model_name,
                'model_type': model_config['type'],
                'best_params': best_params,
                'cv_peak_accuracy': cv_results['cv_peak_accuracy'],
                'cv_peak_accuracy_std': cv_results['cv_peak_accuracy_std'],
                'cv_bhattacharyya': cv_results['cv_bhattacharyya'],
                'cv_bhattacharyya_std': cv_results['cv_bhattacharyya_std'],
                'combined_score': cv_results['cv_peak_accuracy'] + cv_results['cv_bhattacharyya'],
                'valid_names_count': valid_names_count,
                'valid_names_list': valid_names_list,
                'unique_color_names_in_data': len(self.color_names),
                'coverage_percentage': (valid_names_count / len(self.color_names)) * 100 if len(self.color_names) > 0 else 0,
                'rotation_fraction': best_params.get('rotation_fraction', 
                                                  model_config.get('rotation_fraction', 0.0)) if model_config['type'] == 'custom' else None
            }
        
        # Find best model based on combined_score
        best_model_name = max(results.keys(), key=lambda x: results[x]['combined_score'])
        best_model_info = results[best_model_name]
        
        # Create models directory if it doesn't exist
        os.makedirs('models', exist_ok=True)
        
        # Recreate the best model with its optimal parameters
        best_params = best_model_info['best_params']
        model_config = models[best_model_name]
        
        if model_config['type'] == 'sklearn':
            best_model = model_config['class'](
                **{k: v for k, v in best_params.items() if k != 'rotation_fraction'},
                random_state=random_seed,
                n_jobs=-1
            )
        else:
            rotation_fraction = best_params.get('rotation_fraction', model_config.get('rotation_fraction', 0.0))
            best_model = model_config['class'](
                **{k: v for k, v in best_params.items() if k != 'rotation_fraction'},
                rotation_fraction=rotation_fraction,
                random_state=random_seed,
                n_jobs=-1
            )
        
        # Train the best model on full dataset
        best_model.fit(X_scaled, y)
        
        # Save the model
        model_filename = f"models/{language_name.replace(' ', '_')}_{best_model_name.replace(' ', '')}.pkl"
        with open(model_filename, 'wb') as f:
            pickle.dump(best_model, f)
        
        return results
