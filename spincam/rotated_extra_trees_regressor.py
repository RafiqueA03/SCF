import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler


class RotatedExtraTreesRegressor:
    """Custom ExtraTreesRegressor with configurable rotation of 3D decision space."""
    
    def __init__(self, n_estimators=300, max_depth=16, min_samples_split=4, 
                 min_samples_leaf=1, max_features=1.0, random_state=42, n_jobs=-1,
                 rotation_fraction=0.0):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.rotation_fraction = rotation_fraction
        
        self.trees = []
        self.rotation_matrices = []
        self.scaler = StandardScaler()
        
    def generate_rotation_matrix(self, tree_seed):
        """Generate orthogonal 3x3 rotation matrix using Householder QR."""
        np.random.seed(tree_seed)
        A = np.random.randn(3, 3)
        Q, R = np.linalg.qr(A)
        
        for i in range(3):
            if R[i, i] < 0:
                Q[:, i] *= -1
        
        return Q
    
    def fit(self, X, y):
        """Fit trees ensemble with configurable rotation."""
        X_scaled = self.scaler.fit_transform(X)
        self.trees = []
        self.rotation_matrices = []
        
        n_rotated = int(self.n_estimators * self.rotation_fraction)
                
        for tree_idx in range(self.n_estimators):
            tree_seed = self.random_state + tree_idx
            
            if tree_idx < n_rotated:
                rotation_matrix = self.generate_rotation_matrix(tree_seed)
            else:
                rotation_matrix = np.eye(3)
            
            X_rotated = X_scaled @ rotation_matrix
            
            tree = DecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features,
                random_state=tree_seed
            )
            
            tree.fit(X_rotated, y)
            
            self.trees.append(tree)
            self.rotation_matrices.append(rotation_matrix)
    
    def predict(self, X):
        """Predict using ensemble of trees."""
        if not self.trees:
            raise ValueError("Model not fitted yet!")
        
        X_scaled = self.scaler.transform(X)
        all_predictions = []
        
        for tree, rotation_matrix in zip(self.trees, self.rotation_matrices):
            X_rotated = X_scaled @ rotation_matrix
            tree_prediction = tree.predict(X_rotated)
            all_predictions.append(tree_prediction)
        
        all_predictions = np.array(all_predictions)
        ensemble_prediction = np.mean(all_predictions, axis=0)
        
        return ensemble_prediction
