import os
import random
import warnings
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tqdm.notebook import tqdm
import cvxopt
warnings.simplefilter("ignore")

def seed(seed=42):
    """
    Set the random seed for reproducibility.
    
    Parameters:
    - seed: int, optional, default is 42
        Random seed value.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)


# ====================
#         Data        
# ====================

def load_data():
    """
    Load training and test set data from CSV files.
    
    Returns:
    - Xtr: numpy.ndarray
        Training set features.
    - Xte: numpy.ndarray
        Test set features.
    - Ytr: numpy.ndarray
        Training set labels.
    """
    Xtr = np.array(pd.read_csv('Xtr.csv', header=None).dropna(axis='columns'))
    Xte = np.array(pd.read_csv('Xte.csv', header=None).dropna(axis='columns'))
    Ytr = np.array(pd.read_csv('Ytr.csv', usecols=[1])).squeeze()
    return Xtr, Xte, Ytr

def plot_images(X, y, n=3):
    """
    Plot sample images from each class.
    
    Parameters:
    - X: numpy.ndarray
        Feature matrix.
    - y: numpy.ndarray
        Labels.
    - n: int, optional, default is 3
        Number of samples to display per class.
    """
    fig, axs = plt.subplots(n, 10, figsize=(18, 6))
    for j in range(10):
        class_indices = np.where(y == j)[0][:n]
        axs[0, j].set_title(f"Class {j}")
        for i, idx in enumerate(class_indices):
            img = X[idx].reshape(32, 32, 3)
            img = ( 255 * (img - np.min(img)) / (img.max() - img.min()) ).astype(np.uint8)
            axs[i, j].imshow(img)
            axs[i, j].axis('off')
    plt.tight_layout()
    plt.show()
    
def z_score(X, epsilon=1e-10):
    """
    Standardize features by removing the mean and scaling to unit variance.
    
    Parameters:
    - X: numpy.ndarray
        Feature matrix.
    - epsilon: float, optional, default is 1e-10
        Small value to avoid division by zero.
    
    Returns:
    - numpy.ndarray
        Standardized features.
    """
    M = np.mean(X, axis=0)
    S = np.std(X, axis=0)
    S_adjusted = np.where(S > epsilon, S, 1)
    return (X - M) / S_adjusted

def plot_features(Xtr_features, Ytr, dim=2):
    """
    Plot sample features from each class.
    
    Parameters:
    - Xtr_features: numpy.ndarray
        Feature matrix.
    - Ytr: numpy.ndarray
        Labels.
    - dim: int, optional, default is 2
        Dimensionality of features.
    """
    fig, axs = plt.subplots(3, 10, figsize=(18, 6))
    for j in range(10):
        class_indices = np.where(Ytr == j)[0][:3]
        axs[0, j].set_title(f"Class {j}")
        for i, idx in enumerate(class_indices):
            img = Xtr_features[idx].reshape(60, 60) if dim==2 else Xtr_features[idx].reshape(60, 60, 3)
            img = ( 255 * (img - np.min(img)) / (img.max() - img.min()) ).astype(np.uint8)
            axs[i, j].imshow(img)
            axs[i, j].axis('off')
    plt.tight_layout()
    plt.show()
    
    
# ====================
#         PCA            
# ====================
    
class PCA:
    def __init__(self, n_components):
        """
        Principal Component Analysis (PCA) for dimensionality reduction.
        
        Parameters:
        - n_components: int
            Number of principal components to retain.
        """
        self.n_components = n_components
        self.components = None
        self.mean = None

    def fit(self, X):
        """
        Fit PCA model to the data.
        
        Parameters:
        - X: numpy.ndarray
            Input data matrix.
        """
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean
        covariance_matrix = np.cov(X_centered, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        self.components = eigenvectors[:, :self.n_components]

    def transform(self, X):
        """
        Transform data into the reduced dimensional space.
        
        Parameters:
        - X: numpy.ndarray
            Input data matrix.
        
        Returns:
        - numpy.ndarray
            Transformed data matrix.
        """
        X_centered = X - self.mean
        return np.dot(X_centered, self.components)
    
def data_pca(Xtr, Xte, n_components=300):
    """
    Perform Principal Component Analysis (PCA) on input features.
    
    Parameters:
    - Xtr: numpy.ndarray
        Training set features.
    - Xte: numpy.ndarray
        Test set features.
    - n_components: int, optional, default is 300
        Number of principal components to retain.
    
    Returns:
    - Xtr_pca: numpy.ndarray
        Transformed training set features.
    - Xte_pca: numpy.ndarray
        Transformed test set features.
    """
    pca = PCA(n_components=n_components)
    pca.fit(Xtr)
    Xtr_pca = pca.transform(Xtr)
    pca.fit(Xte)
    Xte_pca = pca.transform(Xte)
    return Xtr_pca, Xte_pca
    
    
# ====================
#         FE2D            
# ====================
    
class FeatureExtractor2D:
    def __init__(self, num_filters, filter_size):
        """
        Feature extractor for 2D images using convolutional filters and max pooling.
        
        Parameters:
        - num_filters: int
            Number of convolutional filters.
        - filter_size: int
            Size of each convolutional filter.
        """
        self.num_filters = num_filters
        self.filter_size = filter_size
        # Initialize filters with small random values
        self.filters = np.random.randn(num_filters, filter_size, filter_size) / (filter_size * filter_size)
    
    def relu(self, X):
        """
        Rectified Linear Unit (ReLU) activation function.
        
        Parameters:
        - X: numpy.ndarray
            Input matrix.
        
        Returns:
        - numpy.ndarray
            Output matrix after applying ReLU.
        """
        return np.maximum(0, X)
    
    def convolve2d(self, X, filter, stride=1):
        """
        Perform 2D convolution operation on the input matrix using the specified filter.
        
        Parameters:
        - X: numpy.ndarray
            Input matrix.
        - filter: numpy.ndarray
            Convolutional filter.
        - stride: int, optional (default=1)
            Stride value for the convolution operation.
        
        Returns:
        - numpy.ndarray
            Convolved output matrix.
        """
        filter_size = filter.shape[0]
        result = []
        for i in range(0, X.shape[0] - filter_size + 1, stride):
            row = []
            for j in range(0, X.shape[1] - filter_size + 1, stride):
                region = X[i:i+filter_size, j:j+filter_size]
                convolved_value = np.sum(region * filter)
                row.append(convolved_value)
            result.append(row)
        return np.array(result)
    
    def max_pooling(self, X, pool_size=2, stride=2):
        """
        Perform max pooling operation on the input matrix.
        
        Parameters:
        - X: numpy.ndarray
            Input matrix.
        - pool_size: int, optional (default=2)
            Size of the pooling window.
        - stride: int, optional (default=2)
            Stride value for the pooling operation.
        
        Returns:
        - numpy.ndarray
            Max-pooled output matrix.
        """
        pooled_output = []
        for i in range(0, X.shape[0], stride):
            row = []
            for j in range(0, X.shape[1], stride):
                region = X[i:i+pool_size, j:j+pool_size]
                pooled_value = np.max(region)
                row.append(pooled_value)
            pooled_output.append(row)
        return np.array(pooled_output)
    
    def forward_pass(self, X):
        """
        Perform forward pass through the convolutional layers and pooling.
        
        Parameters:
        - X: numpy.ndarray
            Input image matrix.
        
        Returns:
        - numpy.ndarray
            Feature vector extracted from the input image.
        """
        # Reshape X to 32x32x3 to account for RGB channels
        X = X.reshape(32, 32, 3)
        # Convert to grayscale by averaging the channels
        X_gray = np.mean(X, axis=-1)
        conv_outputs = []
        for filter in self.filters:
            conv_output = self.convolve2d(X_gray, filter)
            conv_output = self.relu(conv_output)
            pooled_output = self.max_pooling(conv_output)
            conv_outputs.append(pooled_output.flatten())
        # Concatenate all feature maps
        feature_vector = np.concatenate(conv_outputs, axis=0)
        return feature_vector
    
def data_features(Xtr, Xte, load=True, num_filters=16, filter_size=3):
    """
    Extract features from 2D images using the FeatureExtractor2D class.
    
    Parameters:
    - Xtr: numpy.ndarray
        Training set images.
    - Xte: numpy.ndarray
        Test set images.
    - load: bool, optional (default=True)
        Flag to indicate whether to load pre-extracted features or extract them again.
    - num_filters: int, optional (default=16)
        Number of convolutional filters.
    - filter_size: int, optional (default=3)
        Size of each convolutional filter.
    
    Returns:
    - Xtr_features: numpy.ndarray
        Extracted features from the training set.
    - Xte_features: numpy.ndarray
        Extracted features from the test set.
    """
    if load:
        Xtr_features = np.load('Xtr_features.npy', allow_pickle=True)
        Xte_features = np.load('Xte_features.npy', allow_pickle=True)
    else:
        cnn = FeatureExtractor2D(num_filters=num_filters, filter_size=filter_size)
        Xtr_features = np.array([cnn.forward_pass(img) for img in Xtr])
        Xte_features = np.array([cnn.forward_pass(img) for img in Xte])
        np.save('Xtr_features.npy', Xtr_features)
        np.save('Xte_features.npy', Xte_features)
    return Xtr_features, Xte_features


# ====================
#         FE3D            
# ====================

class FeatureExtractor3D:
    def __init__(self, num_filters=16, filter_size=3):
        """
        Feature extractor for 3D images using convolutional filters and max pooling.
        
        Parameters:
        - num_filters: int, optional (default=16)
            Number of convolutional filters.
        - filter_size: int, optional (default=3)
            Size of each convolutional filter.
        """
        self.num_filters = num_filters
        self.filter_size = filter_size
        # Initialize filters with small random values for each channel
        self.filters = np.random.randn(num_filters, filter_size, filter_size, 3) / (filter_size * filter_size)
    
    def relu(self, X):
        """
        Rectified Linear Unit (ReLU) activation function.
        
        Parameters:
        - X: numpy.ndarray
            Input matrix.
        
        Returns:
        - numpy.ndarray
            Output matrix after applying ReLU.
        """
        return np.maximum(0, X)
    
    def convolve2d(self, X, filter, stride=2):
        """
        Perform 2D convolution operation on the input matrix using the specified filter.
        
        Parameters:
        - X: numpy.ndarray
            Input matrix.
        - filter: numpy.ndarray
            Convolutional filter.
        - stride: int, optional (default=2)
            Stride value for the convolution operation.
        
        Returns:
        - numpy.ndarray
            Convolved output matrix.
        """
        # Adjusted to perform convolution across the depth of the input image
        filter_size = filter.shape[0]
        depth = X.shape[2]
        result = []
        for d in range(depth):
            channel_result = []
            for i in range(0, X.shape[0] - filter_size + 1, stride):
                row = []
                for j in range(0, X.shape[1] - filter_size + 1, stride):
                    region = X[i:i+filter_size, j:j+filter_size, d]
                    convolved_value = np.sum(region * filter[:, :, d])
                    row.append(convolved_value)
                channel_result.append(row)
            result.append(channel_result)
        return np.array(result)  # This will be a 3D array
    
    def max_pooling(self, X, pool_size=2, stride=3):
        """
        Perform max pooling operation on the input matrix.
        
        Parameters:
        - X: numpy.ndarray
            Input matrix.
        - pool_size: int, optional (default=2)
            Size of the pooling window.
        - stride: int, optional (default=3)
            Stride value for the pooling operation.
        
        Returns:
        - numpy.ndarray
            Max-pooled output matrix.
        """
        # Adjusted to perform max pooling across the depth of the input image
        depth = X.shape[0]
        pooled_output = []
        for d in range(depth):
            channel_pooled_output = []
            for i in range(0, X.shape[1] - pool_size + 1, stride):
                row = []
                for j in range(0, X.shape[2] - pool_size + 1, stride):
                    region = X[d, i:i+pool_size, j:j+pool_size]
                    pooled_value = np.max(region)
                    row.append(pooled_value)
                channel_pooled_output.append(row)
            pooled_output.append(channel_pooled_output)
        return np.array(pooled_output)  # This will be a 3D array
    
    def forward_pass(self, X):
        """
        Perform forward pass through the convolutional layers and pooling.
        
        Parameters:
        - X: numpy.ndarray
            Input image matrix.
        
        Returns:
        - numpy.ndarray
            Feature volume extracted from the input image.
        """
        # Assume X is already in the shape of 32x32x3
        conv_outputs = []
        for filter in self.filters:
            conv_output = self.convolve2d(X, filter)
            conv_output = self.relu(conv_output)
            pooled_output = self.max_pooling(conv_output)
            conv_outputs.append(pooled_output)
        
        # Stack along the new axis to create a 4D array: num_filters x height x width x depth
        feature_volume = np.stack(conv_outputs, axis=0)
        return feature_volume

def data_features_3D(Xtr, Xte, load=True, num_filters=16, filter_size=3):
    """
    Extract features from 3D images using the FeatureExtractor3D class.
    
    Parameters:
    - Xtr: numpy.ndarray
        Training set images.
    - Xte: numpy.ndarray
        Test set images.
    - load: bool, optional (default=True)
        Flag to indicate whether to load pre-extracted features or extract them again.
    - num_filters: int, optional (default=16)
        Number of convolutional filters.
    - filter_size: int, optional (default=3)
        Size of each convolutional filter.
    
    Returns:
    - numpy.ndarray
        Extracted features from the training set.
    - numpy.ndarray
        Extracted features from the test set.
    """
    if load:
        Xtr_features = np.load('Xtr_features_3D.npy', allow_pickle=True)
        Xte_features = np.load('Xte_features_3D.npy', allow_pickle=True)
    else:
        cnn = FeatureExtractor3D(num_filters=num_filters, filter_size=filter_size)
        Xtr_features = np.array([cnn.forward_pass(img.reshape(32, 32, 3)).flatten() for img in Xtr])
        Xte_features = np.array([cnn.forward_pass(img.reshape(32, 32, 3)).flatten() for img in Xte])
        np.save('Xtr_features_3D.npy', Xtr_features)
        np.save('Xte_features_3D.npy', Xte_features)
    return Xtr_features, Xte_features
    
    
# ====================
#        Split            
# ====================    

def stratified_train_test_split(X, y, test_size=0.2, random_state=42):
    """
    Perform stratified train-test split on input data.
    
    Parameters:
    - X: numpy.ndarray
        Input features.
    - y: numpy.ndarray
        Target labels.
    - test_size: float or int, optional (default=0.2)
        Size of the test set. If float, should be between 0.0 and 1.0 and represents the proportion of the dataset.
        If int, represents the absolute number of test samples.
    - random_state: int, optional (default=42)
        Random seed for reproducibility.
    
    Returns:
    - X_train: numpy.ndarray
        Features for the training set.
    - X_test: numpy.ndarray
        Features for the test set.
    - y_train: numpy.ndarray
        Labels for the training set.
    - y_test: numpy.ndarray
        Labels for the test set.
    """
    test_size = int(test_size * len(X)) if isinstance(test_size, float) else test_size
    unique_labels, label_counts = np.unique(y, return_counts=True)
    train_indices, test_indices = [], []
    
    for label in unique_labels:
        label_indices = np.where(y == label)[0]
        np.random.seed(random_state)
        np.random.shuffle(label_indices) 
        
        test_samples = int(test_size * (label_counts[label] / len(X)))
        train_indices.extend(label_indices[test_samples:])
        test_indices.extend(label_indices[:test_samples])
        
    np.random.seed(random_state)
    np.random.shuffle(train_indices)
    np.random.shuffle(test_indices)
    
    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]

    return X_train, X_test, y_train, y_test

def check_split(Xtr, Ytr, X_train, Y_train, X_val, Y_val):
    """
    Check if all images from the original dataset are present in either the training or validation set.
    
    Parameters:
    - Xtr: numpy.ndarray
        Original input features.
    - Ytr: numpy.ndarray
        Original target labels.
    - X_train: numpy.ndarray
        Features for the training set.
    - Y_train: numpy.ndarray
        Labels for the training set.
    - X_val: numpy.ndarray
        Features for the validation set.
    - Y_val: numpy.ndarray
        Labels for the validation set.
    
    Returns:
    - bool
        True if all images from the original dataset are present in either the training or validation set, False otherwise.
    """
    train_labels_set = set(tuple(image) for image, label in zip(X_train, Y_train))
    val_labels_set = set(tuple(image) for image, label in zip(X_val, Y_val))
    
    for i, (image, label) in enumerate(zip(Xtr, Ytr)):
        if tuple(image) not in train_labels_set and tuple(image) not in val_labels_set:
            print(f"Image {i} with label {label} not found in train or validation set.")
            return False
    return True


# ====================
#       Kernels       
# ====================

class LinearKernel:
    def __init__(self):
        """
        Initialize the Linear Kernel.
        """
        self.name = 'linear'
    
    def compute(self, X1, X2):
        """
        Compute the linear kernel matrix.
        
        Parameters:
        - X1: numpy.ndarray
            First input feature matrix.
        - X2: numpy.ndarray
            Second input feature matrix.
        
        Returns:
        - numpy.ndarray
            Linear kernel matrix.
        """
        return X1 @ X2.T
    
    def representation_solution(self, data, alpha, X):
        """
        Compute the representation solution using the linear kernel.
        
        Parameters:
        - data: numpy.ndarray
            Data matrix.
        - alpha: numpy.ndarray
            Alpha values.
        - X: numpy.ndarray
            Feature matrix.
        
        Returns:
        - numpy.ndarray
            Representation solution.
        """
        kernel_val = self.compute(data, X)
        return np.dot(kernel_val, alpha)
    
    def distance(self, X1, X2):
        """
        Compute the distance matrix using the linear kernel.
        
        Parameters:
        - X1: numpy.ndarray
            First input feature matrix.
        - X2: numpy.ndarray
            Second input feature matrix.
        
        Returns:
        - numpy.ndarray
            Distance matrix.
        """
        dist_squared = np.diag(self.compute(X1, X1))[:, None] - 2 * self.compute(X1, X2) + np.diag(self.compute(X2, X2))
        return np.sqrt(np.maximum(dist_squared, 0))

class PolynomialKernel:
    def __init__(self, degree=2, coef=0):
        """
        Initialize the Polynomial Kernel.
        
        Parameters:
         - degree: Degree of the polynomial kernel (default is 2).
         - coef: Coefficient to be added to the polynomial kernel (default is 0).
        """
        self.name = 'polynomial'
        self.degree = degree
        self.coef = coef
    
    def compute(self, X1, X2):
        """
        Compute the polynomial kernel matrix.
        
        Parameters:
        - X1: numpy.ndarray
            First input feature matrix.
        - X2: numpy.ndarray
            Second input feature matrix.
        
        Returns:
        - numpy.ndarray
            Polynomial kernel matrix.
        """
        return (X1 @ X2.T + self.coef) ** self.degree
    
    def representation_solution(self, data, alpha, X):
        """
        Compute the representation solution using the polynomial kernel.
        
        Parameters:
        - data: numpy.ndarray
            Data matrix.
        - alpha: numpy.ndarray
            Alpha values.
        - X: numpy.ndarray
            Feature matrix.
        
        Returns:
        - numpy.ndarray
            Representation solution.
        """
        kernel_val = self.compute(data, X)
        return np.dot(kernel_val, alpha)
    
    def distance(self, X1, X2):
        """
        Compute the distance matrix using the polynomial kernel.
        
        Parameters:
        - X1: numpy.ndarray
            First input feature matrix.
        - X2: numpy.ndarray
            Second input feature matrix.
        
        Returns:
        - numpy.ndarray
            Distance matrix.
        """
        dist_squared = np.diag(self.compute(X1, X1))[:, None] - 2 * self.compute(X1, X2) + np.diag(self.compute(X2, X2))
        return np.sqrt(np.maximum(dist_squared, 0))

class GaussianKernel:
    def __init__(self, sigma=1):
        """
        Initialize the Gaussian Kernel.
        
        Parameters:
         - sigma: Standard deviation of the Gaussian kernel (default is 1).
        """
        self.name = 'gaussian'
        self.sigma = sigma
    
    def compute(self, X1, X2):
        """
        Compute the Gaussian kernel matrix.
        
        Parameters:
        - X1: numpy.ndarray
            First input feature matrix.
        - X2: numpy.ndarray
            Second input feature matrix.
        
        Returns:
        - numpy.ndarray
            Gaussian kernel matrix.
        """
        sq_norm = np.sum(X1 ** 2, axis=1)[:, np.newaxis] + np.sum(X2 ** 2, axis=1) - 2 * X1 @ X2.T
        return np.exp(-sq_norm / (2 * self.sigma ** 2))
    
    def representation_solution(self, data, alpha, X):
        """
        Compute the representation solution using the Gaussian kernel.
        
        Parameters:
        - data: numpy.ndarray
            Data matrix.
        - alpha: numpy.ndarray
            Alpha values.
        - X: numpy.ndarray
            Feature matrix.
        
        Returns:
        - numpy.ndarray
            Representation solution.
        """
        kernel_val = self.compute(data, X)
        return np.dot(kernel_val, alpha)
    
    def distance(self, X1, X2):
        """
        Compute the distance matrix using the Gaussian kernel.
        
        Parameters:
        - X1: numpy.ndarray
            First input feature matrix.
        - X2: numpy.ndarray
            Second input feature matrix.
        
        Returns:
        - numpy.ndarray
            Distance matrix.
        """
        dist_squared = np.diag(self.compute(X1, X1))[:, None] - 2 * self.compute(X1, X2) + np.diag(self.compute(X2, X2))
        return np.sqrt(np.maximum(dist_squared, 0))

class SigmoidKernel:
    def __init__(self, coef=0):
        """
        Initialize the Sigmoid Kernel.
        
        Parameters:
         - coef: Coefficient to be added to the sigmoid kernel (default is 0).
        """

        self.name = 'sigmoid'
        self.coef = coef
    
    def compute(self, X1, X2):
        """
        Compute the sigmoid kernel matrix.
        
        Parameters:
        - X1: numpy.ndarray
            First input feature matrix.
        - X2: numpy.ndarray
            Second input feature matrix.
        
        Returns:
        - numpy.ndarray
            Sigmoid kernel matrix.
        """
        return np.tanh(X1 @ X2.T + self.coef)
    
    def representation_solution(self, data, alpha, X):
        """
        Compute the representation solution using the sigmoid kernel.
        
        Parameters:
        - data: numpy.ndarray
            Data matrix.
        - alpha: numpy.ndarray
            Alpha values.
        - X: numpy.ndarray
            Feature matrix.
        
        Returns:
        - numpy.ndarray
            Representation solution.
        """
        kernel_val = self.compute(data, X)
        return np.dot(kernel_val, alpha)
    
    def distance(self, X1, X2):
        """
        Compute the distance matrix using the sigmoid kernel.
        
        Parameters:
        - X1: numpy.ndarray
            First input feature matrix.
        - X2: numpy.ndarray
            Second input feature matrix.
        
        Returns:
        - numpy.ndarray
            Distance matrix.
        """
        dist_squared = np.diag(self.compute(X1, X1))[:, None] - 2 * self.compute(X1, X2) + np.diag(self.compute(X2, X2))
        return np.sqrt(np.maximum(dist_squared, 0))

class GaussianPolynomialKernel:
    def __init__(self, degree=2, coef=0, sigma=1, poly=3):
        """
        Initialize the Gaussian Polynomial Kernel.
        
        Parameters:
         - degree: Degree of the polynomial kernel (default is 2).
         - coef: Coefficient to be added to the polynomial kernel (default is 0).
         - sigma: Standard deviation of the Gaussian kernel (default is 1).
         - poly: Polynomial factor for the combined kernel (default is 3).
        """
        self.name = 'gauss+poly'
        self.degree = degree
        self.coef = coef
        self.sigma = sigma
        self.poly = poly
    
    def compute(self, X1, X2):
        """
        Compute the combined Gaussian and polynomial kernel matrix.
        
        Parameters:
        - X1: numpy.ndarray
            First input feature matrix.
        - X2: numpy.ndarray
            Second input feature matrix.
        
        Returns:
        - numpy.ndarray
            Combined kernel matrix.
        """
        sq_norm = np.sum(X1 ** 2, axis=1)[:, np.newaxis] + np.sum(X2 ** 2, axis=1) - 2 * X1 @ X2.T
        return np.exp(-sq_norm / (2 * self.sigma ** 2)) + self.poly * (X1 @ X2.T + self.coef) ** self.degree
    
    def representation_solution(self, data, alpha, X):
        """
        Compute the solution representation using the combined Gaussian and polynomial kernel.
        
        Parameters:
        - data: numpy.ndarray
            Data matrix.
        - alpha: numpy.ndarray
            Alpha values.
        - X: numpy.ndarray
            Feature matrix.
        
        Returns:
        - numpy.ndarray
            Representation solution.
        """
        kernel_val = self.compute(data, X)
        return np.dot(kernel_val, alpha)
    
    def distance(self, X1, X2):
        """
        Compute the distance matrix using the combined Gaussian and polynomial kernel.
        
        Parameters:
        - X1: numpy.ndarray
            First input feature matrix.
        - X2: numpy.ndarray
            Second input feature matrix.
        
        Returns:
        - numpy.ndarray
            Distance matrix.
        """
        dist_squared = np.diag(self.compute(X1, X1))[:, None] - 2 * self.compute(X1, X2) + np.diag(self.compute(X2, X2))
        return np.sqrt(np.maximum(dist_squared, 0))

    
# ====================
#         KRR            
# ====================

class KernelRidgeRegression:
    def __init__(self, lam, kernel):
        """
        Initialize the Kernel Ridge Regression model.
        
        Parameters:
         - lam: Regularization parameter lambda.
         - kernel: Kernel function used for regression.
        """
        self.lam = lam
        self.alpha = None
        self.kernel = kernel
    
    def fit(self, y, Gram):
        """
        Fit the model to the training data.
        
        Parameters:
         - y: Target labels.
         - Gram: Gram matrix computed from training data.
        """
        n = len(y)
        self.alpha = np.linalg.solve(Gram + n * self.lam * np.eye(n), y)
    
    def predict(self, X_test, X_train):
        """
        Make predictions on new data.
        
        Parameters:
         - X_test: Test data.
         - X_train: Training data.
         
        Returns:
         - Predicted labels.
        """
        return self.kernel.representation_solution(X_test, self.alpha, X_train)
    
    def score(self, y_true, y_pred):
        """
        Calculate the accuracy score.
        
        Parameters:
         - y_true: True labels.
         - y_pred: Predicted labels.
         
        Returns:
         - Accuracy score.
        """
        return np.mean(y_true == y_pred)

def parameters_krr(X_train_features, X_val_features, Y_train_features, Y_val_features, plot=False, data_3D=True):
    """
    Find optimal hyperparameters for Kernel Ridge Regression.
    
    Parameters:
     - X_train_features: Training data features.
     - X_val_features: Validation data features.
     - Y_train_features: Training labels.
     - Y_val_features: Validation labels.
     - plot: Flag to plot the results (default is False).
     - data_3D: Flag indicating whether data is 3D (default is True).
     
    Returns:
     - kernels_krr: List of kernel functions used.
     - best_lambdas: List of optimal regularization parameters.
    """
    lambda_values = np.logspace(-8, -2, 20)
    best_lambdas = []
    kernel_names = ['Linear', 'Polynomial', 'Gaussian', 'Sigmoid', 'Gaussian+Polynomial']
    colors = plt.cm.viridis(np.linspace(0, 1, len(kernel_names)))
    
    kernels_krr = [
        LinearKernel(),
        PolynomialKernel(degree=2, coef=0.124 if data_3D else 2.637e-2),
        GaussianKernel(sigma=0.67 if data_3D else 0.379),
        SigmoidKernel(coef=1.833 if data_3D else 1.387),
        GaussianPolynomialKernel(degree=2, coef=1e-2, sigma=0.66 if data_3D else 0.37, poly=0.4833 if data_3D else 0.25)
    ]
    
    if plot:
        Y_train_ovr = np.array([1 if label == c else -1 for label in Y_train_features for c in range(10)]).reshape(-1, 10)
        plt.figure(figsize=(12, 8))
        for kernel, name, color in tqdm(zip(kernels_krr, kernel_names, colors), total=len(kernels_krr), desc='Kernels', leave=False):
            scores = []
            for lam in tqdm(lambda_values, desc=f'Lambda values for {name} Kernel', leave=False):
                krr = KernelRidgeRegression(lam=lam, kernel=kernel)
                krr.fit(Y_train_ovr, kernel.compute(X_train_features, X_train_features))
                representation_solution = krr.predict(X_val_features, X_train_features)
                score = krr.score(Y_val_features, np.argmax(representation_solution.reshape(-1, 10), axis=1))
                scores.append(score)
            best_lambdas.append(lambda_values[np.argmax(scores)])
            plt.semilogx(lambda_values, scores, marker='o', linestyle='-', color=color, label=name)
        plt.xlabel('Lambda')
        plt.ylabel('Score')
        plt.title('Score vs. Lambda - Kernel Ridge Regression')
        plt.legend()
        plt.grid(True)
        plt.show()
        
        filename = 'best_lambdas_krr_3D.npy' if data_3D else 'best_lambdas_krr.npy'
        np.save(filename, best_lambdas)
    
    else:
        filename = 'best_lambdas_krr_3D.npy' if data_3D else 'best_lambdas_krr.npy'
        best_lambdas = np.load(filename, allow_pickle=True)
    
    return kernels_krr, best_lambdas

    
def application_krr(Xtr_features, Xte_features, Ytr, lam, kernel, data_3D=True):
    """
    Apply Kernel Ridge Regression to predict labels for test data.
    
    Parameters:
     - Xtr_features: Training data features.
     - Xte_features: Test data features.
     - Ytr: Training labels.
     - lam: Regularization parameter lambda.
     - kernel: Kernel function used for regression.
     - data_3D: Flag indicating whether data is 3D (default is True).
    """
    krr = KernelRidgeRegression(lam=lam, kernel=kernel)
    Ytr_ovr = np.array([1 if label == c else -1 for label in Ytr for c in range(10)]).reshape(-1, 10)
    krr.fit(Ytr_ovr, kernel.compute(Xtr_features, Xtr_features))
    representation_solution = krr.predict(Xte_features, Xtr_features)
    predictions = np.argmax(representation_solution.reshape(-1, 10), axis=1)
    dataframe = pd.DataFrame({'Prediction' : predictions}) 
    dataframe.index += 1
    filename = 'Yte_pred_krr_3D.csv' if data_3D else 'Yte_pred_krr.csv'
    dataframe.to_csv(filename, index_label='Id')
    print('Prediction done')
    

# ====================
#         KLR            
# ====================

class KernelLogisticRegression:
    def __init__(self, lam, kernel, Niter=5):
        """
        Initialize the Kernel Logistic Regression model.
        
        Parameters:
         - lam: Regularization parameter lambda.
         - kernel: Kernel function used for logistic regression.
         - Niter: Number of iterations for optimization (default is 5).
        """
        self.lam = lam
        self.Niter = Niter
        self.alpha = None  # This will now be a list of alpha vectors, one for each class
        self.kernel = kernel

    def sigmoid(self, t):
        """
        Sigmoid function.
        
        Parameters:
         - t: Input value.
         
        Returns:
         - Output of the sigmoid function.
        """
        return 1 / (1 + np.exp(-t))

    def fit(self, y, Gram):
        """
        Fit the model to the training data.
        
        Parameters:
         - y: Target labels (one-hot encoded).
         - Gram: Gram matrix computed from training data.
        """
        n, k = y.shape  # n is the number of samples, k is the number of classes
        self.alpha = np.zeros((n, k))  
        for c in range(k):  
            alpha_c = np.zeros(n)  
            y_c = y[:, c]  
            for _ in range(self.Niter):
                m_c = np.dot(Gram, alpha_c)  
                W_c = self.sigmoid(m_c) * (1 - self.sigmoid(m_c)) 
                z_c = m_c + (y_c - self.sigmoid(m_c)) / W_c 
                alpha_c = np.linalg.solve(Gram * W_c[:, np.newaxis] + self.lam * np.eye(n), W_c * z_c)
            self.alpha[:, c] = alpha_c  

    def predict(self, X_test, X_train):
        """
        Make predictions on new data.
        
        Parameters:
         - X_test: Test data.
         - X_train: Training data.
         
        Returns:
         - Predicted probabilities for each class.
        """
        M = np.dot(self.kernel.compute(X_test, X_train), self.alpha)
        return self.sigmoid(M)

    def score(self, y_true, y_pred):
        """
        Calculate the accuracy score.
        
        Parameters:
         - y_true: True labels.
         - y_pred: Predicted probabilities for each class.
         
        Returns:
         - Accuracy score.
        """
        #predictions = np.argmax(y_pred, axis=0)
        #true_labels = np.argmax(y_true, axis=0)
        #return np.mean(true_labels == predictions)
        return np.mean(y_true == y_pred)
    
def parameters_klr(X_train_features, X_val_features, Y_train_features, Y_val_features, plot=False):
    """
    Find optimal hyperparameters for Kernel Logistic Regression.
    
    Parameters:
     - X_train_features: Training data features.
     - X_val_features: Validation data features.
     - Y_train_features: Training labels.
     - Y_val_features: Validation labels.
     - plot: Flag to plot the results (default is False).
     
    Returns:
     - kernels_klr: List of kernel functions used.
     - best_lambdas: List of optimal regularization parameters.
    """
    lambda_values = np.logspace(-5, 2, 20)
    best_lambdas = []
    kernel_names = ['Linear', 'Polynomial', 'Gaussian', 'Sigmoid', 'Gaussian+Polynomial']
    colors = plt.cm.viridis(np.linspace(0, 1, len(kernel_names)))
    kernels_klr = [
        LinearKernel(),
        PolynomialKernel(degree=2, coef=0.124),
        GaussianKernel(sigma=0.67),
        SigmoidKernel(coef=1.833),
        GaussianPolynomialKernel(degree=2, coef=1e-2, sigma=0.66, poly=0.4833)
    ]
    
    if plot:
        plt.figure(figsize=(12, 8))
        for kernel, name, color in tqdm(zip(kernels_klr, kernel_names, colors), total=len(kernels_klr), desc='Kernels', leave=False):
            scores = []  
            for lam in tqdm(lambda_values, desc=f'Lambda values for {name} Kernel', leave=False):
                klr = KernelLogisticRegression(lam=lam, kernel=kernel)
                Y_train_ovr = np.array([1 if label == c else -1 for label in Y_train_features for c in range(10)]).reshape(-1, 10)
                klr.fit(Y_train_ovr, kernel.compute(X_train_features, X_train_features))
                representation_solution = klr.predict(X_val_features, X_train_features)
                score = klr.score(Y_val_features, np.argmax(representation_solution, axis=1)) 
                scores.append(score)
            best_lambdas.append(lambda_values[np.argmax(scores)])
            plt.semilogx(lambda_values, scores, marker='o', linestyle='-', color=color, label=name)
        plt.xlabel('Lambda')
        plt.ylabel('Score')
        plt.title('Score vs. Lambda - Kernel Logistic Regression')
        plt.legend()
        plt.grid(True)
        plt.show()
        np.save('best_lambdas_klr.npy', best_lambdas)
        
    best_lambdas = np.load('best_lambdas_klr.npy', allow_pickle=True)
    
    return kernels_klr, best_lambdas
    
def application_klr(Xtr_features, Xte_features, Ytr, lam, kernel):
    """
    Apply Kernel Logistic Regression to predict labels for test data.
    
    Parameters:
     - Xtr_features: Training data features.
     - Xte_features: Test data features.
     - Ytr: Training labels.
     - lam: Regularization parameter lambda.
     - kernel: Kernel function used for logistic regression.
    """
    klr = KernelLogisticRegression(lam=lam, kernel=kernel)
    Ytr_ovr = np.array([1 if label == c else -1 for label in Ytr for c in range(10)]).reshape(-1, 10)
    klr.fit(Ytr_ovr, kernel.compute(Xtr_features, Xtr_features))
    representation_solution = klr.predict(Xte_features, Xtr_features)
    predictions = np.argmax(representation_solution, axis=1)
    dataframe = pd.DataFrame({'Prediction' : predictions}) 
    dataframe.index += 1
    dataframe.to_csv(f'Yte_pred_klr.csv', index_label='Id')
    print('Prediction done')

    
# ====================
#         KNN            
# ====================

class KernelKNN:
    def __init__(self, kernel, k=5):
        """
        Initialize the Kernel k Nearest Neighbors (KNN) model.
        
        Parameters:
         - kernel: Kernel function used for distance computation.
         - k: Number of neighbors to consider (default is 5).
        """
        self.kernel = kernel
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X_train, y_train):
        """
        Fit the model to the training data.
        
        Parameters:
         - X_train: Training data.
         - y_train: Training labels.
        """
        self.X_train = z_score(X_train)
        self.y_train = y_train

    def predict(self, X_test):
        """
        Make predictions on new data.
        
        Parameters:
         - X_test: Test data.
         
        Returns:
         - Predicted labels for the test data.
        """
        X_test_standardized = z_score(X_test)
        distances = self.kernel.distance(X_test_standardized, self.X_train)
        nn_indices = np.argsort(distances, axis=1)[:, :self.k]
        nn_labels = self.y_train[nn_indices]
        predictions = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=1, arr=nn_labels)
        return predictions

    def score(self, X_test, y_true):
        """
        Calculate the accuracy score.
        
        Parameters:
         - X_test: Test data.
         - y_true: True labels for the test data.
         
        Returns:
         - Accuracy score.
        """
        y_pred = self.predict(X_test)
        return np.mean(y_pred == y_true)

def parameters_knn(Xtr_features, X_train_features, X_val_features, Y_train_features, Y_val_features, plot=False):
    """
    Find optimal hyperparameters for Kernel k Nearest Neighbors (KNN).
    
    Parameters:
     - Xtr_features: Training data features.
     - X_train_features: Training data features (standardized).
     - X_val_features: Validation data features (standardized).
     - Y_train_features: Training labels.
     - Y_val_features: Validation labels.
     - plot: Flag to plot the results (default is False).
     
    Returns:
     - kernels_knn: List of kernel functions used.
     - best_k: List of optimal number of neighbors (k).
    """
    n = int(np.sqrt((Xtr_features.shape)[0]))
    k_values = np.array(range(n))+1
    best_k = []
    kernel_names = ['Linear', 'Polynomial', 'Gaussian', 'Sigmoid', 'Gaussian+Polynomial']
    colors = plt.cm.viridis(np.linspace(0, 1, len(kernel_names)))
    kernels_knn = [
        LinearKernel(),
        PolynomialKernel(degree=2, coef=1),
        GaussianKernel(sigma=18.33),
        SigmoidKernel(coef=18.33),
        GaussianPolynomialKernel(degree=2, coef=1, sigma=18.33, poly=1)
    ]
    
    if plot:
        plt.figure(figsize=(12, 8))
        for kernel, name, color in tqdm(zip(kernels_knn, kernel_names, colors), total=len(kernels_knn), desc='Kernels', leave=False):
            scores = []  
            for k in tqdm(k_values, desc=f'k values for {name} Kernel', leave=False):
                knn = KernelKNN(kernel=kernel, k=k)
                knn.fit(X_train_features, Y_train_features)
                score = knn.score(X_val_features, Y_val_features) 
                scores.append(score)
            best_k.append(k_values[np.argmax(scores)])
            plt.semilogx(k_values, scores, marker='o', linestyle='-', color=color, label=name)
        plt.xlabel('k')
        plt.ylabel('Score')
        plt.title('Score vs. k - Kernel k Nearest Neighbors')
        plt.legend()
        plt.grid(True)
        plt.show()
        np.save('best_k_knn.npy', best_k)
    
    best_k = np.load('best_k_knn.npy', allow_pickle=True)
    
    return kernels_knn, best_k
    
def application_knn(Xtr_features, Xte_features, Ytr, k, kernel):
    """
    Apply Kernel k Nearest Neighbors (KNN) to predict labels for test data.
    
    Parameters:
     - Xtr_features: Training data features.
     - Xte_features: Test data features.
     - Ytr: Training labels.
     - k: Number of neighbors.
     - kernel: Kernel function used for distance computation.
    """
    knn = KernelKNN(kernel=kernel, k=k)
    knn.fit(Xtr_features, Ytr)
    predictions = knn.predict(Xte_features)
    dataframe = pd.DataFrame({'Prediction' : predictions}) 
    dataframe.index += 1
    dataframe.to_csv(f'Yte_pred_knn.csv', index_label='Id')
    print('Prediction done')
    
    
# ====================
#         SVM            
# ====================

class KernelSVM:
    def __init__(self, C=1, kernel=LinearKernel()):
        """
        Initialize the Kernel Support Vector Machine (SVM) model.
        
        Parameters:
         - C: Regularization parameter (default is 1).
         - kernel: Kernel function used for computation (default is LinearKernel()).
        """
        self.C = C
        self.kernel = kernel
        self.nb_classes = None
        
    def fit(self, X, y):
        """
        Fit the model to the training data.
        
        Parameters:
         - X: Training data.
         - y: Training labels.
        """
        self.X = X
        self.a = {}
        self.nb_classes = len(np.unique(y))
        
        n = y.shape[0]
        Gram = self.kernel.compute(self.X, self.X)
        
        for idx in tqdm(range(self.nb_classes), desc='Training', leave=False):
            d = (2 * (y == idx) - 1).astype(np.float64)
            P = cvxopt.matrix(Gram)
            q = cvxopt.matrix(-d, tc='d')
            G = cvxopt.matrix(np.vstack((np.diag(-1*d), np.diag(d))))
            h = cvxopt.matrix(np.vstack((np.zeros((n,1)), self.C * np.ones((n,1)))))
            cvxopt.solvers.options['show_progress'] = False
            solution = cvxopt.solvers.qp(P, q, G, h)
            self.a[idx] = np.ravel(solution['x'])
    
    def predict(self, X_test):
        """
        Make predictions on new data.
        
        Parameters:
         - X_test: Test data.
         
        Returns:
         - Predicted labels for the test data.
        """
        self.X_test = X_test
        n_test = np.shape(self.X_test)[0]
        self.y_test = np.zeros([self.nb_classes, n_test])
        n_train = np.shape(self.X)[0]
        Gram_test = np.zeros([n_train, n_test])
        Gram_test = self.kernel.compute(self.X, self.X_test) 
        for idx in tqdm(range(self.nb_classes), desc='Prediction', leave=False):
            self.y_test[idx, :] = np.dot(self.a[idx], Gram_test)
        y_pred = np.argmax(self.y_test, axis=0)
        return y_pred
            
    def score(self, X, y):
        """
        Calculate the accuracy score.
        
        Parameters:
         - X: Data to score.
         - y: True labels.
         
        Returns:
         - Accuracy score.
        """
        y_pred = self.predict(X)
        return np.mean(y_pred == y)

def parameters_svm(X_train, X_val, y_train, y_val, C_values=np.logspace(0, 0.7, 10), kernel=LinearKernel(), plot=False):
    """
    Find optimal hyperparameters for Kernel Support Vector Machine (SVM).
    
    Parameters:
     - X_train: Training data.
     - X_val: Validation data.
     - y_train: Training labels.
     - y_val: Validation labels.
     - C_values: List of regularization parameter values (default is np.logspace(-1, 1, 10)).
     - kernel: Kernel function used for computation (default is LinearKernel()).
     - plot: Flag to plot the results (default is False).
     
    Returns:
     - best_C: Optimal regularization parameter.
    """
    best_C = None

    if plot:
        best_score = -np.inf
        scores = []

        for C in tqdm(C_values, desc='C values', leave=False):
            svm = KernelSVM(C=C, kernel=kernel)
            svm.fit(X_train, y_train)
            score = svm.score(X_val, y_val)
            scores.append(score)
            if score > best_score:
                best_score = score
                best_C = C
                
        plt.figure(figsize=(12, 8))
        plt.semilogx(C_values, scores, marker='o', linestyle='-', label='Validation Score')
        plt.xlabel('C')
        plt.ylabel('Score')
        plt.title('Score vs. C for SVM')
        plt.legend()
        plt.grid(True)
        plt.show()
        np.save('best_C_svm.npy', best_C)        

    best_C = float(np.load('best_C_svm.npy', allow_pickle=True)) 
    return best_C

def application_svm(X_train, X_test, y_train, C, kernel):
    """
    Apply Kernel Support Vector Machine (SVM) to predict labels for test data.
    
    Parameters:
     - X_train: Training data.
     - X_test: Test data.
     - y_train: Training labels.
     - C: Regularization parameter.
     - kernel: Kernel function used for computation.
    """
    svm = KernelSVM(C=C, kernel=kernel)
    svm.fit(X_train, y_train)
    predictions = svm.predict(X_test)
    dataframe = pd.DataFrame({'Prediction' : predictions}) 
    dataframe.index += 1
    dataframe.to_csv(f'Yte_pred_svm.csv', index_label='Id')
    print('Prediction done')