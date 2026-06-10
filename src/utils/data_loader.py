import numpy as np
from keras.datasets import mnist


class DataLoader:
    """
    Utilities for loading and preprocessing datasets.
    """
    
    @staticmethod
    def load_mnist(normalize=True):
        """
        Load MNIST dataset.
        
        Args:
            normalize: Whether to normalize to [0, 1]
        
        Returns:
            Tuple of (X_train, Y_train, X_test, Y_test)
        """
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
        
        if normalize:
            X_train = X_train / 255.0
            X_test = X_test / 255.0
        
        return X_train, y_train, X_test, y_test
    
    @staticmethod
    def flatten_images(X):
        """
        Flatten images to 1D vectors.
        
        Args:
            X: Image array of shape (samples, height, width)
        
        Returns:
            Flattened array of shape (flattened_size, samples)
        """
        samples = X.shape[0]
        X_flat = X.reshape(samples, -1)
        return X_flat.T
    
    @staticmethod
    def one_hot_encode(Y, num_classes=10):
        """
        One-hot encode labels.
        
        Args:
            Y: Label array
            num_classes: Number of classes
        
        Returns:
            One-hot encoded array of shape (num_classes, samples)
        """
        one_hot_Y = np.zeros((num_classes, Y.size))
        one_hot_Y[Y, np.arange(Y.size)] = 1
        return one_hot_Y
    
    @staticmethod
    def train_val_split(X, Y, val_size=0.2):
        """
        Split data into training and validation sets.
        
        Args:
            X: Input data
            Y: Labels
            val_size: Validation set fraction
        
        Returns:
            Tuple of (X_train, Y_train, X_val, Y_val)
        """
        num_samples = X.shape[1]
        num_val = int(num_samples * val_size)
        
        indices = np.random.permutation(num_samples)
        val_indices = indices[:num_val]
        train_indices = indices[num_val:]
        
        X_train = X[:, train_indices]
        Y_train = Y[:, train_indices]
        X_val = X[:, val_indices]
        Y_val = Y[:, val_indices]
        
        return X_train, Y_train, X_val, Y_val
    
    @staticmethod
    def get_batch(X, Y, batch_size, batch_idx):
        """
        Get a batch of data.
        
        Args:
            X: Input data
            Y: Labels
            batch_size: Size of batch
            batch_idx: Batch index
        
        Returns:
            Tuple of (X_batch, Y_batch)
        """
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, X.shape[1])
        
        X_batch = X[:, start_idx:end_idx]
        Y_batch = Y[:, start_idx:end_idx]
        
        return X_batch, Y_batch
    
    @staticmethod
    def create_batches(X, Y, batch_size):
        """
        Create batches of data.
        
        Args:
            X: Input data
            Y: Labels
            batch_size: Size of batches
        
        Returns:
            List of (X_batch, Y_batch) tuples
        """
        num_batches = int(np.ceil(X.shape[1] / batch_size))
        batches = []
        
        for i in range(num_batches):
            X_batch, Y_batch = DataLoader.get_batch(X, Y, batch_size, i)
            batches.append((X_batch, Y_batch))
        
        return batches
