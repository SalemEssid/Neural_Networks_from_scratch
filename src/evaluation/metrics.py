import numpy as np

class Metrics:
    """
    Compute and track various metrics for model evaluation.
    """
    
    @staticmethod
    def accuracy(y_pred, y_true):
        """
        Compute accuracy.
        
        Args:
            y_pred: Predicted class indices
            y_true: True class indices
        
        Returns:
            Accuracy score (0-1)
        """
        return np.mean(y_pred == y_true)
    
    @staticmethod
    def top_k_accuracy(A_output, y_true, k=5):
        """
        Compute top-k accuracy.
        
        Args:
            A_output: Network output (softmax probabilities)
            y_true: True labels (one-hot encoded)
            k: Number of top predictions to consider
        
        Returns:
            Top-k accuracy score
        """
        y_pred_indices = np.argsort(A_output, axis=0)[-k:, :]
        y_true_indices = np.argmax(y_true, axis=0)
        
        matches = np.sum([y_true_indices == y_pred_indices[i] for i in range(k)])
        return matches / y_true.shape[1]
    
    @staticmethod
    def confusion_matrix(y_pred, y_true, num_classes=10):
        """
        Compute confusion matrix.
        
        Args:
            y_pred: Predicted class indices
            y_true: True labels (one-hot encoded or class indices)
            num_classes: Number of classes
        
        Returns:
            Confusion matrix
        """
        # Convert one-hot to indices if needed
        if y_true.ndim > 1 and y_true.shape[0] != len(y_pred):
            y_true_indices = np.argmax(y_true, axis=0)
        else:
            y_true_indices = y_true
        
        # Ensure they're 1D arrays
        y_pred = np.asarray(y_pred).flatten()
        y_true_indices = np.asarray(y_true_indices).flatten()
        
        cm = np.zeros((num_classes, num_classes))
        
        for pred, true in zip(y_pred, y_true_indices):
            cm[int(true), int(pred)] += 1
        
        return cm
    
    @staticmethod
    def precision_recall_f1(cm):
        """
        Compute precision, recall, and F1 score from confusion matrix.
        
        Args:
            cm: Confusion matrix
        
        Returns:
            Dictionary with precision, recall, and f1 scores
        """
        tp = np.diag(cm)
        fp = np.sum(cm, axis=0) - tp
        fn = np.sum(cm, axis=1) - tp
        
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
        
        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "macro_precision": np.mean(precision),
            "macro_recall": np.mean(recall),
            "macro_f1": np.mean(f1)
        }
