import numpy as np
from src.core.network import forward, backward, computing_cost
from src.core.regularization import Regularizer, Dropout
from src.optimizers import get_optimizer
from src.evaluation.metrics import Metrics
from src.utils.data_loader import DataLoader


class Trainer:
    """
    Neural network trainer with logging and metrics tracking.
    """
    
    def __init__(self, model, logger=None):
        """
        Initialize trainer.
        
        Args:
            model: Model dictionary with parameters
            logger: Logger instance for tracking metrics
        """
        self.model = model
        self.logger = logger
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": [],
            "epochs": []
        }
    
    def train(self, X_train, Y_train, X_val, Y_val, config, architecture):
        """
        Train the neural network with mini-batch gradient descent.
        
        Each epoch shuffles the training set and makes one optimizer step per
        mini-batch. Dropout (if enabled) and the regularization gradient are
        applied on every step. At the end of each epoch, loss and accuracy are
        measured on the full training and validation sets without dropout, so
        the two are directly comparable. Reported losses are cross-entropy only
        (no penalty term), the same quantity evaluate() reports for the test set.
        
        Args:
            X_train: Training data
            Y_train: Training labels (one-hot encoded)
            X_val: Validation data
            Y_val: Validation labels (one-hot encoded)
            config: Configuration dictionary
            architecture: Network architecture list
        
        Returns:
            Trained model (parameters restored to the best validation epoch)
        """
        lr = config["learning_rate"]
        epochs = config["epochs"]
        batch_size = config.get("batch_size", 64)
        activation = config.get("activation", "relu")
        
        # Get optimizer
        optimizer_type = config.get("optimizer", "sgd")
        optimizer_params = config.get("optimizer_params", {})
        optimizer = get_optimizer(optimizer_type, learning_rate=lr, **optimizer_params)
        
        # Regularization settings
        reg_type = config.get("regularization_type", "none")
        lambda_reg = config.get("lambda_reg", 0.0)
        l1_ratio = config.get("l1_ratio", 0.5)
        
        # Dropout settings: one dropout rate per epoch
        if config.get("use_dropout", False):
            dropout_schedule = Dropout.create_dropout_schedule(
                initial_rate=config.get("dropout_rate", 0.5),
                epochs=epochs,
                schedule_type=config.get("dropout_schedule", "constant")
            )
        else:
            dropout_schedule = [0.0] * epochs
        
        best_val_acc = 0
        best_params = None
        
        for epoch in range(epochs):
            dropout = Dropout(dropout_schedule[epoch])
            
            for X_batch, Y_batch in DataLoader.iterate_batches(X_train, Y_train, batch_size, shuffle=True):
                # Forward pass (with dropout)
                A_batch, cache = forward(X_batch, self.model["parameters"], architecture,
                                         activation, dropout=dropout)
                
                # Backward pass
                grads = backward(X_batch, Y_batch, A_batch, cache, architecture, activation)
                
                # Add regularization gradients
                reg_grads = Regularizer.gradient(reg_type, self.model["parameters"], lambda_reg,
                                                 m=X_batch.shape[1], l1_ratio=l1_ratio)
                for key in reg_grads:
                    grads[key] += reg_grads[key]
                
                # Update parameters using optimizer
                self.model["parameters"] = optimizer.update(self.model["parameters"], grads)
            
            # Epoch metrics in inference mode (no dropout)
            train_loss, train_acc = self._loss_and_accuracy(X_train, Y_train, architecture, activation)
            val_loss, val_acc = self._loss_and_accuracy(X_val, Y_val, architecture, activation)
            
            # Store history
            self.history["epochs"].append(epoch)
            self.history["train_loss"].append(float(train_loss))
            self.history["val_loss"].append(float(val_loss))
            self.history["train_acc"].append(float(train_acc))
            self.history["val_acc"].append(float(val_acc))
            
            # Log metrics
            if self.logger:
                metrics_dict = {
                    "epoch": epoch,
                    "train_loss": float(train_loss),
                    "val_loss": float(val_loss),
                    "train_acc": float(train_acc),
                    "val_acc": float(val_acc)
                }
                self.logger.log_metrics(metrics_dict)
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_params = {k: v.copy() for k, v in self.model["parameters"].items()}
            
            # Print progress
            print(f"Epoch {epoch:3d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                  f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")
        
        # Restore best model
        if best_params:
            self.model["parameters"] = best_params
            self.model["best_val_accuracy"] = best_val_acc
        
        return self.model
    
    def _loss_and_accuracy(self, X, Y, architecture, activation):
        """Cross-entropy loss and accuracy of the current parameters (no dropout)."""
        A, _ = forward(X, self.model["parameters"], architecture, activation)
        y_pred = np.argmax(A, axis=0)
        y_true = np.argmax(Y, axis=0)
        return computing_cost(A, Y), Metrics.accuracy(y_pred, y_true)
    
    def evaluate(self, X, Y, architecture, activation="relu"):
        """
        Evaluate model on data.
        
        Args:
            X: Input data
            Y: Labels (one-hot encoded)
            architecture: Network architecture
            activation: Activation function the model was trained with
        
        Returns:
            Dictionary with evaluation metrics
        """
        A, _ = forward(X, self.model["parameters"], architecture, activation)
        loss = computing_cost(A, Y)
        
        y_pred = np.argmax(A, axis=0)
        y_true = np.argmax(Y, axis=0)
        accuracy = Metrics.accuracy(y_pred, y_true)
        
        # Compute confusion matrix and other metrics
        cm = Metrics.confusion_matrix(y_pred, y_true)
        prf = Metrics.precision_recall_f1(cm)
        
        return {
            "loss": float(loss),
            "accuracy": float(accuracy),
            "y_pred": y_pred,
            "y_true": y_true,
            "confusion_matrix": cm,
            "precision": prf["macro_precision"],
            "recall": prf["macro_recall"],
            "f1": prf["macro_f1"]
        }
    
    def get_history(self):
        """Get training history."""
        return self.history
