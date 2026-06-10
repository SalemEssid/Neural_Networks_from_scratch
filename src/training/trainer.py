import numpy as np
from src.core.network import forward, backward, update_parameters, computing_cost
from src.core.regularization import Regularizer
from src.optimizers import get_optimizer
from src.evaluation.metrics import Metrics


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
        Train the neural network.
        
        Args:
            X_train: Training data
            Y_train: Training labels (one-hot encoded)
            X_val: Validation data
            Y_val: Validation labels (one-hot encoded)
            config: Configuration dictionary
            architecture: Network architecture list
        
        Returns:
            Trained model
        """
        lr = config["learning_rate"]
        epochs = config["epochs"]
        activation = config.get("activation", "relu")
        
        # Get optimizer
        optimizer_type = config.get("optimizer", "sgd")
        optimizer_params = config.get("optimizer_params", {})
        optimizer = get_optimizer(optimizer_type, learning_rate=lr, **optimizer_params)
        
        # Regularization settings
        reg_type = config.get("regularization_type", "none")
        lambda_reg = config.get("lambda_reg", 0.0)
        l1_ratio = config.get("l1_ratio", 0.5)
        
        best_val_acc = 0
        best_params = None
        
        for epoch in range(epochs):
            # Forward pass
            A_train, cache_train = forward(X_train, self.model["parameters"], architecture, activation)
            
            # Compute loss
            data_loss = computing_cost(A_train, Y_train)
            
            # Add regularization to loss
            if reg_type == "l2":
                reg_loss = Regularizer.l2_penalty(self.model["parameters"], lambda_reg)
            elif reg_type == "l1":
                reg_loss = Regularizer.l1_penalty(self.model["parameters"], lambda_reg)
            elif reg_type == "elastic_net":
                reg_loss = Regularizer.elastic_net_penalty(self.model["parameters"], lambda_reg, l1_ratio)
            else:
                reg_loss = 0
            
            train_loss = data_loss + reg_loss
            
            # Backward pass
            grads = backward(X_train, Y_train, A_train, cache_train, architecture, activation)
            
            # Add regularization gradients
            if reg_type == "l2":
                reg_grads = Regularizer.l2_gradient(self.model["parameters"], lambda_reg)
            elif reg_type == "l1":
                reg_grads = Regularizer.l1_gradient(self.model["parameters"], lambda_reg)
            elif reg_type == "elastic_net":
                reg_grads = Regularizer.elastic_net_gradient(self.model["parameters"], lambda_reg, l1_ratio)
            else:
                reg_grads = {}
            
            # Combine gradients
            for key in reg_grads:
                if key in grads:
                    grads[key] += reg_grads[key]
                else:
                    grads[key] = reg_grads[key]
            
            # Update parameters using optimizer
            self.model["parameters"] = optimizer.update(self.model["parameters"], grads)
            
            # Compute training accuracy
            y_pred_train = np.argmax(A_train, axis=0)
            y_true_train = np.argmax(Y_train, axis=0)
            train_acc = Metrics.accuracy(y_pred_train, y_true_train)
            
            # Validation
            A_val, _ = forward(X_val, self.model["parameters"], architecture, activation)
            val_data_loss = computing_cost(A_val, Y_val)
            
            # Add regularization to validation loss
            if reg_type == "l2":
                val_reg_loss = Regularizer.l2_penalty(self.model["parameters"], lambda_reg)
            elif reg_type == "l1":
                val_reg_loss = Regularizer.l1_penalty(self.model["parameters"], lambda_reg)
            elif reg_type == "elastic_net":
                val_reg_loss = Regularizer.elastic_net_penalty(self.model["parameters"], lambda_reg, l1_ratio)
            else:
                val_reg_loss = 0
            
            val_loss = val_data_loss + val_reg_loss
            y_pred_val = np.argmax(A_val, axis=0)
            y_true_val = np.argmax(Y_val, axis=0)
            val_acc = Metrics.accuracy(y_pred_val, y_true_val)
            
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
            if epoch % 10 == 0:
                print(f"Epoch {epoch:3d} | Loss: {train_loss:.4f} | "
                      f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")
        
        # Restore best model
        if best_params:
            self.model["parameters"] = best_params
            self.model["best_val_accuracy"] = best_val_acc
        
        return self.model
    
    def evaluate(self, X, Y, architecture):
        """
        Evaluate model on data.
        
        Args:
            X: Input data
            Y: Labels (one-hot encoded)
            architecture: Network architecture
        
        Returns:
            Dictionary with evaluation metrics
        """
        A, _ = forward(X, self.model["parameters"], architecture)
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


def train(model, X_train, Y_train, X_test, Y_test, lr, epochs):
    """
    Legacy training function for backward compatibility.
    """
    parameters = model["parameters"]
    
    for epoch in range(epochs):
        # TRAIN STEP
        A3, cache = forward(X_train, model["parameters"], [784, 128, 64, 10])
        loss = computing_cost(A3, Y_train)
        
        grads = backward(X_train, Y_train, A3, cache, [784, 128, 64, 10])
        
        model["parameters"] = update_parameters(
            model["parameters"], grads, lr, [784, 128, 64, 10]
        )
        
        # EVALUATION
        A3_train, _ = forward(X_train, model["parameters"], [784, 128, 64, 10])
        train_acc = evaluate(model, X_train, Y_train)
        test_acc = evaluate(model, X_test, Y_test)
        
        # LOGGING
        if epoch % 10 == 0:
            print(f"epoch {epoch} | loss {loss:.4f} | train {train_acc:.4f} | test {test_acc:.4f}")
        
        model["test_accuracy"] = test_acc
    
    return model


def evaluate(model, X, Y):
    """
    Legacy evaluation function.
    """
    A3, _ = forward(X, model["parameters"], [784, 128, 64, 10])
    
    y_pred = np.argmax(A3, axis=0)
    y_true = np.argmax(Y, axis=0)
    
    return np.mean(y_pred == y_true)