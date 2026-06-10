import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from sklearn.metrics import confusion_matrix


class Visualizer:
    """
    Visualization utilities for training results and metrics.
    """
    
    def __init__(self, save_dir=None):
        """
        Initialize visualizer.
        
        Args:
            save_dir: Directory to save plots (if None, only displays)
        """
        self.save_dir = Path(save_dir) if save_dir else None
    
    def plot_loss(self, train_loss, val_loss, save_filename="loss_curve.png"):
        """
        Plot training and validation loss.
        
        Args:
            train_loss: List of training losses
            val_loss: List of validation losses
            save_filename: Filename to save plot
        """
        plt.figure(figsize=(10, 6))
        
        plt.plot(train_loss, label="Train Loss", linewidth=2)
        plt.plot(val_loss, label="Validation Loss", linewidth=2)
        
        plt.title("Training and Validation Loss", fontsize=14, fontweight='bold')
        plt.xlabel("Epoch", fontsize=12)
        plt.ylabel("Loss", fontsize=12)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        
        if self.save_dir:
            save_path = self.save_dir / save_filename
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Loss plot saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_accuracy(self, train_acc, val_acc, save_filename="accuracy_curve.png"):
        """
        Plot training and validation accuracy.
        
        Args:
            train_acc: List of training accuracies
            val_acc: List of validation accuracies
            save_filename: Filename to save plot
        """
        plt.figure(figsize=(10, 6))
        
        plt.plot(train_acc, label="Train Accuracy", linewidth=2)
        plt.plot(val_acc, label="Validation Accuracy", linewidth=2)
        
        plt.title("Training and Validation Accuracy", fontsize=14, fontweight='bold')
        plt.xlabel("Epoch", fontsize=12)
        plt.ylabel("Accuracy", fontsize=12)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        
        if self.save_dir:
            save_path = self.save_dir / save_filename
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Accuracy plot saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_loss_and_accuracy(self, train_loss, val_loss, train_acc, val_acc, 
                               save_filename="training_metrics.png"):
        """
        Plot loss and accuracy in subplots.
        
        Args:
            train_loss: List of training losses
            val_loss: List of validation losses
            train_acc: List of training accuracies
            val_acc: List of validation accuracies
            save_filename: Filename to save plot
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Loss plot
        ax1.plot(train_loss, label="Train Loss", linewidth=2)
        ax1.plot(val_loss, label="Validation Loss", linewidth=2)
        ax1.set_title("Loss", fontsize=12, fontweight='bold')
        ax1.set_xlabel("Epoch", fontsize=11)
        ax1.set_ylabel("Loss", fontsize=11)
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # Accuracy plot
        ax2.plot(train_acc, label="Train Accuracy", linewidth=2)
        ax2.plot(val_acc, label="Validation Accuracy", linewidth=2)
        ax2.set_title("Accuracy", fontsize=12, fontweight='bold')
        ax2.set_xlabel("Epoch", fontsize=11)
        ax2.set_ylabel("Accuracy", fontsize=11)
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if self.save_dir:
            save_path = self.save_dir / save_filename
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Combined plot saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_confusion_matrix(self, cm, labels=None, save_filename="confusion_matrix.png"):
        """
        Plot confusion matrix.
        
        Args:
            cm: Confusion matrix (numpy array)
            labels: Class labels
            save_filename: Filename to save plot
        """
        plt.figure(figsize=(10, 8))
        
        im = plt.imshow(cm, cmap='Blues', aspect='auto')
        
        plt.title("Confusion Matrix", fontsize=14, fontweight='bold')
        plt.xlabel("Predicted", fontsize=12)
        plt.ylabel("True", fontsize=12)
        plt.colorbar(im)
        
        if labels:
            plt.xticks(range(len(labels)), labels, rotation=45)
            plt.yticks(range(len(labels)), labels)
        
        # Add text annotations
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, f'{int(cm[i, j])}', ha='center', va='center',
                        color='white' if cm[i, j] > cm.max() / 2 else 'black')
        
        plt.tight_layout()
        
        if self.save_dir:
            save_path = self.save_dir / save_filename
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Confusion matrix saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_weight_distribution(self, weights, save_filename="weight_distribution.png"):
        """
        Plot weight distribution.
        
        Args:
            weights: Weight matrix or list of weights
            save_filename: Filename to save plot
        """
        plt.figure(figsize=(10, 6))
        
        if isinstance(weights, dict):
            # Multiple weight matrices
            for name, w in weights.items():
                plt.hist(w.flatten(), bins=50, alpha=0.6, label=name)
        else:
            plt.hist(weights.flatten(), bins=50, alpha=0.7)
        
        plt.title("Weight Distribution", fontsize=14, fontweight='bold')
        plt.xlabel("Weight Value", fontsize=12)
        plt.ylabel("Frequency", fontsize=12)
        if isinstance(weights, dict):
            plt.legend()
        plt.grid(True, alpha=0.3)
        
        if self.save_dir:
            save_path = self.save_dir / save_filename
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Weight distribution saved to {save_path}")
        else:
            plt.show()
        
        plt.close()


# Legacy functions for backward compatibility
def plot_loss(train_loss, val_loss):
    plt.figure()
    plt.plot(train_loss, label="Train")
    plt.plot(val_loss, label="Validation")
    plt.title("Training and Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()
    plt.close()

def plot_accuracy(train_acc, val_acc):
    plt.figure()
    plt.plot(train_acc, label="Train")
    plt.plot(val_acc, label="Validation")
    plt.title("Training and Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()
    plt.close()

def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure()
    plt.imshow(cm)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.colorbar()
    plt.show()
    plt.close()

def plot_weight_distribution(weights):
    plt.figure()
    plt.hist(weights.flatten(), bins=50)
    plt.title("Weight Distribution")
    plt.xlabel("Weight Value")
    plt.ylabel("Frequency")
    plt.show()
    plt.close()
