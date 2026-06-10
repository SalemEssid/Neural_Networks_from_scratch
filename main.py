import numpy as np
from src.core.network import initialize_parameter
from src.training.trainer import Trainer
from src.utils.data_loader import DataLoader
from src.utils.logger import ExperimentLogger
from src.utils.experiment_manager import ExperimentManager
from src.utils.visualization import Visualizer
from src.utils.seed import set_seed
from configs.config import CONFIG


def main():
    """
    Main training pipeline with experiment tracking.
    """
    
    # ============================================
    # CUSTOMIZE EXPERIMENT NAME HERE
    # ============================================
    CUSTOM_EXPERIMENT_NAME = "Adam_as_optimizer"
    # Examples:
    # - "baseline_v1"
    # - "high_lr_experiment"
    # - "deep_network_test"
    # - "final_model"
    # ============================================
    
    # Set seed for reproducibility
    set_seed(CONFIG["seed"])
    
    # Initialize experiment manager
    exp_manager = ExperimentManager(root="experiments")
    experiment_paths = exp_manager.create_experiment(
        experiment_name=CUSTOM_EXPERIMENT_NAME.strip()  # Remove any leading/trailing spaces
    )
    
    print(f"Experiment directory: {experiment_paths['experiment_dir']}")
    
    # Save configuration
    exp_manager.save_config(experiment_paths, CONFIG)
    
    # Initialize logger
    logger = ExperimentLogger(experiment_dir=experiment_paths["experiment_dir"])
    logger.log_config(CONFIG)
    
    # Load and preprocess data
    print("Loading MNIST dataset...")
    X_train, y_train, X_test, y_test = DataLoader.load_mnist(normalize=True)
    
    # Flatten images
    X_train = DataLoader.flatten_images(X_train)
    X_test = DataLoader.flatten_images(X_test)
    
    # One-hot encode labels
    Y_train = DataLoader.one_hot_encode(y_train, num_classes=10)
    Y_test = DataLoader.one_hot_encode(y_test, num_classes=10)
    
    # Split training data into train/validation
    X_train, Y_train, X_val, Y_val = DataLoader.train_val_split(
        X_train, Y_train, val_size=0.2
    )
    
    print(f"Training samples: {X_train.shape[1]}")
    print(f"Validation samples: {X_val.shape[1]}")
    print(f"Test samples: {X_test.shape[1]}")
    
    # Initialize network parameters using flexible architecture
    print(f"Network architecture: {CONFIG['architecture']}")
    parameters = initialize_parameter(
        X_train, 
        CONFIG["architecture"], 
        weight_init=CONFIG.get("weight_initialization", "he")
    )
    
    model = {"parameters": parameters}
    
    # Create trainer
    trainer = Trainer(model, logger=logger)
    
    # Train model
    print("\nStarting training...")
    model = trainer.train(
        X_train, Y_train,
        X_val, Y_val,
        CONFIG,
        CONFIG["architecture"]
    )
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_results = trainer.evaluate(X_test, Y_test, CONFIG["architecture"])
    
    # Print results
    print(f"\nTest Accuracy: {test_results['accuracy']:.4f}")
    print(f"Test Loss: {test_results['loss']:.4f}")
    print(f"Test Precision: {test_results['precision']:.4f}")
    print(f"Test Recall: {test_results['recall']:.4f}")
    print(f"Test F1: {test_results['f1']:.4f}")
    
    # Save results
    summary = {
        "test_accuracy": float(test_results["accuracy"]),
        "test_loss": float(test_results["loss"]),
        "test_precision": float(test_results["precision"]),
        "test_recall": float(test_results["recall"]),
        "test_f1": float(test_results["f1"]),
        "best_val_accuracy": float(model.get("best_val_accuracy", 0)),
        "architecture": CONFIG["architecture"],
        "learning_rate": CONFIG["learning_rate"],
        "epochs": CONFIG["epochs"],
        "activation": CONFIG["activation"]
    }
    
    exp_manager.save_summary(experiment_paths, summary)
    
    # Create visualizer
    visualizer = Visualizer(save_dir=experiment_paths["figures_dir"])
    
    # Plot training history
    history = trainer.get_history()
    visualizer.plot_loss_and_accuracy(
        history["train_loss"],
        history["val_loss"],
        history["train_acc"],
        history["val_acc"],
        save_filename="training_metrics.png"
    )
    
    # Plot confusion matrix
    visualizer.plot_confusion_matrix(
        test_results["confusion_matrix"],
        labels=list(range(10)),
        save_filename="confusion_matrix.png"
    )
    
    # Save model weights
    import pickle
    with open(experiment_paths["final_model"], 'wb') as f:
        pickle.dump(model["parameters"], f)
    
    print(f"\nExperiment completed! Results saved to: {experiment_paths['experiment_dir']}")
    print(f"  - Metrics: {experiment_paths['metrics_file']}")
    print(f"  - Config: {experiment_paths['config_file']}")
    print(f"  - Summary: {experiment_paths['summary_file']}")
    print(f"  - Figures: {experiment_paths['figures_dir']}")
    print(f"  - Model: {experiment_paths['final_model']}")


if __name__ == "__main__":
    main()










# To save the experiments
import json

experiment_name = "experiment"

with open(
    f"logs/{experiment_name}_config.json", "w"
) as f:
    json.dump(CONFIG, f, indent=4)