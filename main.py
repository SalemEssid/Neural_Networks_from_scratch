import pickle
import time
import numpy as np
from src.core.network import initialize_parameter
from src.training.trainer import Trainer
from src.utils.data_loader import DataLoader
from src.utils.logger import ExperimentLogger
from src.utils.experiment_manager import ExperimentManager
from src.utils.visualization import Visualizer
from src.utils.seed import set_seed
from configs.config import CONFIG


# ============================================
# CUSTOMIZE EXPERIMENT NAME HERE
# ============================================
CUSTOM_EXPERIMENT_NAME = "baseline"
# Examples:
# - "baseline_v1"
# - "high_lr_experiment"
# - "deep_network_test"
# - "final_model"
# ============================================


def main(config=CONFIG, experiment_name=CUSTOM_EXPERIMENT_NAME):
    """
    Main training pipeline with experiment tracking.
    
    Args:
        config: Configuration dictionary (defaults to configs/config.py)
        experiment_name: Folder name under experiments/ (reused names are overwritten)
    
    Returns:
        Summary dictionary that is also saved to summary.json
    """
    
    # Set seed for reproducibility
    set_seed(config["seed"])
    
    # Initialize experiment manager
    exp_manager = ExperimentManager(root="experiments")
    experiment_paths = exp_manager.create_experiment(
        experiment_name=experiment_name.strip()  # Remove any leading/trailing spaces
    )
    
    print(f"Experiment directory: {experiment_paths['experiment_dir']}")
    
    # Save configuration
    exp_manager.save_config(experiment_paths, config)
    
    # Initialize logger
    logger = ExperimentLogger(experiment_dir=experiment_paths["experiment_dir"])
    logger.log_config(config)
    
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
    print(f"Network architecture: {config['architecture']}")
    parameters = initialize_parameter(
        X_train,
        config["architecture"],
        weight_init=config.get("weight_initialization", "he")
    )
    
    model = {"parameters": parameters}
    
    # Create trainer
    trainer = Trainer(model, logger=logger)
    
    # Train model
    print("\nStarting training...")
    start_time = time.time()
    model = trainer.train(
        X_train, Y_train,
        X_val, Y_val,
        config,
        config["architecture"]
    )
    training_time = time.time() - start_time
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_results = trainer.evaluate(X_test, Y_test, config["architecture"], config.get("activation", "relu"))
    
    # Print results
    print(f"\nTest Accuracy: {test_results['accuracy']:.4f}")
    print(f"Test Loss: {test_results['loss']:.4f}")
    print(f"Test Precision: {test_results['precision']:.4f}")
    print(f"Test Recall: {test_results['recall']:.4f}")
    print(f"Test F1: {test_results['f1']:.4f}")
    
    # Save results
    history = trainer.get_history()
    summary = {
        "test_accuracy": float(test_results["accuracy"]),
        "test_loss": float(test_results["loss"]),
        "test_precision": float(test_results["precision"]),
        "test_recall": float(test_results["recall"]),
        "test_f1": float(test_results["f1"]),
        "best_val_accuracy": float(model.get("best_val_accuracy", 0)),
        "best_epoch": int(np.argmax(history["val_acc"])),
        "final_train_accuracy": history["train_acc"][-1],
        "final_val_accuracy": history["val_acc"][-1],
        "training_time_seconds": round(training_time, 1),
        "architecture": config["architecture"],
        "activation": config.get("activation", "relu"),
        "optimizer": config.get("optimizer", "sgd"),
        "learning_rate": config["learning_rate"],
        "epochs": config["epochs"],
        "batch_size": config.get("batch_size", 64),
        "regularization_type": config.get("regularization_type", "none"),
        "lambda_reg": config.get("lambda_reg", 0.0),
        "dropout_rate": config.get("dropout_rate", 0.5) if config.get("use_dropout", False) else 0.0
    }
    
    exp_manager.save_summary(experiment_paths, summary)
    
    # Create visualizer
    visualizer = Visualizer(save_dir=experiment_paths["figures_dir"])
    
    # Plot training history
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
    
    # Save model weights (the best-validation checkpoint restored by the trainer)
    with open(experiment_paths["final_model"], 'wb') as f:
        pickle.dump(model["parameters"], f)
    
    print(f"\nExperiment completed! Results saved to: {experiment_paths['experiment_dir']}")
    print(f"  - Metrics: {experiment_paths['metrics_file']}")
    print(f"  - Config: {experiment_paths['config_file']}")
    print(f"  - Summary: {experiment_paths['summary_file']}")
    print(f"  - Figures: {experiment_paths['figures_dir']}")
    print(f"  - Model: {experiment_paths['final_model']}")
    
    return summary


if __name__ == "__main__":
    main()
