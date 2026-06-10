#write metrics
#save configs

import csv
import json
from pathlib import Path
from datetime import datetime


class ExperimentLogger:
    """
    Flexible logger for experiment tracking and metrics.
    """
    
    def __init__(self, filepath=None, experiment_dir=None):
        """
        Initialize logger.
        
        Args:
            filepath: Path to CSV metrics file
            experiment_dir: Path to experiment directory (alternative to filepath)
        """
        if experiment_dir:
            self.experiment_dir = Path(experiment_dir)
            self.filepath = self.experiment_dir / "metrics.csv"
        else:
            self.filepath = Path(filepath) if filepath else None
        
        self.history = []
        self.fieldnames = None
        self.csv_initialized = False
    
    def log(self, epoch, train_loss, val_loss, train_acc, val_acc):
        """Legacy method for backward compatibility"""
        self.log_metrics({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_acc": train_acc,
            "val_acc": val_acc
        })
    
    def log_metrics(self, metrics_dict):
        """
        Log metrics for an epoch.
        
        Args:
            metrics_dict: Dictionary of metrics to log
        """
        if "timestamp" not in metrics_dict:
            metrics_dict["timestamp"] = datetime.now().isoformat()
        
        self.history.append(metrics_dict)
        
        # Initialize fieldnames from first log
        if self.fieldnames is None:
            self.fieldnames = list(metrics_dict.keys())
        
        # Append to CSV
        if self.filepath:
            with open(self.filepath, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                
                if not self.csv_initialized:
                    writer.writeheader()
                    self.csv_initialized = True
                
                writer.writerow(metrics_dict)
    
    def log_config(self, config):
        """Save configuration to JSON file."""
        if self.experiment_dir:
            config_file = self.experiment_dir / "config.json"
            with open(config_file, 'w') as f:
                json.dump(config, f, indent=4)
    
    def save_results_summary(self, summary):
        """Save final results summary."""
        if self.experiment_dir:
            summary_file = self.experiment_dir / "summary.json"
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=4)
    
    def get_history(self):
        """Get complete metrics history."""
        return self.history