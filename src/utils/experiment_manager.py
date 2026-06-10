#Create experiment folders
#Generate experiment ids
#Return paths
from pathlib import Path
from datetime import datetime
import json


class ExperimentManager:
    """
    Manage experiment directories, configuration, and results.
    """
    
    def __init__(self, root="experiments"):
        """
        Initialize experiment manager.
        
        Args:
            root: Root directory for experiments
        """
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
    
    def create_experiment(self, experiment_name=None):
        """
        Create a new experiment directory structure.
        
        Args:
            experiment_name: Name for the experiment (auto-generated if None)
        
        Returns:
            Dictionary with paths to experiment directories
        """
        if experiment_name is None:
            experiment_name = f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        experiment_dir = self.root / experiment_name
        figures_dir = experiment_dir / "figures"
        models_dir = experiment_dir / "models"
        
        experiment_dir.mkdir(parents=True, exist_ok=True)
        figures_dir.mkdir(parents=True, exist_ok=True)
        models_dir.mkdir(parents=True, exist_ok=True)
        
        return {
            "experiment_dir": experiment_dir,
            "figures_dir": figures_dir,
            "models_dir": models_dir,
            "metrics_file": experiment_dir / "metrics.csv",
            "config_file": experiment_dir / "config.json",
            "summary_file": experiment_dir / "summary.json",
            "best_model": models_dir / "best_model.pkl",
            "final_model": models_dir / "final_model.pkl",
            "experiment_name": experiment_name
        }
    
    def save_config(self, experiment_paths, config):
        """
        Save configuration to experiment.
        
        Args:
            experiment_paths: Paths dictionary from create_experiment
            config: Configuration dictionary
        """
        with open(experiment_paths["config_file"], 'w') as f:
            json.dump(config, f, indent=4)
    
    def load_config(self, experiment_dir):
        """
        Load configuration from experiment.
        
        Args:
            experiment_dir: Path to experiment directory
        
        Returns:
            Configuration dictionary
        """
        config_file = Path(experiment_dir) / "config.json"
        with open(config_file, 'r') as f:
            return json.load(f)
    
    def save_summary(self, experiment_paths, summary):
        """
        Save results summary.
        
        Args:
            experiment_paths: Paths dictionary
            summary: Summary dictionary
        """
        with open(experiment_paths["summary_file"], 'w') as f:
            json.dump(summary, f, indent=4)
    
    def load_summary(self, experiment_dir):
        """
        Load results summary.
        
        Args:
            experiment_dir: Path to experiment directory
        
        Returns:
            Summary dictionary
        """
        summary_file = Path(experiment_dir) / "summary.json"
        if summary_file.exists():
            with open(summary_file, 'r') as f:
                return json.load(f)
        return None
    
    def list_experiments(self):
        """
        List all experiments in root directory.
        
        Returns:
            List of experiment directory names
        """
        return [d.name for d in self.root.iterdir() if d.is_dir()]