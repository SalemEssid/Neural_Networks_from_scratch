"""
Run the experiment suite whose results are reported in README.md.

Every experiment starts from BASE_CONFIG and changes only the keys listed for it,
so each comparison isolates one choice. Results go to experiments/<name>/.

Usage:
    python run_experiments.py                       # run the whole suite
    python run_experiments.py opt_sgd reg_l2_0.1    # run only the named experiments
    python run_experiments.py --table               # print the results table only
"""
import copy
import json
import sys
from pathlib import Path


BASE_CONFIG = {
    "seed": 42,
    "learning_rate": 0.001,
    "epochs": 15,
    "batch_size": 64,
    "architecture": [784, 256, 128, 64, 10],
    "activation": "relu",
    "weight_initialization": "he",
    "optimizer": "adam",
    "optimizer_params": {
        "momentum": 0.9,
        "decay_rate": 0.99,
        "beta1": 0.9,
        "beta2": 0.999,
        "epsilon": 1e-8
    },
    "regularization_type": "none",
    "lambda_reg": 0.0,
    "l1_ratio": 0.5,
    "use_dropout": False,
    "dropout_rate": 0.0,
    "dropout_schedule": "constant",
}

EXPERIMENTS = {
    # Optimizers (opt_adam is also the baseline for every other group)
    "opt_sgd": {"optimizer": "sgd", "learning_rate": 0.1},
    "opt_momentum": {"optimizer": "momentum", "learning_rate": 0.01},
    "opt_rmsprop": {"optimizer": "rmsprop", "learning_rate": 0.001},
    "opt_adam": {},
    "opt_adam_lr0.1": {"learning_rate": 0.1},
    
    # Weight penalties
    "reg_l2_0.01": {"regularization_type": "l2", "lambda_reg": 0.01},
    "reg_l2_0.1": {"regularization_type": "l2", "lambda_reg": 0.1},
    "reg_l2_1": {"regularization_type": "l2", "lambda_reg": 1.0},
    "reg_l1_0.01": {"regularization_type": "l1", "lambda_reg": 0.01},
    
    # Dropout
    "dropout_0.1": {"use_dropout": True, "dropout_rate": 0.1},
    "dropout_0.3": {"use_dropout": True, "dropout_rate": 0.3},
    "dropout_0.5": {"use_dropout": True, "dropout_rate": 0.5},
    
    # Architecture
    "arch_128": {"architecture": [784, 128, 10]},
    "arch_128_64": {"architecture": [784, 128, 64, 10]},
    "arch_512_256_128_64": {"architecture": [784, 512, 256, 128, 64, 10]},
    
    # Activation
    "act_sigmoid": {"activation": "sigmoid", "weight_initialization": "xavier"},
}


def build_config(overrides):
    """Return BASE_CONFIG with the given keys replaced."""
    config = copy.deepcopy(BASE_CONFIG)
    config.update(copy.deepcopy(overrides))
    return config


def print_table(names):
    """Print a markdown table of the summaries saved for the given experiments."""
    print("| Experiment | Test Acc | Best Val Acc | Best Epoch | Final Train Acc | Final Val Acc | Gap | Time (s) |")
    print("|---|---|---|---|---|---|---|---|")
    for name in names:
        summary_file = Path("experiments") / name / "summary.json"
        if not summary_file.exists():
            print(f"| {name} | (not run) | | | | | | |")
            continue
        s = json.loads(summary_file.read_text())
        gap = s["final_train_accuracy"] - s["final_val_accuracy"]
        print(f"| {name} | {s['test_accuracy']:.2%} | {s['best_val_accuracy']:.2%} | {s['best_epoch']} | "
              f"{s['final_train_accuracy']:.2%} | {s['final_val_accuracy']:.2%} | {gap:+.2%} | "
              f"{s['training_time_seconds']} |")


if __name__ == "__main__":
    args = sys.argv[1:]
    
    if args == ["--table"]:
        print_table(EXPERIMENTS)
        sys.exit(0)
    
    unknown = [name for name in args if name not in EXPERIMENTS]
    if unknown:
        sys.exit(f"Unknown experiment(s): {', '.join(unknown)}. Choose from: {', '.join(EXPERIMENTS)}")
    
    # Imported here so --table works without loading TensorFlow
    from main import main
    
    names = args or list(EXPERIMENTS)
    for name in names:
        print(f"\n{'=' * 60}\nRunning {name}\n{'=' * 60}")
        main(build_config(EXPERIMENTS[name]), experiment_name=name)
    
    print()
    print_table(names)
