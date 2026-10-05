# Neural Network from Scratch for MNIST

A fully connected neural network written in plain NumPy (no deep-learning framework), with configurable depth, four optimizers, L1/L2 penalties, dropout and per-run experiment tracking.

**Best result: {{BEST}}**

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/figures/test_accuracy-dark.png">
  <img alt="Dot plot of test accuracy for every experiment, grouped by optimizer, weight penalty, dropout, hidden layers and activation" src="docs/figures/test_accuracy-light.png">
</picture>

## Quick start

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt

python main.py                  # train one model with configs/config.py
python run_experiments.py       # reproduce every result below
python make_figures.py          # rebuild the README figures
```

Each run writes `experiments/<name>/`: `config.json`, per-epoch `metrics.csv`, `summary.json`, training-curve and confusion-matrix plots, and the best-validation weights (`models/final_model.pkl`, not tracked in git).

## What's implemented

| Part | Details |
|---|---|
| Network | Any list of layer sizes, ReLU or sigmoid hidden layers, softmax output, He or Xavier initialization |
| Training | Mini-batch gradient descent, reshuffled every epoch; the weights from the best validation epoch are kept |
| Optimizers | SGD, Momentum, RMSprop, Adam (with bias correction) |
| Regularization | L2, L1 and elastic net (penalty divided by the batch size, `λ/2m · ΣW²` for L2); inverted dropout with optional rate schedules |
| Data | MNIST split into 48,000 train / 12,000 validation / 10,000 test, pixels scaled to [0, 1] |

Backpropagation is checked against numerical gradients.

## Results

{{RESULTS}}

## Configuration

Everything is set in [`configs/config.py`](configs/config.py); the experiment name is set at the top of [`main.py`](main.py).

```python
CONFIG = {
    "seed": 42,
    "learning_rate": 0.001,
    "epochs": 15,
    "batch_size": 64,
    "architecture": [784, 256, 128, 64, 10],
    "activation": "relu",                 # "relu" or "sigmoid"
    "weight_initialization": "he",        # "he" or "xavier"
    "optimizer": "adam",                  # "sgd", "momentum", "rmsprop", "adam"
    "regularization_type": "none",        # "none", "l2", "l1", "elastic_net"
    "lambda_reg": 0.01,
    "use_dropout": False,
    "dropout_rate": 0.5,
    "dropout_schedule": "constant",       # "constant", "linear", "exponential", "step"
    # plus optimizer_params (momentum, decay_rate, beta1, beta2, epsilon) and l1_ratio
}
```

## Project structure

```
main.py               train one model
run_experiments.py    experiment suite + results table (--table)
make_figures.py       README figures -> docs/figures/
configs/config.py     configuration
src/
  core/               network (forward/backward), activations, loss, regularization + dropout
  optimizers/         SGD, Momentum, RMSprop, Adam
  training/           mini-batch training loop
  evaluation/         accuracy, confusion matrix, precision/recall/F1
  utils/              data loading, logging, experiment folders, plots, seeding
experiments/          one folder per run
```

**Author:** Essid Salem
