CONFIG = {
    "seed": 42,

    "learning_rate": 0.001,
    "epochs": 15,
    # Number of samples per gradient step (one optimizer update per mini-batch)
    "batch_size": 64,

    "architecture": [784, 256, 128, 64, 10],

    # Choose: "relu" or "sigmoid" (use "xavier" initialization with sigmoid)
    "activation": "relu",

    "weight_initialization": "he",
    
    # ==================== OPTIMIZER ====================
    # Choose: "sgd", "momentum", "rmsprop", "adam"
    "optimizer": "adam",
    
    # Optimizer-specific parameters
    "optimizer_params": {
        # For Momentum
        "momentum": 0.9,
        
        # For RMSprop
        "decay_rate": 0.99,
        
        # For Adam
        "beta1": 0.9,
        "beta2": 0.999,
        
        # For all adaptive optimizers
        "epsilon": 1e-8
    },
    # ===================================================
    
    # ==================== REGULARIZATION ====================
    # Choose: "none", "l2", "l1", or "elastic_net"
    "regularization_type": "none",
    
    # Regularization strength (lambda). The penalty is divided by the batch size,
    # e.g. L2 adds (lambda / 2m) * sum(W^2) to each mini-batch loss
    "lambda_reg": 0.01,
    
    # For elastic net: ratio of L1 (0=pure L2, 1=pure L1)
    "l1_ratio": 0.5,
    # =========================================================
    
    # ==================== DROPOUT ====================
    # Enable dropout: True or False
    "use_dropout": False,
    
    # Dropout rate (0-1): probability of dropping a neuron
    # 0.0 = no dropout, 0.5 = drop 50% of neurons
    "dropout_rate": 0.5,
    
    # Dropout schedule: "constant", "linear", "exponential", "step"
    "dropout_schedule": "constant",
    # ==================================================
}