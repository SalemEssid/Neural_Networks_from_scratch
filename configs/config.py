CONFIG = {
    "seed": 42,

    "learning_rate": 0.1,
    "epochs": 80,
    "batch_size": 64,

    "architecture": [784, 128, 64, 10],

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
        "beta2": 0.95,
        
        # For all adaptive optimizers
        "epsilon": 1e-8
    },
    # ===================================================
    
    # ==================== REGULARIZATION ====================
    # Choose: "none", "l2", "l1", or "elastic_net"
    "regularization_type": "none",
    
    # Regularization strength (lambda)
    "lambda_reg": 0.1,
    
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