import numpy as np


class Regularizer:
    """
    Regularization techniques for neural networks.
    
    Penalties are divided by the mini-batch size m, the same way the
    cross-entropy loss is averaged over m (the usual "lambda / 2m" convention).
    """
    
    @staticmethod
    def l2_penalty(parameters, lambda_reg, m=1):
        """
        L2 regularization penalty (Ridge).
        
        Args:
            parameters: Dictionary of weights
            lambda_reg: Regularization strength
            m: Number of samples in the batch
        
        Returns:
            L2 penalty value
        """
        penalty = 0
        for key in parameters:
            if key.startswith('W'):  # Only regularize weights, not biases
                penalty += np.sum(np.square(parameters[key]))
        return (lambda_reg / (2 * m)) * penalty
    
    @staticmethod
    def l1_penalty(parameters, lambda_reg, m=1):
        """
        L1 regularization penalty (Lasso).
        
        Args:
            parameters: Dictionary of weights
            lambda_reg: Regularization strength
            m: Number of samples in the batch
        
        Returns:
            L1 penalty value
        """
        penalty = 0
        for key in parameters:
            if key.startswith('W'):
                penalty += np.sum(np.abs(parameters[key]))
        return (lambda_reg / m) * penalty
    
    @staticmethod
    def elastic_net_penalty(parameters, lambda_reg, l1_ratio=0.5, m=1):
        """
        Elastic Net regularization (combination of L1 and L2).
        
        Args:
            parameters: Dictionary of weights
            lambda_reg: Regularization strength
            l1_ratio: Proportion of L1 vs L2 (0-1)
            m: Number of samples in the batch
        
        Returns:
            Elastic Net penalty value
        """
        l2_penalty = Regularizer.l2_penalty(parameters, lambda_reg * (1 - l1_ratio), m)
        l1_penalty = Regularizer.l1_penalty(parameters, lambda_reg * l1_ratio, m)
        return l1_penalty + l2_penalty
    
    @staticmethod
    def l2_gradient(parameters, lambda_reg, m=1):
        """
        Compute L2 regularization gradients.
        
        Args:
            parameters: Dictionary of weights
            lambda_reg: Regularization strength
            m: Number of samples in the batch
        
        Returns:
            Dictionary of regularization gradients
        """
        grads = {}
        for key in parameters:
            if key.startswith('W'):
                grads[f"d{key}"] = (lambda_reg / m) * parameters[key]
            else:  # Bias terms
                grads[f"d{key}"] = np.zeros_like(parameters[key])
        return grads
    
    @staticmethod
    def l1_gradient(parameters, lambda_reg, m=1):
        """
        Compute L1 regularization gradients.
        
        Args:
            parameters: Dictionary of weights
            lambda_reg: Regularization strength
            m: Number of samples in the batch
        
        Returns:
            Dictionary of regularization gradients
        """
        grads = {}
        for key in parameters:
            if key.startswith('W'):
                grads[f"d{key}"] = (lambda_reg / m) * np.sign(parameters[key])
            else:  # Bias terms
                grads[f"d{key}"] = np.zeros_like(parameters[key])
        return grads
    
    @staticmethod
    def elastic_net_gradient(parameters, lambda_reg, l1_ratio=0.5, m=1):
        """
        Compute Elastic Net regularization gradients.
        
        Args:
            parameters: Dictionary of weights
            lambda_reg: Regularization strength
            l1_ratio: Proportion of L1 vs L2
            m: Number of samples in the batch
        
        Returns:
            Dictionary of regularization gradients
        """
        grads_l2 = Regularizer.l2_gradient(parameters, lambda_reg * (1 - l1_ratio), m)
        grads_l1 = Regularizer.l1_gradient(parameters, lambda_reg * l1_ratio, m)
        
        grads = {}
        for key in grads_l2:
            grads[key] = grads_l2[key] + grads_l1[key]
        return grads
    
    @staticmethod
    def gradient(reg_type, parameters, lambda_reg, m, l1_ratio=0.5):
        """
        Compute regularization gradients for the configured regularization type.
        
        Args:
            reg_type: "none", "l2", "l1" or "elastic_net"
            parameters: Dictionary of weights
            lambda_reg: Regularization strength
            m: Number of samples in the batch
            l1_ratio: Proportion of L1 vs L2 (elastic net only)
        
        Returns:
            Dictionary of regularization gradients (empty for "none")
        """
        if reg_type in (None, "none"):
            return {}
        if reg_type == "l2":
            return Regularizer.l2_gradient(parameters, lambda_reg, m)
        if reg_type == "l1":
            return Regularizer.l1_gradient(parameters, lambda_reg, m)
        if reg_type == "elastic_net":
            return Regularizer.elastic_net_gradient(parameters, lambda_reg, l1_ratio, m)
        raise ValueError(f"Unknown regularization type: {reg_type}")


class Dropout:
    """
    Dropout regularization technique.
    Randomly drops neurons during training to prevent overfitting.
    """
    
    def __init__(self, dropout_rate=0.5):
        """
        Initialize Dropout.
        
        Args:
            dropout_rate: Probability of dropping a neuron (0-1)
                         0.0 = no dropout, 1.0 = drop all neurons
        """
        if not 0 <= dropout_rate < 1:
            raise ValueError(f"Dropout rate must be in [0, 1), got {dropout_rate}")
        
        self.dropout_rate = dropout_rate
        self.keep_prob = 1 - dropout_rate
        self.mask = None
    
    def forward(self, A, training=True):
        """
        Apply dropout to activations.
        
        Args:
            A: Activation matrix (neurons, samples)
            training: If True, apply dropout. If False, return as-is (inference)
        
        Returns:
            A_dropout: Activations after dropout
            mask: Dropout mask for backward pass
        """
        if not training or self.dropout_rate == 0:
            return A, None
        
        # Create dropout mask
        self.mask = np.random.binomial(1, self.keep_prob, size=A.shape) / self.keep_prob
        
        # Apply dropout
        A_dropout = A * self.mask
        
        return A_dropout, self.mask
    
    def backward(self, dA, mask):
        """
        Apply dropout to gradients during backpropagation.
        
        Args:
            dA: Gradient of activation
            mask: Dropout mask from forward pass
        
        Returns:
            dA_dropout: Gradients after dropout
        """
        if mask is None:
            return dA
        
        return dA * mask
    
    @staticmethod
    def create_dropout_schedule(initial_rate=0.5, epochs=100, schedule_type="constant"):
        """
        Create a dropout rate schedule for the training process.
        
        Args:
            initial_rate: Starting dropout rate
            epochs: Total number of epochs
            schedule_type: "constant", "linear", "exponential", "step"
        
        Returns:
            List of dropout rates for each epoch
        """
        schedule = []
        
        if schedule_type == "constant":
            schedule = [initial_rate] * epochs
        
        elif schedule_type == "linear":
            # Linearly decrease dropout over time
            schedule = [initial_rate * (1 - epoch / epochs) for epoch in range(epochs)]
        
        elif schedule_type == "exponential":
            # Exponentially decay dropout
            decay_rate = np.log(0.1) / epochs
            schedule = [initial_rate * np.exp(decay_rate * epoch) for epoch in range(epochs)]
        
        elif schedule_type == "step":
            # Step-wise decrease dropout
            step_size = epochs // 3
            for epoch in range(epochs):
                if epoch < step_size:
                    schedule.append(initial_rate)
                elif epoch < 2 * step_size:
                    schedule.append(initial_rate * 0.5)
                else:
                    schedule.append(initial_rate * 0.25)
        
        else:
            raise ValueError(f"Unknown dropout schedule: {schedule_type}")
        
        return schedule
    
    @staticmethod
    def get_dropout_info(dropout_rate):
        """
        Get information about dropout configuration.
        
        Args:
            dropout_rate: Dropout rate value
        
        Returns:
            Dictionary with dropout statistics
        """
        return {
            "dropout_rate": dropout_rate,
            "keep_probability": 1 - dropout_rate,
            "neurons_dropped_percent": dropout_rate * 100,
            "effective_scaling": 1 / (1 - dropout_rate) if dropout_rate < 1 else float('inf')
        }
