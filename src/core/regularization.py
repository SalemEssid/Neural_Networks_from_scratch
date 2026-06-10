import numpy as np


class Regularizer:
    """
    Regularization techniques for neural networks.
    """
    
    @staticmethod
    def l2_penalty(parameters, lambda_reg):
        """
        L2 regularization penalty (Ridge).
        
        Args:
            parameters: Dictionary of weights
            lambda_reg: Regularization strength
        
        Returns:
            L2 penalty value
        """
        penalty = 0
        for key in parameters:
            if key.startswith('W'):  # Only regularize weights, not biases
                penalty += np.sum(np.square(parameters[key]))
        return (lambda_reg / 2) * penalty
    
    @staticmethod
    def l1_penalty(parameters, lambda_reg):
        """
        L1 regularization penalty (Lasso).
        
        Args:
            parameters: Dictionary of weights
            lambda_reg: Regularization strength
        
        Returns:
            L1 penalty value
        """
        penalty = 0
        for key in parameters:
            if key.startswith('W'):
                penalty += np.sum(np.abs(parameters[key]))
        return lambda_reg * penalty
    
    @staticmethod
    def elastic_net_penalty(parameters, lambda_reg, l1_ratio=0.5):
        """
        Elastic Net regularization (combination of L1 and L2).
        
        Args:
            parameters: Dictionary of weights
            lambda_reg: Regularization strength
            l1_ratio: Proportion of L1 vs L2 (0-1)
        
        Returns:
            Elastic Net penalty value
        """
        l2_penalty = Regularizer.l2_penalty(parameters, lambda_reg * (1 - l1_ratio))
        l1_penalty = Regularizer.l1_penalty(parameters, lambda_reg * l1_ratio)
        return l1_penalty + l2_penalty
    
    @staticmethod
    def l2_gradient(parameters, lambda_reg):
        """
        Compute L2 regularization gradients.
        
        Args:
            parameters: Dictionary of weights
            lambda_reg: Regularization strength
        
        Returns:
            Dictionary of regularization gradients
        """
        grads = {}
        for key in parameters:
            if key.startswith('W'):
                grads[f"d{key}"] = lambda_reg * parameters[key]
            else:  # Bias terms
                grads[f"d{key}"] = np.zeros_like(parameters[key])
        return grads
    
    @staticmethod
    def l1_gradient(parameters, lambda_reg):
        """
        Compute L1 regularization gradients.
        
        Args:
            parameters: Dictionary of weights
            lambda_reg: Regularization strength
        
        Returns:
            Dictionary of regularization gradients
        """
        grads = {}
        for key in parameters:
            if key.startswith('W'):
                grads[f"d{key}"] = lambda_reg * np.sign(parameters[key])
            else:  # Bias terms
                grads[f"d{key}"] = np.zeros_like(parameters[key])
        return grads
    
    @staticmethod
    def elastic_net_gradient(parameters, lambda_reg, l1_ratio=0.5):
        """
        Compute Elastic Net regularization gradients.
        
        Args:
            parameters: Dictionary of weights
            lambda_reg: Regularization strength
            l1_ratio: Proportion of L1 vs L2
        
        Returns:
            Dictionary of regularization gradients
        """
        grads_l2 = Regularizer.l2_gradient(parameters, lambda_reg * (1 - l1_ratio))
        grads_l1 = Regularizer.l1_gradient(parameters, lambda_reg * l1_ratio)
        
        grads = {}
        for key in grads_l2:
            grads[key] = grads_l2[key] + grads_l1[key]
        return grads


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
