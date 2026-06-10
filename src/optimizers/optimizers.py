import numpy as np


class Optimizer:
    """
    Base optimizer class.
    """
    
    def __init__(self, learning_rate=0.01):
        """
        Initialize optimizer.
        
        Args:
            learning_rate: Learning rate (step size)
        """
        self.learning_rate = learning_rate
    
    def update(self, parameters, grads):
        """
        Update parameters based on gradients.
        
        Args:
            parameters: Model parameters
            grads: Gradients
        
        Returns:
            Updated parameters
        """
        raise NotImplementedError


class SGD(Optimizer):
    """
    Stochastic Gradient Descent optimizer.
    Simple gradient descent with learning rate.
    """
    
    def __init__(self, learning_rate=0.01):
        """
        Initialize SGD optimizer.
        
        Args:
            learning_rate: Learning rate
        """
        super().__init__(learning_rate)
    
    def update(self, parameters, grads):
        """
        Update parameters using SGD.
        
        Args:
            parameters: Model parameters
            grads: Gradients
        
        Returns:
            Updated parameters
        """
        num_layers = (len(parameters) + 1) // 2  # Count W and b pairs
        
        for layer_idx in range(1, num_layers + 1):
            parameters[f"W{layer_idx}"] -= self.learning_rate * grads[f"dW{layer_idx}"]
            parameters[f"b{layer_idx}"] -= self.learning_rate * grads[f"db{layer_idx}"]
        
        return parameters


class Momentum(Optimizer):
    """
    SGD with Momentum optimizer.
    Accelerates gradient descent by accumulating gradients.
    """
    
    def __init__(self, learning_rate=0.01, momentum=0.9):
        """
        Initialize Momentum optimizer.
        
        Args:
            learning_rate: Learning rate
            momentum: Momentum coefficient (0-1, typically 0.9)
        """
        super().__init__(learning_rate)
        self.momentum = momentum
        self.velocities = None
    
    def update(self, parameters, grads):
        """
        Update parameters using Momentum.
        
        Args:
            parameters: Model parameters
            grads: Gradients
        
        Returns:
            Updated parameters
        """
        # Initialize velocities on first call
        if self.velocities is None:
            self.velocities = {}
            for key in parameters:
                self.velocities[key] = np.zeros_like(parameters[key])
        
        num_layers = (len(parameters) + 1) // 2
        
        for layer_idx in range(1, num_layers + 1):
            # Update velocity
            self.velocities[f"W{layer_idx}"] = (
                self.momentum * self.velocities[f"W{layer_idx}"] - 
                self.learning_rate * grads[f"dW{layer_idx}"]
            )
            self.velocities[f"b{layer_idx}"] = (
                self.momentum * self.velocities[f"b{layer_idx}"] - 
                self.learning_rate * grads[f"db{layer_idx}"]
            )
            
            # Update parameters
            parameters[f"W{layer_idx}"] += self.velocities[f"W{layer_idx}"]
            parameters[f"b{layer_idx}"] += self.velocities[f"b{layer_idx}"]
        
        return parameters


class RMSprop(Optimizer):
    """
    RMSprop optimizer.
    Adapts learning rate based on root mean square of gradients.
    """
    
    def __init__(self, learning_rate=0.01, decay_rate=0.99, epsilon=1e-8):
        """
        Initialize RMSprop optimizer.
        
        Args:
            learning_rate: Learning rate
            decay_rate: Decay rate for moving average (typically 0.9-0.999)
            epsilon: Small value to prevent division by zero
        """
        super().__init__(learning_rate)
        self.decay_rate = decay_rate
        self.epsilon = epsilon
        self.mean_squares = None
    
    def update(self, parameters, grads):
        """
        Update parameters using RMSprop.
        
        Args:
            parameters: Model parameters
            grads: Gradients
        
        Returns:
            Updated parameters
        """
        # Initialize mean squares on first call
        if self.mean_squares is None:
            self.mean_squares = {}
            for key in parameters:
                self.mean_squares[key] = np.zeros_like(parameters[key])
        
        num_layers = (len(parameters) + 1) // 2
        
        for layer_idx in range(1, num_layers + 1):
            # Update mean squares
            self.mean_squares[f"W{layer_idx}"] = (
                self.decay_rate * self.mean_squares[f"W{layer_idx}"] +
                (1 - self.decay_rate) * np.square(grads[f"dW{layer_idx}"])
            )
            self.mean_squares[f"b{layer_idx}"] = (
                self.decay_rate * self.mean_squares[f"b{layer_idx}"] +
                (1 - self.decay_rate) * np.square(grads[f"db{layer_idx}"])
            )
            
            # Update parameters
            parameters[f"W{layer_idx}"] -= (
                self.learning_rate * grads[f"dW{layer_idx}"] /
                (np.sqrt(self.mean_squares[f"W{layer_idx}"]) + self.epsilon)
            )
            parameters[f"b{layer_idx}"] -= (
                self.learning_rate * grads[f"db{layer_idx}"] /
                (np.sqrt(self.mean_squares[f"b{layer_idx}"]) + self.epsilon)
            )
        
        return parameters


class Adam(Optimizer):
    """
    Adam optimizer.
    Combines benefits of momentum and RMSprop.
    Adaptive moment estimation.
    """
    
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        """
        Initialize Adam optimizer.
        
        Args:
            learning_rate: Learning rate (default 0.001 for Adam)
            beta1: Exponential decay rate for first moment (typically 0.9)
            beta2: Exponential decay rate for second moment (typically 0.999)
            epsilon: Small value to prevent division by zero
        """
        super().__init__(learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = None  # First moment (mean)
        self.v = None  # Second moment (variance)
        self.t = 0     # Time step counter
    
    def update(self, parameters, grads):
        """
        Update parameters using Adam.
        
        Args:
            parameters: Model parameters
            grads: Gradients
        
        Returns:
            Updated parameters
        """
        # Initialize first and second moments on first call
        if self.m is None:
            self.m = {}
            self.v = {}
            for key in parameters:
                self.m[key] = np.zeros_like(parameters[key])
                self.v[key] = np.zeros_like(parameters[key])
        
        self.t += 1
        
        # Bias correction terms
        bias_correction1 = 1 - self.beta1 ** self.t
        bias_correction2 = 1 - self.beta2 ** self.t
        
        num_layers = (len(parameters) + 1) // 2
        
        for layer_idx in range(1, num_layers + 1):
            # Update biased first moment estimate (momentum)
            self.m[f"W{layer_idx}"] = (
                self.beta1 * self.m[f"W{layer_idx}"] +
                (1 - self.beta1) * grads[f"dW{layer_idx}"]
            )
            self.m[f"b{layer_idx}"] = (
                self.beta1 * self.m[f"b{layer_idx}"] +
                (1 - self.beta1) * grads[f"db{layer_idx}"]
            )
            
            # Update biased second moment estimate (RMS)
            self.v[f"W{layer_idx}"] = (
                self.beta2 * self.v[f"W{layer_idx}"] +
                (1 - self.beta2) * np.square(grads[f"dW{layer_idx}"])
            )
            self.v[f"b{layer_idx}"] = (
                self.beta2 * self.v[f"b{layer_idx}"] +
                (1 - self.beta2) * np.square(grads[f"db{layer_idx}"])
            )
            
            # Compute bias-corrected first moment estimate
            m_corrected_W = self.m[f"W{layer_idx}"] / bias_correction1
            m_corrected_b = self.m[f"b{layer_idx}"] / bias_correction1
            
            # Compute bias-corrected second moment estimate
            v_corrected_W = self.v[f"W{layer_idx}"] / bias_correction2
            v_corrected_b = self.v[f"b{layer_idx}"] / bias_correction2
            
            # Update parameters
            parameters[f"W{layer_idx}"] -= (
                self.learning_rate * m_corrected_W /
                (np.sqrt(v_corrected_W) + self.epsilon)
            )
            parameters[f"b{layer_idx}"] -= (
                self.learning_rate * m_corrected_b /
                (np.sqrt(v_corrected_b) + self.epsilon)
            )
        
        return parameters


def get_optimizer(optimizer_type, learning_rate, **kwargs):
    """
    Factory function to get optimizer instance.
    
    Args:
        optimizer_type: Type of optimizer ("sgd", "momentum", "rmsprop", "adam")
        learning_rate: Learning rate
        **kwargs: Additional optimizer-specific arguments
    
    Returns:
        Optimizer instance
    """
    optimizer_type = optimizer_type.lower()
    
    if optimizer_type == "sgd":
        return SGD(learning_rate=learning_rate)
    
    elif optimizer_type == "momentum":
        momentum = kwargs.get("momentum", 0.9)
        return Momentum(learning_rate=learning_rate, momentum=momentum)
    
    elif optimizer_type == "rmsprop":
        decay_rate = kwargs.get("decay_rate", 0.99)
        epsilon = kwargs.get("epsilon", 1e-8)
        return RMSprop(learning_rate=learning_rate, decay_rate=decay_rate, epsilon=epsilon)
    
    elif optimizer_type == "adam":
        beta1 = kwargs.get("beta1", 0.9)
        beta2 = kwargs.get("beta2", 0.999)
        epsilon = kwargs.get("epsilon", 1e-8)
        return Adam(learning_rate=learning_rate, beta1=beta1, beta2=beta2, epsilon=epsilon)
    
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")


def get_optimizer_info(optimizer_type):
    """
    Get information about an optimizer.
    
    Args:
        optimizer_type: Type of optimizer
    
    Returns:
        Dictionary with optimizer information
    """
    info = {
        "sgd": {
            "name": "Stochastic Gradient Descent",
            "description": "Basic gradient descent with constant learning rate",
            "pros": ["Simple", "Fast", "Low memory"],
            "cons": ["Can get stuck in local minima", "No adaptive learning rate"],
            "use_case": "Baseline, simple problems"
        },
        "momentum": {
            "name": "Momentum",
            "description": "SGD with momentum to accelerate convergence",
            "pros": ["Faster convergence", "Escapes shallow local minima"],
            "cons": ["Requires momentum parameter tuning"],
            "use_case": "General purpose, medium difficulty"
        },
        "rmsprop": {
            "name": "RMSprop",
            "description": "Adaptive learning rate based on gradient magnitude",
            "pros": ["Adaptive learning rates", "Handles sparse gradients"],
            "cons": ["More complex", "More hyperparameters"],
            "use_case": "RNNs, sparse gradients"
        },
        "adam": {
            "name": "Adam",
            "description": "Combines momentum and RMSprop (Adaptive Moment Estimation)",
            "pros": ["Combines best of momentum and RMSprop", "Works well in practice"],
            "cons": ["Most parameters to tune", "Slower per iteration"],
            "use_case": "Default choice for most problems"
        }
    }
    
    return info.get(optimizer_type.lower(), None)
