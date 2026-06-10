import numpy as np
from src.core.activations import relu, softmax, sigmoid
from src.core.losses import computing_cost


def initialize_parameter(X, architecture, weight_init="he"):
    """
    Initialize parameters for a flexible neural network architecture.
    
    Args:
        X: Input data with shape (features, samples)
        architecture: List of layer sizes [input_size, hidden1, hidden2, ..., output_size]
        weight_init: Weight initialization method ("he" or "xavier")
    
    Returns:
        Dictionary with W and b for each layer
    """
    np.random.seed(42)
    parameters = {}
    input_size = X.shape[0]
    
    # Use provided architecture or infer from X
    if isinstance(architecture, list):
        layer_sizes = architecture
    else:
        layer_sizes = [input_size] + architecture[1:]
    
    for i in range(len(layer_sizes) - 1):
        layer_idx = i + 1
        prev_size = layer_sizes[i]
        curr_size = layer_sizes[i + 1]
        
        if weight_init == "he":
            scale = np.sqrt(2 / prev_size)
        elif weight_init == "xavier":
            scale = np.sqrt(1 / prev_size)
        else:
            scale = 0.01
        
        W = np.random.randn(curr_size, prev_size) * scale
        b = np.zeros((curr_size, 1))
        
        parameters[f"W{layer_idx}"] = W
        parameters[f"b{layer_idx}"] = b
    
    return parameters


def linear_forward(A, W, b):
    """Compute Z = W*A + b"""
    Z = W.dot(A) + b
    cache = (A, W, b)
    return Z, cache


def forward(X, parameters, architecture, activation="relu"):
    """
    Forward pass through the entire network.
    
    Args:
        X: Input data
        parameters: Dictionary of weights and biases
        architecture: List of layer sizes
        activation: Activation function ("relu" or "sigmoid")
    
    Returns:
        A_final: Output of the network
        cache: Dictionary of all intermediate values
    """
    cache = {}
    A = X
    num_layers = len(architecture) - 1
    
    # Forward pass through hidden layers
    for layer_idx in range(1, num_layers):
        Z, linear_cache = linear_forward(A, parameters[f"W{layer_idx}"], parameters[f"b{layer_idx}"])
        
        if activation == "relu":
            A = relu(Z)
        else:
            A = sigmoid(Z)
        
        cache[f"cache{layer_idx}"] = linear_cache
        cache[f"Z{layer_idx}"] = Z
        cache[f"A{layer_idx}"] = A
    
    # Output layer with softmax
    Z_final, linear_cache_final = linear_forward(A, parameters[f"W{num_layers}"], parameters[f"b{num_layers}"])
    A_final = softmax(Z_final)
    
    cache[f"cache{num_layers}"] = linear_cache_final
    cache[f"Z{num_layers}"] = Z_final
    cache[f"A{num_layers}"] = A_final
    
    return A_final, cache


def relu_backward(dA, Z):
    """Compute gradient of ReLU activation"""
    dZ = dA * (Z > 0).astype(int)
    return dZ


def softmax_backward(Y, A_final):
    """Compute gradient for softmax + cross-entropy"""
    return A_final - Y


def linear_backward(dZ, cache):
    """Compute gradients for linear layer"""
    A_prev, W, b = cache
    m = A_prev.shape[1]
    
    dW = (1/m) * dZ.dot(A_prev.T)
    db = (1/m) * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = W.T.dot(dZ)
    
    return dA_prev, dW, db


def backward(X, Y, A_final, cache, architecture, activation="relu"):
    """
    Backward pass through the entire network.
    
    Args:
        X: Input data
        Y: Ground truth labels (one-hot encoded)
        A_final: Output of the network
        cache: Dictionary of intermediate values
        architecture: List of layer sizes
        activation: Activation function
    
    Returns:
        grads: Dictionary of gradients
    """
    grads = {}
    num_layers = len(architecture) - 1
    
    # Output layer gradient (softmax + cross-entropy)
    dA = softmax_backward(Y, A_final)
    
    # Backward pass through layers (from output to input)
    for layer_idx in range(num_layers, 0, -1):
        # Compute linear gradients
        dA_prev, dW, db = linear_backward(dA, cache[f"cache{layer_idx}"])
        
        grads[f"dW{layer_idx}"] = dW
        grads[f"db{layer_idx}"] = db
        
        # Apply activation derivative for previous layer (if not input)
        if layer_idx > 1:
            dA = relu_backward(dA_prev, cache[f"Z{layer_idx-1}"])
        else:
            dA = dA_prev
    
    return grads


def update_parameters(parameters, grads, learning_rate, architecture):
    """
    Update parameters using gradient descent.
    
    Args:
        parameters: Dictionary of weights and biases
        grads: Dictionary of gradients
        learning_rate: Learning rate
        architecture: List of layer sizes
    
    Returns:
        Updated parameters
    """
    num_layers = len(architecture) - 1
    
    for layer_idx in range(1, num_layers + 1):
        parameters[f"W{layer_idx}"] -= learning_rate * grads[f"dW{layer_idx}"]
        parameters[f"b{layer_idx}"] -= learning_rate * grads[f"db{layer_idx}"]
    
    return parameters


def predict(X, parameters, architecture):
    """Make predictions on data"""
    A_final, _ = forward(X, parameters, architecture)
    return np.argmax(A_final, axis=0)

