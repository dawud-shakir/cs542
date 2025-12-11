"""
layer.py
"""
import numpy as np

from package.pmat import pmat

### Activations ##############################################################
def ReLU(z): 
    return np.maximum(0.0, z)           # (out_size, batch_size)               

def ReLU_derivative(z):
    return (z > 0).astype(float)        # (out_size, batch_size)

def linear(z): 
    return z                            # (in_size, batch_size)

def linear_derivative(z):
    return np.ones_like(z)              # (in_size, batch_size)

# Activation for output layer
def log_softmax(z):
    # For numerical stability 
    max_z = np.max(z, axis=1, keepdims=True)  
    
    exp_z = np.exp(z - max_z)
    log_z = z - np.log(np.sum(exp_z, axis=1, keepdims=True)) - max_z

    return log_z                        # (num_classes, batch_size)

### Loss ##############################################################
def nll_loss(log_probs, true_labels):
    """.
    log_probs: (batch_size, num_classes)
    true_labels: (batch_size,)
    Returns: scalar loss value
    """
    
    # Extract log probabilities of the true classes
    batch_size = log_probs.shape[0]     # (batch_size, num_classes) 
    nll_loss_value = -np.mean(log_probs[np.arange(batch_size), true_labels])    
    
    return nll_loss_value # scalar
 
def nll_loss_derivative(log_probs, true_labels):
    """Derivative of NLL loss w.r.t. log_softmax output"""
    
    # Convert log probabilities to probabilities
    probs = np.exp(log_probs)
    batch_size, _ = probs.shape         # (batch_size, num_classes)  
    
    # One-hot encode true labels (1=correct class, 0=all others)
    one_hot = np.zeros_like(probs)
    one_hot[np.arange(batch_size), true_labels] = 1  # (batch_size, num_classes)
    
    # Combined gradient
    return probs - one_hot              # (batch_size, num_classes)

class Parallel_Layer:
    def __init__(self, input_size, output_size):
        # Uniform initialization (Xavier)
        weight_bound = np.sqrt(6.0 / (input_size + output_size))
        W = np.random.uniform(-weight_bound, weight_bound, size=(output_size, input_size))
        
        # Bias column
        bias_bound = 1.0 / np.sqrt(input_size)
        bias = np.random.uniform(-bias_bound, bias_bound, size=(output_size, 1))

        # Add bias column to weights
        W = np.hstack([bias, W])            # (out_size, in_size + 1)

        self.p_W = pmat.from_numpy(W)


        # Store input size (without bias) so _as_2d can check whether transpose is needed
        self.in_features = input_size

        # Default activation is linear
        self.phi = linear
        self.phi_prime = linear_derivative

        # Weights and biases gradient
        self.p_dL_dW = None # indicate no gradient stored yet

        # Update variables (Adam optimizer)
        self.b1, self.b2 = (0.9, 0.999)
        self.epsilon = 1e-8
        self.weight_decay =  0.0
        
        self.m = pmat(self.p_W.shape[0], self.p_W.shape[1]) # first moment
        self.v = pmat(self.p_W.shape[0], self.p_W.shape[1]) # second moment
        self.t = 0  # time step

    # Ensures that input is always (in_size, batch_size)
    def _as_2d(self, p_X):
        # Transpose only if not already transposed
        if p_X.shape[0] != self.in_features:
            if p_X.ndim == 2 and p_X.shape[1] == self.in_features:
                p_X = p_X.T
            else:
                raise ValueError(f"Expected input with {self.in_features} rows, got {p_X.shape}")
        return p_X

    # Add bias row to input    
    def _with_bias(self, p_X_no_bias):
        return p_X_no_bias.stack_ones_on_top()

    ############################################################################

    def forward(self, p_X_no_bias):
        # We need to add the bias here since X_no_bias is the activation of the previous layer, which has no bias row
        p_X_no_bias = self._as_2d(p_X_no_bias)      # (in_size, batch_size)
        self.p_X = self._with_bias(p_X_no_bias)     # (in_size + 1, batch_size)
        
        # Compute A = W * X
        self.p_A = self.p_W @ self.p_X              # (out_size, batch_size)
        
        # Apply activation function
        self.p_H = self.phi(self.p_A)               # (out_size, batch_size)

        assert self.p_W.shape[1] == self.p_X.shape[0], (self.p_W.shape, self.p_X.shape)

        # Pass activation to next layer
        return self.p_H
    
    ############################################################################

    def backward(self, p_dL_dH_next):
        """
            dL_dh_next (out_size, batch_size) is the gradient, ∂L/∂H, from the next layer. Objviously, there is no next layer after the final layer, so this is ∂L/∂H of that layer.
        """
      
        # Compute ∂L/∂A = ∂L/∂H * φ'(A)
        p_dL_dA = p_dL_dH_next * self.phi_prime(self.p_A)

        assert p_dL_dA.shape == self.p_A.shape, (p_dL_dA.shape, self.p_A.shape)
        
        # Compute ∂L/∂W = ∂L/∂A * X^T
        self.p_dL_dW = p_dL_dA @ self.p_X.T

        # Exclude bias from W
        p_W_no_bias = self.p_W.remove_first_column()

        # Compute ∂L/∂H_prev = W^T * ∂L/∂A
        p_dL_dH_prev = p_W_no_bias.T @ p_dL_dA      # (in_prev_size, batch_size)

        # Pass gradient to previous layer
        return p_dL_dH_prev 
    
    ############################################################################
        
    def update_weights(self, alpha=1e-3):
        if self.p_dL_dW is None:
            raise ValueError("No gradient stored. Cannot update weights.")

        self.t += 1             # increment time
        p_g = self.p_dL_dW      # intermediate variable for gradient

        # Weight decay
        if self.weight_decay != 0:
            p_g = p_g + self.weight_decay * self.p_W

        # Copy in case MPI-communicator is freed during update 
        p_m = self.m 
        p_v = self.v

        # First and second moment estimates
        p_m = self.b1 * p_m + (1 - self.b1) * p_g
        p_v = self.b2 * p_v + (1 - self.b2) * (p_g * p_g)

        # Bias correction
        p_m_hat = p_m / (1 - self.b1 ** self.t)
        p_v_hat = p_v / (1 - self.b2 ** self.t)

        # Update weights
        self.p_W -= alpha * p_m_hat * (1 / (np.sqrt(p_v_hat) + self.epsilon))

        self.p_dL_dW = None    # do not allow the same gradient to be used again