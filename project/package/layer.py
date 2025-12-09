"""
layer.py
"""
from mpi4py import MPI
from package.pmat import pmat, create_grid_comm


import numpy as np

# Activations
def ReLU(z): return np.maximum(0.0, z)        # (out, batch)               
def ReLU_derivative(z): return (z > 0).astype(float)    # (out, batch)
def linear(z): return z                             # (features, batch)
def linear_derivative(z): return np.ones_like(z)    # (features, batch)

# Not used directly right now
# def mean_squared_error(y, y_hat):
#     return np.mean((y_hat - y) ** 2)

# def mean_squared_error_derivative(y, y_hat):
#     # d/dy_hat of MSE with mean over all elements
#     return (2.0 / y.size) * (y_hat - y)

def log_softmax(z):
    """
    log(softmax(z)) = log(exp(z)/sum(exp(z))) = z - log(sum(exp(z)))
    """
    max_z = np.max(z, axis=1, keepdims=True)  # For numerical stability
    exp_z = np.exp(z - max_z)
    log_z = z - np.log(np.sum(exp_z, axis=1, keepdims=True)) - max_z

    return log_z

def nll_loss(log_probs, true_labels):
    """
    Computes Negative Log Likelihood (NLL) loss.
    log_probs: (batch_size, num_classes)
    true_labels: (batch_size,)
    Returns: scalar loss value
    """
    # Extract log probabilities of true classes
    batch_size = log_probs.shape[0]
    nll_loss_value = -np.mean(log_probs[np.arange(batch_size), true_labels])
    
    return nll_loss_value

def nll_loss_derivative(log_probs, true_labels):
    """Derivative of NLL loss w.r.t. log_softmax output"""
    probs = np.exp(log_probs)
    batch_size, num_classes = probs.shape
    
    # Create one-hot encoding
    one_hot = np.zeros_like(probs)
    one_hot[np.arange(batch_size), true_labels] = 1
    
    # Combined gradient
    return probs - one_hot  # (batch_size, num_classes)

class Parallel_Layer:
    def __init__(self, input_size, output_size):
        


        ############################################################
        # Initialize weights and biases
        ############################################################

        # Weights: Uniform initialization (Xavier)
        weight_bound = np.sqrt(6.0 / (input_size + output_size))
        
        # Bias column: Uniform(-1/sqrt(fan_in), +1/sqrt(fan_in))
        bias_bound = 1.0 / np.sqrt(input_size)

        W = np.random.uniform(-weight_bound, weight_bound, size=(output_size, input_size))
        bias = np.random.uniform(-bias_bound, bias_bound, size=(output_size, 1))

        W = np.hstack([bias, W]) # (out, in+1)

        self.p_W = pmat.from_numpy(W)

        ############################################################
        # Initialize local blocks directly
        ############################################################

        # self.p_W = pmat(n=output_size, m=input_size+1) # +1 for bias 

        # if self.p_W.coords[1] == 0:
        #     # Only blocks with the first column have a bias column
        #     weight_local = np.random.uniform(-weight_bound, weight_bound, size=(self.p_W.n_loc, self.p_W.m_loc-1))
            
        #     bias_local = np.random.uniform(-bias_bound, bias_bound, size=(self.p_W.n_loc, 1))

        #     local = np.hstack([bias_local, weight_local])

        # else:
        #     # Every other block has no bias column
        #     local = np.random.uniform(-weight_bound, weight_bound, size=(self.p_W.n_loc, self.p_W.m_loc))


        # self.p_W._set_local(local) # (out, in+1)

        ############################################################

        self.in_features = input_size

        # Default is linear activation
        self.phi = linear
        self.phi_prime = linear_derivative

        
        self.p_dL_dW = None # indicate no gradient stored yet

        # Update-related variables

        betas = (0.9, 0.999)
        eps = 1e-8
        weight_decay = 0.0

        self.b1, self.b2 = betas
        self.epsilon = eps
        self.weight_decay = weight_decay

        # First and second mements
        self.m = pmat(self.p_W.shape[0], self.p_W.shape[1])
        self.v = pmat(self.p_W.shape[0], self.p_W.shape[1])
        self.t = 0  # time step
        ############################################################
       

    def _as_2d(self, p_X):

        # Transpose only if it matches (batch, in)
        if p_X.shape[0] != self.in_features:
            if p_X.ndim == 2 and p_X.shape[1] == self.in_features:
                p_X = p_X.T
            else:
                raise ValueError(f"Expected input with {self.in_features} rows, got {p_X.shape}")
        return p_X
        
    def _with_bias(self, p_X_no_bias):
        return p_X_no_bias.stack_ones_on_top()

       
    def forward(self, p_X_no_bias):
        ######
        # Since X_no_bias is the activation of a previous layer, which has no 
        # bias row, we need to add it here
        p_X_no_bias = self._as_2d(p_X_no_bias)      # (in, batch)
        self.p_X = self._with_bias(p_X_no_bias)     # (in+1, batch)

        
        self.p_A = self.p_W @ self.p_X      # (out, batch)
        self.p_H = self.phi(self.p_A)       # (out, batch)

        assert self.p_W.shape[1] == self.p_X.shape[0], (self.p_W.shape, self.p_X.shape)

        return self.p_H
    
    ############################################################################

    def backward(self, p_dL_dH_next):
        """
            dL_dh_next (out_size, batch_size): incoming gradient ∂L/∂h from the next layer. For the final layer, this is ∂L/∂h of that layer.

        Returns:
            dL_dh_prev (in_prev_size, batch_size): gradient to pass to previous layer.
        """
    
        p_dL_dA = p_dL_dH_next * self.phi_prime(self.p_A)
        assert p_dL_dA.shape == self.p_A.shape, (p_dL_dA.shape, self.p_A.shape)
        self.p_dL_dW = p_dL_dA @ self.p_X.T

        p_W_no_bias = self.p_W.remove_first_column()

        p_dL_dH_prev = p_W_no_bias.T @ p_dL_dA  # (in_prev, batch)

        return p_dL_dH_prev
    
    ############################################################################
        
    def update_weights(self, alpha=1e-3):
        if self.p_dL_dW is None:
            raise ValueError("No gradient stored. Cannot update weights.")


        """ Adam optimizer update """
        self.t += 1
        p_g = self.p_dL_dW

        if self.weight_decay != 0:
            p_g = p_g + self.weight_decay * self.p_W

        p_m = self.m
        p_v = self.v

        # first/second moments (momentum/velocity)
        p_m = self.b1 * p_m + (1 - self.b1) * p_g
        p_v = self.b2 * p_v + (1 - self.b2) * (p_g * p_g)

        # bias correction
        p_m_hat = p_m / (1 - self.b1 ** self.t)
        p_v_hat = p_v / (1 - self.b2 ** self.t)

        self.p_W -= alpha * p_m_hat * (1 / (np.sqrt(p_v_hat) + self.epsilon))

        self.p_dL_dW = None # Do not allow the same gradient to be used again

        """ 
            SGD update -- very slow (uncomment below and comment above to use
        """
        # self.p_W -= alpha * self.p_dL_dW
        # self.p_dL_dW = None # Do not allow the same gradient to be used again


    ############################################################################