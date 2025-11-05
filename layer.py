"""
layer.py
"""
from math import log
from venv import create
from mpi4py import MPI
from pmat import pmat, create_grid_comm


import numpy as np

# Activations
def ReLU(z): return np.maximum(0.0, z)        # (out, batch)               
def ReLU_derivative(z): return (z > 0).astype(float)    # (out, batch)
def linear(z): return z                             # (features, batch)
def linear_derivative(z): return np.ones_like(z)    # (features, batch)

def mean_squared_error(y, y_hat):
    return np.mean((y_hat - y) ** 2)

def mean_squared_error_derivative(y, y_hat):
    # d/dy_hat of MSE with mean over all elements
    return (2.0 / y.size) * (y_hat - y)

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
    return probs - one_hot  # Shape: (batch_size, num_classes)

class Parallel_Layer:
    def __init__(self, input_size, output_size):
        
        self.in_features = input_size
         
        self.W = pmat(n=output_size, m=input_size+1) # +1 for bias 




        # Weights: Uniform initialization (Xavier)
        weight_bound = np.sqrt(6.0 / (input_size + output_size))
        
        # Bias column: Uniform(-1/sqrt(fan_in), +1/sqrt(fan_in))
        bias_bound = 1.0 / np.sqrt(input_size)

        # Only W[:,0] has bias column
        if self.W.coords[1] == 0:
            weight_local = np.random.uniform(-weight_bound, weight_bound, size=(self.W.n_loc, self.W.m_loc-1))
            
            bias_local = np.random.uniform(-bias_bound, bias_bound, size=(self.W.n_loc, 1))

            local = np.hstack([bias_local, weight_local])

        else:
            local = np.random.uniform(-weight_bound, weight_bound, size=(self.W.n_loc, self.W.m_loc))


        



        # print(f"global shape = {self.W.shape}")

        # print(f"local coord = {self.W.coords}")
        # print(f"local shape = {self.W.local.shape}")
        # print(f"local padding = {(self.W.n_pad, self.W.m_pad)}")

        # Concatenate bias and weights horizontally
        self.W._set_local(local) # (out, in+1)
        # exit()

        # # Weights: Uniform initialization (Xavier)
        # weight_bound = np.sqrt(6.0 / (input_size + output_size))
        # W_no_bias = np.random.uniform(-weight_bound, weight_bound, size=(output_size, input_size))

        # # Weights Bias column: Uniform(-1/sqrt(fan_in), +1/sqrt(fan_in))
        # bias_bound = 1.0 / np.sqrt(input_size)
        # W_bias_col = np.random.uniform(-bias_bound, bias_bound, size=(output_size, 1))

        # # Stack [bias | weights]
        # self.W = np.hstack([W_bias_col, W_no_bias]) # (out, in+1)

        # Default is linear activation
        self.phi = linear
        self.phi_prime = linear_derivative

        betas=(0.9, 0.999)
        eps=1e-8
        weight_decay=0.0

        self.b1, self.b2 = betas
        self.epsilon = eps
        self.weight_decay = weight_decay
        # self.m, self.v = np.zeros_like(self.W), np.zeros_like(self.W)  # first/second moments
        ############################################################
        self.m, self.v = np.zeros(self.W.shape), np.zeros(self.W.shape)

        ############################################################
        self.t = 0

    def _as_2d(self, X):
        #### Reshape not implemented in pmat yet
        if X.ndim == 1:
            return X.reshape(-1, 1)             # (in, 1)

        p_X = pmat.from_numpy(X).get_full()
        # Enforce (in, batch); transpose only if it exactly matches (batch, in)
        if p_X.shape[0] != self.in_features:
            if p_X.ndim == 2 and p_X.shape[1] == self.in_features:
                p_X = p_X.T
            else:
                raise ValueError(f"Expected input with {self.in_features} rows, got {p_X.shape}")
        return p_X
        # X = pmat.from_numpy(p_X, grid_comm).get_full()
        # return X

        # X = np.asarray(X)
        # if X.ndim == 1:
        #     X = X.reshape(-1, 1)             # (in, 1)
        # # Enforce (in, batch); transpose only if it exactly matches (batch, in)
        # if X.shape[0] != self.in_features:
        #     if X.ndim == 2 and X.shape[1] == self.in_features:
        #         X = X.T
        #     else:
        #         raise ValueError(f"Expected input with {self.in_features} rows, got {X.shape}")
        # return X

    def _with_bias(self, X_no_bias):
        # Input's first row is the bias row (1s)
        ones = np.ones((1, X_no_bias.shape[1])) 
        return np.vstack([ones, X_no_bias])

       
    def forward(self, X_no_bias):
        # X_no_bias is activations from previous layer (no bias row)
        X_no_bias = self._as_2d(X_no_bias)   # (in, batch)

        # (in+1, batch)
        p_X = pmat.from_numpy(self._with_bias(X_no_bias))

        # (out, batch)
        p_a = self.W @ p_X
        p_h = self.phi(p_a)

        assert self.W.shape[1] == p_X.shape[0], (self.W.shape, p_X.shape)

        self.X = p_X.get_full()  # (in+1, batch)
        self.a = p_a.get_full()  # (out, batch)
        self.h = p_h.get_full()  # (out, batch)

        return self.h

    def backward(self, dL_dh_next):
        p_a = pmat.from_numpy(self.a)
        p_X = pmat.from_numpy(self.X)
        p_dL_dh_next = pmat.from_numpy(dL_dh_next)

        p_dL_da = p_dL_dh_next * self.phi_prime(p_a)
        assert p_dL_da.shape == p_a.shape, (p_dL_da.shape, p_a.shape)
        p_dL_dW = p_dL_da @ p_X.T

        p_W_no_bias = pmat.resize(0, self.W.n, 1, self.W.m, self.W)
        p_dL_dh_prev = p_W_no_bias.T @ p_dL_da  # (in_prev, batch)

        self.dL_dW = p_dL_dW.get_full()
        return p_dL_dh_prev.get_full()

        """
        Backprop through this layer.

        Args:
            dL_dh_next: incoming gradient ∂L/∂h from the next layer (shape: out, batch).
                        For the final layer, this is ∂L/∂h of that layer.

        Returns:
            dL_dh_prev: gradient to pass to previous layer (shape: in_prev, batch).
        """


        # # Local gradient wrt pre-activation a
        # dL_da = dL_dh_next * self.phi_prime(self.a)          # (out, batch)
        # assert dL_da.shape == self.a.shape, (dL_da.shape, self.a.shape)

        # # Weight (incl. bias) gradient; X must have a leading column of 1s
        # self.dL_dW = dL_da @ self.X.T                        # (out, in_prev+1)

        # # Pass gradient back (exclude bias weights)
        # W_no_bias = self.W[:, 1:]                            # (out, in_prev)
        # dL_dh_prev = W_no_bias.T @ dL_da                     # (in_prev, batch)
        # return dL_dh_prev

        
    def update_weights(self, alpha=1e-3):
        p_dL_dW = pmat.from_numpy(self.dL_dW)

        """ Adam optimizer update """
        self.t += 1
        p_g = p_dL_dW
        if self.weight_decay != 0:
            p_g = p_g + self.weight_decay * self.W

        p_m = pmat.from_numpy(self.m)
        p_v = pmat.from_numpy(self.v)

        # first/second moments (momentum/velocity)
        p_m = self.b1 * p_m + (1 - self.b1) * p_g
        p_v = self.b2 * p_v + (1 - self.b2) * (p_g * p_g)

        # bias correction
        p_m_hat = p_m / (1 - self.b1 ** self.t)
        p_v_hat = p_v / (1 - self.b2 ** self.t)

        self.W -= alpha * p_m_hat * (1 / (np.sqrt(p_v_hat) + self.epsilon))

        self.m = p_m.get_full()
        self.v = p_v.get_full()

        self.dL_dW = None # Do not allow the same gradient to be used again


        # ### Original ##################################################
        # """ Adam optimizer update """
        # self.t += 1
        # g = self.dL_dW
        # if self.weight_decay != 0:
        #     g = g + self.weight_decay * self.W

        # # m = np.zeros_like(self.W)
        # # v = np.zeros_like(self.W)

        # m, v = self.m, self.v

        #  # first/second moments (momentum/velocity)
        # m = self.b1 * m + (1 - self.b1) * g
        # v = self.b2 * v + (1 - self.b2) * (g * g)

        # # bias correction
        # m_hat = m / (1 - self.b1 ** self.t)
        # v_hat = v / (1 - self.b2 ** self.t)

        # self.W -= alpha * m_hat / (np.sqrt(v_hat) + self.epsilon)

        # self.m, self.v = m, v
        
        # """ SGD update (comment the above to use) """
        # # self.W -= alpha * self.dL_dW

        # self.dL_dW = None # Do not allow the same gradient to be used again