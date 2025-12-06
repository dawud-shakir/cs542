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
    return probs - one_hot  # Shape: (batch_size, num_classes)

class Parallel_Layer:
    def __init__(self, input_size, output_size):
        
        self.in_features = input_size
         
        

        

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
        ############################################################
        # First/Second Moments

        # First comment:
        # self.m, self.v = np.zeros_like(self.W), np.zeros_like(self.W)  
        
        # Second comment:
        # self.m, self.v = np.zeros(self.W.shape), np.zeros(self.W.shape) 

        n, m = self.p_W.shape
        self.m, self.v = pmat(n, m), pmat(n, m)

        ############################################################
        self.t = 0

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
        p_X_no_bias = self._as_2d(p_X_no_bias)   # (in, batch)
        self.p_X = self._with_bias(p_X_no_bias) # (in+1, batch)

        # (out, batch)
        self.p_A = self.p_W @ self.p_X
        self.p_H = self.phi(self.p_A)

        assert self.p_W.shape[1] == self.p_X.shape[0], (self.p_W.shape, self.p_X.shape)

        return self.p_H
    
    def flops_forward(self, batch_size, activation=None):
        """
        - batch_size : B
        - activation  : 'relu' | 'linear' | None (approximate)
        """
        F = self.in_features
        # output rows = self.p_W.shape[0]
        O = self.p_W.shape[0]
        K = F + 1  # includes bias column
        # exact matmul: O * B * (2*K - 1)
        matmul = O * batch_size * (2 * K - 1)
        act_cost = 0
        if activation is None:
            # try to guess from stored phi
            name = getattr(self.phi, "__name__", "").lower()
            activation = name if name else "linear"
        if activation.startswith("relu"):
            act_cost = O * batch_size    # ~1 compare/op per element
        # small cost for stacking ones
        bias_stack = batch_size
        return matmul + act_cost + bias_stack
    
    #####################################################################################

    def backward(self, dL_dh_next):
        """
            dL_dh_next: incoming gradient ∂L/∂h from the next layer (shape: out, batch).
                        For the final layer, this is ∂L/∂h of that layer.

        Returns:
            dL_dh_prev: gradient to pass to previous layer (shape: in_prev, batch).
        """
        p_dL_dh_next = dL_dh_next

        p_dL_da = p_dL_dh_next * self.phi_prime(self.p_A)
        assert p_dL_da.shape == self.p_A.shape, (p_dL_da.shape, self.p_A.shape)
        self.p_dL_dW = p_dL_da @ self.p_X.T

        p_W_no_bias = self.p_W.remove_first_column()

        p_dL_dh_prev = p_W_no_bias.T @ p_dL_da  # (in_prev, batch)

        return p_dL_dh_prev

    def flops_backward(self, batch_size, activation=None):
        """
        - batch_size : B
        - activation : 'relu' | 'linear' | None (approximate; phi' cost ~1 op/element)
        Returns integer FLOP count.
        """
        F = self.in_features             # input features (no bias)
        O = self.p_W.shape[0]            # output features
        K = F + 1                        # includes bias column
        B = batch_size

        # elementwise cost: phi' and multiply (~1 op per element)
        elemwise = O * B

        # gradient wrt weights: (out, batch) @ (batch, K) -> out * K * (2*B - 1)
        dW_matmul = O * K * (2 * B - 1)

        # gradient to previous layer: (in, out) @ (out, batch) -> in * B * (2*O - 1)
        dprev_matmul = F * B * (2 * O - 1)

        return elemwise + dW_matmul + dprev_matmul
    
    #####################################################################################
        
    def update_weights(self, alpha=1e-3):
        
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

        # """ SGD update (comment the above to use) """
        # self.p_W -= alpha * self.p_dL_dW

        # self.p_dL_dW = None # Do not allow the same gradient to be used again

    def flops_update(self, include_weight_decay=False):
        """
        Estimate FLOPs for the Adam update implemented in update_weights().
        Counts per-weight-element operations (approximate):
         - p_m update: 3 ops/element
         - p_v update: 4 ops/element (includes p_g*p_g)
         - bias-correct (two divisions): 2 ops/element
         - final W update (sqrt, add eps, reciprocal, mult, mult, sub): ~6 ops/element
        Total per element (no weight_decay) ≈ 15 ops.
        If weight decay is enabled the gradient modification adds ~2 ops/element.
        A small fixed scalar cost is added to account for b1**t and b2**t.
        """
        n, m = self.p_W.shape   # n = out, m = in + 1
        elements = n * m

        per_elem = 15
        if include_weight_decay and self.weight_decay != 0:
            per_elem += 2

        # small scalar overhead for computing b1**t, b2**t and scalar subtractions/divisions
        scalar_overhead = 4

        return elements * per_elem + scalar_overhead

    #####################################################################################

    def flops_total(self, batch_size, activation=None, include_weight_decay=False):
        """Return total FLOPs for forward + backward + update for a given batch_size."""
        return (self.flops_forward(batch_size, activation=activation)
                + self.flops_backward(batch_size, activation=activation)
                + self.flops_update(include_weight_decay=include_weight_decay))