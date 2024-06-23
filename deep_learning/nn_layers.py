import numpy as np
from . import no_resources
from . import node_funcs


# WEB LAYER
class Web:

    learnable = True
    """Weights layer to linearly transform activation values"""
    def __init__(self, output_shape, input_shape=None):
        
        self.output_layer = None

        self.input_shape = input_shape
        self.output_shape = output_shape

        self._weights = None
        self._bias = None

        self._input = None

    # INITIALIZE PARAMS

    def initialize_params(self):

        input_size = self.dim_size(self.input_shape)

        if isinstance(self.output_layer, (Activation, Loss)):
            if isinstance(self.output_layer.activation_func, (node_funcs.ReLU, node_funcs.LeakyReLU)): 
                factor = 2
            else:
                factor = 1
        else:
            factor = 1

        self._weights = np.random.normal(size=(self.input_shape, self.output_shape)) * np.sqrt(factor / input_size)
        self._bias = np.zeros(shape=self.output_shape)

    @staticmethod
    def dim_size(dims):
        if isinstance(dims, tuple):
            product = 1
            for dim in dims:
                product *= dim
        elif isinstance(dims, int):
            product = dims
        else:
            raise TypeError("Dims must be int or tuple")
        
        return product
    
    # PROPAGATE

    def advance(self, input, cache=True):
        """Move forward in the Neural net"""

        if cache: # don't save the input if the input layer
            self.input = input

        return input @ self._weights + self._bias

    def back_up(self, output_grad_to_loss, learning_rate, reg_strength, update_params_flag=True):
        
        if update_params_flag and self.input is not None:
            
            weights_grad_to_loss = self._calc_weights_grads(output_grad_to_loss, reg_strength=reg_strength)
            self._update_param(self._weights, weights_grad_to_loss, learning_rate)

            bias_grad_to_loss = self._calc_bias_grads(output_grad_to_loss)
            self._update_param(self._bias, bias_grad_to_loss, learning_rate)

        # discharge the input
        self.input = None
        
        input_grad_to_loss = output_grad_to_loss @ self._weights.T

        return input_grad_to_loss
    
    # CALCULATE GRADIENTS
    
    def _calc_weights_grads(self, output_grad_to_loss, reg_strength):

        m = len(self.input)

        if reg_strength != 0:
            weights_grad_to_loss = self.input.T @ (output_grad_to_loss / m) + 2 * reg_strength * self._weights
        else:
            weights_grad_to_loss = self.input.T @ (output_grad_to_loss / m)
        
        return weights_grad_to_loss
    
    def _calc_bias_grads(self, output_grad_to_loss):
        
        bias_grad_to_loss = output_grad_to_loss.mean(axis=0)

        return bias_grad_to_loss

    # UPDATE PARAMS

    def _update_param(self, param, grad, learning_rate):

        if isinstance(grad, no_resources.RowSparseArray):
            (learning_rate * grad).subtract_from_update(param)
        else:
            param -= (learning_rate * grad)
        
class Activation:

    learnable = False

    def __init__(self, activation_func):
        
        self.activation_func = activation_func

        self.input = None

    def advance(self, input, cache=True):

        if cache:
            self.input = input

        return self.activation_func.forward(input)
    
    def back_up(self, output_grad_to_loss):

        input_grad_to_output = self.activation_func.backward(self.input)

        input_grad_to_loss = output_grad_to_loss * input_grad_to_output

        return input_grad_to_loss
    
# LOSS LAYER

class Loss:
    
    def __init__(self, activation_func, loss_func):
        
        self.activation_func = activation_func
        self.loss_func = loss_func

        self.input = None
        self.output = None

        self.learnable = False
        
    def advance(self, input, cache=True):
        
        output = self.activation_func.forward(input)
        if cache:
            self.input = input 
            self.output = output

        return output
    
    def get_total_loss(self, y_pred, y_true):

        return self.loss_func.forward(y_pred, y_true)

    def get_cost(self, y_pred, y_true):

        cost = np.mean(self.get_total_loss(y_pred, y_true))

        # L2 regularization

        return cost

    def back_up(self, y_true):

        input_grad_to_loss = self.loss_func.backward(self.output, y_true)

        self.input = None
        self.output = None

        return input_grad_to_loss
    
# EFFICIENCY LAYERS

class Dropout:
    
    learnable = False

    def __init__(self, keep_prob, input_shape=None, seed=100):
        
        # validate the keep prob
        self._val_keep_prob(keep_prob)

        self._keep_prob = keep_prob
        self._seed = seed

        self._epoch_dropout_mask = None

        self.input_shape = input_shape

    @staticmethod
    def _val_keep_prob(keep_prob):
        if keep_prob <=0 or keep_prob >= 1:
            raise ValueError("keep prob must be between 0 and 1 exclusive")

    def set_epoch_dropout_mask(self, epoch):
        """Create masks for the epoch so masks can stay consistent throughout an epoch, 
        but also differ from epoch to epoch"""

        mask_rng = np.random.default_rng(self._seed+epoch)

        self._epoch_dropout_mask = (mask_rng.random(self.input_shape) < self._keep_prob).astype(int)

    def advance(self, input, cache=True):
        if cache:
            output = (input * self._epoch_dropout_mask) / self._keep_prob
            return output
        else:
            return input
        

    def back_up(self, output_grad_to_loss):
        
        input_grad_to_loss = (output_grad_to_loss * self._epoch_dropout_mask) / self._keep_prob

        return input_grad_to_loss
        

class BatchNorm:
    learnable = True

    pass



