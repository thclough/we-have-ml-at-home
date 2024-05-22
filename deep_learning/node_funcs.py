import numpy as np

## ACTIVATION FUNCTION CLASSES
class Sigmoid:
    
    @staticmethod
    def forward(x):
        """compute the sigmoid for the input x

        Args:
            x: A numpy float array

        Returns: 
            A numpy float array containing the sigmoid of the input
        """
        return 1 / (1 + np.exp(-x))
    
    @staticmethod
    def backward(x):
        """Gradient of the sigmoid function evaluated at a vector x
        calculates Jacobian of "a" with respect to "z" in NN
        
        Args:
            x (vector) : input vector
        
        Returns:
            (vector) : Jacobian wrt x
        
        """
        return Sigmoid.forward(x) * (1 - Sigmoid.forward(x))

class Identity:

    @staticmethod
    def forward(x):
        """Identity function, the output is the input
    
        Args:
            x (numeric, or vector) : the input

        Returns:
            x (numeric, or vector) : the input
        """
        return x
    
    @staticmethod
    def backward(x):
        """Calculate Jacobian of the identity function wrt x
        
        Args:
            x (vector) : input vector

        Returns:
            (vector) : Jacobian wrt x, just a vector of ones
        """
        return np.ones(x.shape)

class ReLU:

    @staticmethod
    def forward(x):
        """Rectified linear unit: max(0,input)
        
        Args:
            x (vector): input vector of ReLU
        
        Returns:
            (vector) : ReLU output
        
        """
        return np.maximum(0,x)
    
    @staticmethod
    def backward(x):
        """Jacobian of ReLU
        
        Args:
            x (vector) : input of ReLU
        
        Returns:
            (vector) : Jacobian of the ReLU
        """
        return x >= 0
    
class LeakyReLU:

    @staticmethod
    def forward(x):
        """Leaky rectified linear unit: max(0.1*input,input)
        
        Args:
            x (vector): input vector ofLeaky ReLU
        
        Returns:
            (vector) : Leaky ReLU output
        
        """
        return np.maximum(0.1*x,x)
    
    @staticmethod
    def backward(x):
        """Jacobian of leaky ReLU
        
        Args:
            x (vector) : input of Leaky ReLU
        
        Returns:
            (vector) : Jacobian of the Leaky ReLU
        """
        return np.where(x < 0, 0.1, 1)
    
class Softmax:

    @staticmethod
    def forward(x):
        # x_shifted = x - np.max(x, axis=1, keepdims=True)
    
        # exp_mat = np.exp(x_shifted)

        # sm_mat = exp_mat / np.sum(exp_mat, axis=1, keepdims=True)

        # return sm_mat
        x = x - np.max(x,axis=1)[:,np.newaxis]
        exp = np.exp(x)
        s = exp / np.sum(exp,axis=1)[:,np.newaxis]
        return s

    @staticmethod
    def backward(x):
        return Softmax.forward(x) * (1-Softmax.forward(x))

## LOSS FUNCTIONS

class BCE: 
    name = "Binary Cross Entropy"
    """binary cross entropy for binary classification"""
    @staticmethod
    def forward(y_pred,y_true):
        """
        Args:
            y_pred (vector): output probabilities
            y_true (vector): ground_truth labels

        Returns:
            (vector) : binary cross entropy loss
        """
        return -(y_true * np.log(y_pred) + (1-y_true) * np.log(1-y_pred))
    
    @staticmethod
    def backward(y_pred, y_true):
        """Compute the gradient with respect to linear combo (z's)
        
        Args: 
            y_pred (vector): output probabilities
            y_true (vector): ground_truth labels

        Returns:
            (vector) : Jacobian wrt linear combination
        """
        return y_pred - y_true

class WeightedBCE(BCE):
    """For imbalanced dataset"""
    name = "Weighted Binary Cross Entropy"
    
    def __init__(self, prop_neg):
        """
        Args:
            prop_neg (float) : proportion of examples that are negative
        """
        self._weight_neg = .5 / prop_neg
        self._weight_pos = .5 / (1-prop_neg)

    def forward(self, y_pred, y_true):
        """
        Args:
            y_pred (vector): output probabilities
            y_true (vector): ground truth labels

        Returns:
            (vector) : binary cross entropy
        """
        weight_vector = self._weight_pos * y_true + self._weight_neg * (1-y_true)
        return weight_vector * super().forward(y_pred, y_true)

    def backward(self, y_pred, y_true):
        """

        Args: 
            y_pred (vector): output probabilities
            y_true (vector): ground truth labels

        Returns:
            (vector) : Jacobian wrt linear combination
        """
        weight_vector = self._weight_pos * y_true + self._weight_neg * (1-y_true)
        return weight_vector * super().backward(y_pred, y_true)
    
    def _get_weight_vector(self, y_true):
        """Obtain the correct weight vector for forward and backward
        
        Args:
            y_true (vector) : ground truth labels

        Returns:
            (vector) : weight vector with same dimensions as y_true
        """
        return self._weight_pos * y_true + self._weight_neg * (1-y_true)

class CE:
    name = "Cross Entropy"
    """regular cross entropy for multi-class classification"""
    @staticmethod
    def forward(y_pred, y_true):
        """y_true is a sparse array"""

        # can use log1p because monotonically increases
        return -np.sum(y_true * np.log1p(y_pred), axis=1)
    
    @staticmethod
    def backward(y_pred, y_true):
        return y_pred - y_true 