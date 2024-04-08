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
        return x > 0


## LOSS FUNCTIONS

class BCE:
    """binary cross entropy"""
    @staticmethod
    def forward(y_pred,y_true):
        """
        Args:
            y_pred (vector): output probabilities
            y_true (vector): ground_truth labels

        Returns:
            (vector) : binary cross entropy
        """
        return -(y_true * np.log(y_pred) + (1-y_true) * np.log(1-y_pred))
    
    def backward(y_pred,y_true):
        """Compute the gradient with respect to linear combo (z's)
        
        Args: 
            y_pred (vector): output probabilities
            y_true (vector): ground_truth labels

        ReturnsL
            (vector) : Jacobian wrt linear combination
        """

        return y_pred - y_true