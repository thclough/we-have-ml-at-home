import numpy as np
import matplotlib.pyplot as plt
import node_funcs


## TODO
### Graph for loss vs epoch
### different types of regularization
### different types of loss
### learning rate schedulers

class NN:
    """Simple neural network class"""
    def __init__(self):
        # layers list 
        self.layers = []
        self.num_layers = 0
        self.loss=None

    def add_layer(self, layer_size: int, activation=None, loss=None):
        """Add layer to the neural network
        
        Args:
            layer_size (int) : size of the layer
            activation (function) : vectorized function to use as activation function
            loss (function) : loss function for the 
        """
        # Validation
        if self.loss != None:
            raise Exception("Last layer in network already declared given loss function exists")

        ## for loss
        if loss == node_funcs.BCE:
            if layer_size != 1:
                raise Exception("Should use output layer of size 1 when using binary cross entropy loss,\
                                decrease layer size to 1 or use CE (regular cross entropy)")

        if loss == node_funcs.CE:
            if layer_size < 2:
                raise Exception("Should use cross entropy loss for multi-class classification, increase layer-size or use BCE")
        
        ## for first layer
        if len(self.layers) == 0:
            if activation != None:
                raise Exception("First layer should not have an activation function")
            if loss:
                raise Exception("First layer should not have a loss function")

        # append layer to the layers list
        self.layers.append({"size": layer_size, "activation": activation, "loss":loss})

        # increment the amount of layers
        self.num_layers += 1

        # declare loss for the loss function
        self.loss = loss

    def fit(self, 
            X_train, y_train, 
            X_dev=None, y_dev=None,
            batch_size=None, 
            learning_rate=1, 
            reg_strength=0.0001,
            num_epochs=30,
            verbose = True,
            display = True):
        """Fits the neural network to the data.
        
        Args:
            X_train (numpy array) : training examples
            y_train (numpy array) : training labels
            X_dev (numpy array) : dev examples
            y_dev (numpy array) : dev labels
            batch_size (int) : size of the batches
            learning_rate (numeric) : learning rate for gradient descent
            reg_strength (numeric) : multiplier for regularization in gradient descent
            num_epochs (int) : number of cycles through the training data
            verbose (bool, default = True) : whether or not to print training loss after each epoch
            display (bool, default = True) : whether or not to plot training (and dev) average loss after each epoch
        """
        # attributes
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.reg_strength = reg_strength
        self.dev_flag = False # flag if there is a dev set

        # validate inputs
        if self.num_layers == 0:
            raise Exception("No layers in network")

        if self.loss == None:
            raise Exception("Please declare a loss function for a layer")

        if (X_dev is not None) or (y_dev is not None):
            if not (X_dev is not None and y_dev is not None):
                raise Exception("Please input both X_dev and y_dev")
            else:
                self.dev_flag = True

        n_train,_ = X_train.shape

        # set batch size to the size of the whole training set if none passed
        if batch_size is None:
            self.batch_size = n_train

        # get the initial weights and biases
        self.get_initial_params()

        # initialize lists for
        self.cost_train = []
        self.cost_dev = []
        
        # set up loss plot
        if display:
            fig, ax = plt.subplots()
            ax.set_title(f"Average Loss ({self.loss.name}) vs. Epoch")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Average Loss")

        # go through the epochs
        for epoch in range(num_epochs):
            self.gradient_descent_epoch(X_train, y_train)

            # calculate avg loss for dev and test sets
            self.cost_train.append(self.avg_loss(X_train, y_train))
            if self.dev_flag:
                self.cost_dev.append(self.avg_loss(X_dev, y_dev))

            if verbose:
                print(f"Training Avg Loss: {self.cost_train[-1]}")
                
            if display:
                # epoch in loop indexed at 0, add 1 to start indexing at 1
                true_epoch = epoch+1
                self.update_training_plot(fig, ax, true_epoch, num_epochs)

    def get_initial_params(self):
        """Create the initial parameters dictionary for the neural network starting at idx 1
        
        Populates self.params dictionary with W (weights) and b (biases) in the form of numpy arrays
        Weights and biases numbers correspond to their output layer number
        ex. W1 and b1 are weights and biases used to calculate node precursors in layer 2
        
        W dimensions are (input layer size x output layer size)
        b is one dimensional the size (output layer size)
        """
        self.params = {}

        for layer_idx in range(1,self.num_layers):
            input_size = self.layers[layer_idx - 1]["size"]
            output_size = self.layers[layer_idx]["size"]
            
            self.params[f"W{layer_idx}"] = np.random.normal(size=(input_size, output_size))
            self.params[f"b{layer_idx}"] = np.zeros(shape=output_size)

    def gradient_descent_epoch(self, X_train, y_train):
        """Performs one epoch of gradient descent to update the params dictionary
        
        Args:
            X_train (numpy array) : training examples (num_examples x num_features)
            y_train (numpy array) : training array (num_examples x 1)
        
        """
        n = len(X_train)

        # go through the batches for ea
        for start_idx in range(0,n-1,self.batch_size):
            end_idx = min(start_idx + self.batch_size, n)
            
            # locate relevant fields to 
            X_train_batch = X_train[start_idx:end_idx]
            y_train_batch = y_train[start_idx:end_idx]

            # forward pass
            node_vals_batch = self.forward_prop(X_train_batch)

            # perform back prop to obtain gradients
            grad_dict = self.backward_prop(X_train_batch, y_train_batch, node_vals_batch)

            # update params
            for param in self.params:
                self.params[param] = self.params[param] - (self.learning_rate * grad_dict[param])
        
    def forward_prop(self, X_train):
        """Perform forward pass in neural net. 
        
        Args: 
            X_train (numpy array) : training examples (num_examples x num_features)
        
        Returns:
            za_vals (dictionary) : dictionary of node precursors and activation values
                where "an" or "an" would correspond to the nth layer indexed at 0
        """

        # to hold activation values (a)
        za_vals = {}

        # set the data as "a0"
        za_vals["a0"] = X_train

        # go through the layers and save the activations
        for layer in range(self.num_layers-1):
            web_idx = layer + 1

            activation_func = self.layers[web_idx]["activation"]
            W = self.params[f"W{web_idx}"]
            b = self.params[f"b{web_idx}"]
            a_behind = za_vals[f"a{layer}"]

            za_vals[f"z{web_idx}"] = a_behind @ W + b
            za_vals[f"a{web_idx}"] = activation_func.forward(za_vals[f"z{web_idx}"])

            # za_vals[f"z{web_idx}"] = a_behind.dot(W) + b
            # za_vals[f"a{web_idx}"] = activation_func.forward(za_vals[f"z{web_idx}"])
        
        return za_vals

    def avg_loss(self, X, y):
        """Calculate the average loss depending on the loss function
        
        Args:
            X (numpy array) : examples to predict on (num_examples x num_features)
            y (numpy array) : true labels (num_examples x 1)
        
        """
        # calculate activation values for each layer (includes predicted values)
        za_vals = self.forward_prop(X)

        # gather predicted values
        y_pred = za_vals[f"a{self.num_layers-1}"]
        # return avg of the losses
        return np.mean(self.loss.forward(y_pred, y))
        
    def backward_prop(self, X, y, za_vals):
        """perform backprop
        
        Args:
            X (numpy array) : examples to predict on (num_examples x num_features)
            y (numpy array) : true labels (num_examples x 1)
            za_vals (dictionary) : dictionary of node precursors and activation values
                where "an" or "an" would correspond to the nth layer indexed at 0 
        """

        # get number of training examples
        n = len(X)

        grad_dict = {}
        # go through the layers backwards
        for layer in range(self.num_layers-1,0,-1):
            
            # get the node precursors (z's) and activation values
            a_behind = za_vals[f"a{layer-1}"]
            a_ahead = za_vals[f"a{layer}"]
            z_ahead = za_vals[f"z{layer}"]
            activation = self.layers[layer]["activation"]

            W = self.params[f"W{layer}"]

            # W_ahead = dJ_dz_past = 1
            
            # find dJ/dz for the layer
            if layer == self.num_layers-1:
                dJ_dz_ahead = self.loss.backward(a_ahead, y)
            else: # move gradient backward
                dJ_da_ahead = dJ_dz_past @ W_ahead.T
                da_dz_ahead = activation.backward(z_ahead)
                dJ_dz_ahead = dJ_da_ahead * da_dz_ahead

            grad_dict[f"W{layer}"] = (1/n) * (a_behind.T @ dJ_dz_ahead) + 2 * self.reg_strength * W
            grad_dict[f"b{layer}"] = (dJ_dz_ahead).mean(axis=0)

            # save dJ_dz for ahead for next back step
            dJ_dz_past = dJ_dz_ahead
            W_ahead = W

        return grad_dict

    def update_training_plot(self, fig, ax, epoch, num_epochs):
        """Updates training plot to display average losses

        Args:
            fig (matplotlib.pyplot.figure) : figure containing plot axis
            ax (matplotlib.pyplot.axis) : axis that contains line plots
            epoch (int) : epoch number to graph new data for
            num_epochs (int) : total number of epochs
        """
        
        if epoch == 1:
            self.train_line, = ax.plot(range(1,epoch+1), self.cost_train, label="Average training loss")
            if self.dev_flag:
                self.dev_line, = ax.plot(range(1, epoch+1), self.cost_dev, label="Average dev loss")
            ax.legend()
        else:
            self.train_line.set_data(range(1,epoch+1), self.cost_train)
            if self.dev_flag:
                self.dev_line.set_data(range(1,epoch+1), self.cost_dev)

        max_val = np.max(np.concatenate([self.cost_dev, self.cost_train]))
        ax.set(xlim=[0,num_epochs], ylim=[0,max_val])
        plt.pause(.1)

    def evaluate(self, X_test, y_test):
        # calculate activation values for each layer (includes predicted values)
        za_vals = self.forward_prop(X_test)

        # gather predicted values
        y_pred = za_vals[f"a{self.num_layers-1}"] > .5
        accuracy = (np.argmax(y_pred,axis=1) == np.argmax(y_test,axis=1)).sum() * (1. / y_test.shape[0])

        return accuracy
    

