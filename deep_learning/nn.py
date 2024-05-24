import numpy as np
import matplotlib.pyplot as plt
#from we_have_ml_at_home.deep_learning import node_funcs
import node_funcs
import time
import joblib
import no_resources

# TODO
## Superchunk epoch costs and display
## Weighting BCE and derivative for imbalanced dataset
## Estimating time to completion 

## different types of regularization
## different types of loss
## learning rate schedulers/optimizers

class SmoothNN:
    """Simple neural network class"""
    def __init__(self):
        # layers list 
        self.layers = []
        self.num_layers = 0
        self.loss=None

    def add_layer(self, layer_size, activation=None, loss=None):
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
            verbose=True,
            display=True):
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
        self._dev_flag = False # flag if there is a dev set

        # validate inputs def validate_structure
        self._val_structure()

        if (X_dev is not None) or (y_dev is not None):
            if not (X_dev is not None and y_dev is not None):
                raise Exception("Please input both X_dev and y_dev")
            else:
                self._dev_flag = True

        n_train,_ = X_train.shape

        # set batch size to the size of the whole training set if none passed
        if batch_size is None:
            self.batch_size = n_train

        # get the initial weights and biases
        self._get_initial_params() # use current params

        # initialize lists for costs
        self._train_costs = []
        self._dev_costs = []
        
        # set up loss plot
        if display:
            fig, ax = plt.subplots()
            ax.set_title(f"Average Loss ({self.loss.name}) vs. Epoch")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Average Loss")

        # go through the epochs
        for epoch in range(num_epochs):
            self._gradient_descent_epoch(X_train, y_train)
            # calculate avg loss for dev and test sets
            self._train_costs.append(self.avg_loss(X_train, y_train))
            if self._dev_flag:
                self._dev_costs.append(self.avg_loss(X_dev, y_dev))

            if verbose:
                print(f"Training Avg Loss: {self._train_costs[-1]}")
                
            if display:
                # epoch in loop indexed at 0, add 1 to start indexing at 1
                true_epoch = epoch+1
                self._update_epoch_plot(fig, ax, true_epoch, num_epochs)

    def _val_structure(self):
        """validate the structure of the the NN"""
        if self.num_layers == 0:
            raise Exception("No layers in network")

        if self.loss == None:
            raise Exception("Please add a loss function")
        
    def _get_initial_params(self):
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

    def _gradient_descent_epoch(self, X_train, y_train):
        """Performs one epoch of gradient descent to update the params dictionary
        
        Args:
            X_train (numpy array) : training examples (num_examples x num_features)
            y_train (numpy array) : training array (num_examples x 1)
        
        """
        n = len(X_train)

        start_train_time = time.time()
        # go through the batches
        for start_idx in range(0,n-1,self.batch_size):
            end_idx = min(start_idx + self.batch_size, n)
            
            # locate relevant fields to 
            X_train_batch = X_train[start_idx:end_idx]
            y_train_batch = y_train[start_idx:end_idx]

            # create batch_total_pass, just return this
            self._batch_total_pass(X_train_batch, y_train_batch)

            cur_train_time = time.time()
            #print(cur_train_time-start_train_time)

            start_train_time = cur_train_time

    def _batch_total_pass(self, X_train_batch, y_train_batch):
        """forward propagation, backward propagation and parameter updates for """
        time3 = time.time()
        # forward pass
        node_vals_batch = self._forward_prop(X_train_batch)
        time4 = time.time()
        # print(f"Time for forward: {time4-time3}")

        # perform back prop to obtain gradients
        grad_dict = self._backward_prop(X_train_batch, y_train_batch, node_vals_batch)
        time5 = time.time()
        print(f"time for backwards {time5-time4}")

        # update params
        for param in self.params:

            gradient = grad_dict[param]

            if isinstance(gradient, no_resources.RowSparseArray):
                (self.learning_rate * gradient).subtract_from_update(self.params[param])
            else:
                self.params[param] = self.params[param] - (self.learning_rate * gradient)
        time6 = time.time()
        print(f"Time for params update {time6-time5}")
        
    def _forward_prop(self, X_train):
        """Perform forward pass in neural net while storing intermediaries. 
        
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
    
        return za_vals

    def _backward_prop(self, X, y, za_vals):
        """perform backprop
        
        Args:
            X (numpy array) : examples to predict on (num_examples x num_features)
            y (numpy array) : true labels (num_examples x 1)
            za_vals (dictionary) : dictionary of node precursors and activation values
                where "an" or "an" would correspond to the nth layer indexed at 0 
        """
        print("Enter Backprop")
        # get number of training examples
        n = len(X)

        grad_dict = {}
        # go through the layers backwards
        for layer in range(self.num_layers-1,0,-1):

            time0_1 = time.time()
            # get the node precursors (z's) and activation values
            a_behind = za_vals[f"a{layer-1}"]
            a_ahead = za_vals[f"a{layer}"]
            z_ahead = za_vals[f"z{layer}"]
            activation = self.layers[layer]["activation"]

            W = self.params[f"W{layer}"]
            
            # find dJ/dz for the layer
            if layer == self.num_layers-1:
                dJ_dz_ahead = self.loss.backward(a_ahead, y)
            else: # move gradient backward
                dJ_da_ahead = dJ_dz_past @ W_ahead.T
                da_dz_ahead = activation.backward(z_ahead)
                dJ_dz_ahead = dJ_da_ahead * da_dz_ahead

            time0_3 = time.time()

            # cases for efficiency, for regularization, operating on large W matrix (even with reg strength of 0)
            # can be very time expensive
            if self.reg_strength == 0:
                grad_dict[f"W{layer}"] = a_behind.T @ (dJ_dz_ahead * (1/n)) 
            else:
                grad_dict[f"W{layer}"] = a_behind.T @ (dJ_dz_ahead * (1/n)) + 2 * self.reg_strength * W

            grad_dict[f"b{layer}"] = dJ_dz_ahead.mean(axis=0)

            time0_4 = time.time()
            print(f"Time for grad dict calc {time0_4-time0_3}")

            # save dJ_dz for ahead for next back step
            dJ_dz_past = dJ_dz_ahead
            W_ahead = W

        return grad_dict
    
    def predict_prob(self, X):
        """Obtain output layer activations
        
        Args: 
            X (numpy array) : examples (num_examples x num_features)
        
        Returns:
            cur_a (numpy array) : probabilities of output layer
        """

        # set the data as "a0"
        a_behind = X

        # go through the layers and save the activations
        for layer in range(self.num_layers-1):
            web_idx = layer + 1

            activation_func = self.layers[web_idx]["activation"]
            W = self.params[f"W{web_idx}"]
            b = self.params[f"b{web_idx}"]

            cur_z = a_behind @ W + b
            cur_a = activation_func.forward(cur_z)

            a_behind = cur_a

        return cur_a
    
    def avg_loss(self, X, y):
        """Calculate the average loss depending on the loss function
        
        Args:
            X (numpy array) : examples to predict on (num_examples x num_features)
            y (numpy array) : true labels (num_examples x 1)
        
        """
        # calculate activation values for each layer (includes predicted values)
        y_pred = self.predict_prob(X)

        # return avg of the losses
        return np.mean(self.loss.forward(y_pred, y))

    def _update_epoch_plot(self, fig, ax, epoch, num_epochs, epoch_gap=1):
        """Updates training plot to display average losses

        Args:
            fig (matplotlib.pyplot.figure) : figure containing plot axis
            ax (matplotlib.pyplot.axis) : axis that contains line plots
            epoch (int) : epoch number to graph new data for
            num_epochs (int) : total number of epochs
        """
        if epoch == 1:
            self.train_line, = ax.plot(range(1,2), self._train_costs, label="Average training loss")
            if self._dev_flag:
                self.dev_line, = ax.plot(range(1, 2), self._dev_costs, label="Average dev loss")
            ax.legend()
        else:
            self.train_line.set_data(range(1,epoch+1, epoch_gap), self._train_costs)
            if self._dev_flag:
                self.dev_line.set_data(range(1,epoch+1, epoch_gap), self._dev_costs)

        max_val = np.max(np.concatenate([self._dev_costs, self._train_costs]))
        ax.set(xlim=[0,num_epochs], ylim=[min(0, max_val*2),max(0, max_val*2)])
        plt.pause(.2)

    def predict_labels(self, X):
        """Predict labels of given X examples
        
        Args:
            X (numpy array) : array of examples by row

        Returns:
            predictions (numpy array) : array of prediction for each example by row
        
        """
        # calculate activation values for each layer (includes predicted values)
        final_activations = self.predict_prob(X)

        if self.loss == node_funcs.BCE:
            predictions = final_activations > .5
        else:
            predictions = np.argmax(final_activations, axis=1)

        return predictions
    
    def accuracy(self, X_test, y_test):
        """Calculate accuracy of inference for given examples and their ground truths
        
        Args:
            X_test (numpy array) : array of examples
            y_test (numpy array) : array of ground truth labels in sparse form

        Returns:
            accuracy (float) : accuracy of prediction
        
        """
        predictions = self.predict_labels(X_test)
        
        if self.loss == node_funcs.BCE:
            accuracy = (predictions == y_test).sum() * (1. / y_test.shape[0])
        else:
            accuracy = (predictions == np.argmax(y_test,axis=1)).sum() * (1. / y_test.shape[0])

        return accuracy
    
class ChunkNN(SmoothNN):
    """NN class for chunking data from separate locations using no_resources.Chunk"""
    def fit(self,
            train_chunk,
            dev_chunk=None,
            learning_rate=1, 
            reg_strength=0.0001,
            num_epochs=30,
            epoch_gap=5,
            batch_prob=.01,
            param_path=None,
            verbose=True, 
            display=True):
        
        self.learning_rate = learning_rate
        self.reg_strength = reg_strength
        self._dev_flag = False # flag if there is a dev set

        if dev_chunk:
            self._dev_flag = True

        if not train_chunk._train_chunk:
            raise Exception("Given train chunk must be a valid train chunk (_train_chunk attribute set to True)")

        # validate structure
        super()._val_structure()

        # get initial params
        super()._get_initial_params()

        # initialize lists for costs
        self._train_costs = []
        self._dev_costs = []

        if display:
            fig, ax = plt.subplots()
            ax.set_title(f"Average Loss ({self.loss.name}) vs. Epoch Gap")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Average Loss")

        for epoch in range(num_epochs):
            print(f"Epoch: {epoch}")
            time_0 = time.time()
            for X_train, y_train in train_chunk.generate():

                time_1 = time.time()
                #print(f"\n Time for data gen: {time_1-time_0}")
                super()._batch_total_pass(X_train, y_train)
                time_2 = time.time()
                print(f"Time for batch pass {time_2 - time_1}")

                if verbose:
                    if np.random.binomial(1, batch_prob):
                        sampled_batch_loss = super().avg_loss(X_train, y_train)

                        print(f"\t Sampled batch loss: {sampled_batch_loss}")

                time_0 = time.time()

            if param_path:
                joblib.dump(self.params,param_path)

            # record costs after each epoch gap
            if epoch % epoch_gap == 0:
                epoch_train_cost = self.chunk_loss(train_chunk)
                self._train_costs.append(epoch_train_cost)
                if verbose:
                    print(f"\t Training cost: {epoch_train_cost}")

                if self._dev_flag:
                    epoch_dev_cost = self.chunk_loss(dev_chunk)
                    self._dev_costs.append(epoch_dev_cost)
                    if verbose:
                        print(f"\t Dev cost: {epoch_dev_cost}")

                if display:
                    true_epoch = epoch + 1
                    super()._update_epoch_plot(fig, ax, true_epoch, num_epochs, epoch_gap)
                
    def chunk_loss(self, eval_chunk):
        """"Calculate loss on the eval chunk"""
        
        loss_sum = 0
        length = 0

        for X_data, y_data in eval_chunk.generate():
            y_probs = super().predict_prob(X_data)

            chunk_loss_sum = np.sum(self.loss.forward(y_probs, y_data))
            chunk_length = X_data.shape[0]

            loss_sum += chunk_loss_sum
            length += chunk_length

        return loss_sum / length

    def accuracy(self, eval_chunk):
        
        eval_right_sum = 0
        eval_len_sum = 0

        for X_eval, y_eval in eval_chunk.generate():
            y_pred = super().predict_labels(X_eval)

            if self.loss != node_funcs.BCE:
                y_eval = np.argmax(y_eval, axis=1)

            eval_right_sum += (y_pred == y_eval).sum()
            eval_len_sum += X_eval.shape[0]
        
        accuracy = eval_right_sum / eval_len_sum

        return accuracy

class SuperChunkNN(SmoothNN):
    """NN class for SuperChunk iterator for large datasets
    see no_resources.SuperChunk"""
    def fit(self, 
            super_chunk, 
            learning_rate=1, 
            reg_strength=0.0001,
            num_epochs=30,
            epoch_gap=5,
            batch_prob=.01,
            param_path=None,
            verbose=True, 
            display=True):
        """Feeds data from chunk generator into model for training
        
        Args:
        """
        # set attributes
        self.super_chunk = super_chunk
        if self.super_chunk.tdt_sizes[1] > 0:
            self._dev_flag = True
        else:
            self._dev_flag = False
        self.learning_rate = learning_rate
        self.reg_strength = reg_strength

        # validate inputs
        super()._val_structure

        # get initial params
        super()._get_initial_params() 

        # initialize lists for costs
        self._train_costs = []
        self._dev_costs = []

        if display:
            fig, ax = plt.subplots()
            ax.set_title(f"Average Loss ({self.loss.name}) vs. Epoch Gap")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Average Loss")

        for epoch in range(num_epochs):
            print(f"epoch: {epoch}")
            start_time = time.time()
            for data in self.super_chunk.generate():
                X_train, y_train, _, _, _, _ = data

                super()._batch_total_pass(X_train, y_train)
                end_time = time.time()
                print(f"{end_time-start_time}")
                if verbose:
                    if np.random.binomial(1, batch_prob):
                        sampled_batch_loss = super().avg_loss(X_train, y_train)

                        print(f"\t Sampled batch loss: {sampled_batch_loss}")

                start_time = time.time()
                
            if param_path:
                joblib.dump(self.params,param_path)

            # record costs after each epoch gap
            if epoch % epoch_gap == 0:
                epoch_train_cost, epoch_dev_cost = self.get_td_costs()
                self._train_costs.append(epoch_train_cost)
                if verbose:
                    print(f"\t Training cost: {epoch_train_cost}")

                if epoch_dev_cost:
                    self._dev_costs.append(epoch_dev_cost)
                    if verbose:
                        print(f"\t Dev cost: {epoch_dev_cost}")

                if display:
                    true_epoch = epoch + 1
                    super()._update_epoch_plot(fig, ax, true_epoch, num_epochs, epoch_gap)
                    
            

    def get_td_costs(self):

        train_loss_sum = 0
        train_length = 0

        dev_loss_sum = 0
        dev_length = 0

        for X_train, y_train, X_dev, y_dev, _, _ in self.super_chunk.generate():
            y_train_probs = super().predict_prob(X_train)
            
            chunk_train_loss_sum = np.sum(self.loss.forward(y_train_probs, y_train))
            chunk_train_length = X_train.shape[0]
            
            train_loss_sum += chunk_train_loss_sum
            train_length += chunk_train_length
            
            if self._dev_flag:
                y_dev_probs = super().predict_prob(X_dev)
                chunk_dev_loss_sum = np.sum(self.loss.forward(y_dev_probs, y_dev))
                chunk_dev_length = X_dev.shape[0]

                dev_loss_sum += chunk_dev_loss_sum
                dev_length += chunk_dev_length

        train_cost = train_loss_sum / train_length
        if self._dev_flag:
            dev_cost = dev_loss_sum / dev_length
        else:
            dev_cost = None

        return train_cost, dev_cost
            
    def accuracy(self):
        """Evaluate accuracy efficiently
        
        Returns:
            train_accuracy (float)
            dev_accuracy (float)
            test_accuracy (float)
        """

        train_right_sum = 0
        train_len_sum = 0

        dev_right_sum = 0
        dev_len_sum = 0

        test_right_sum = 0
        test_len_sum = 0

        for data in self.super_chunk.generate():
            X_train, y_train, X_dev, y_dev, X_test, y_test = data

            # calculate activation values for each layer (includes predicted values)
            y_pred_train = super().predict_labels(X_train)
            y_pred_dev = super().predict_labels(X_dev)
            y_pred_test = super().predict_labels(X_test)

            if self.loss != node_funcs.BCE:
                y_train = np.argmax(y_train,axis=1)
                y_dev = np.argmax(y_dev,axis=1)
                y_test = np.argmax(y_test,axis=1)
            
            train_right_sum += (y_pred_train == y_train).sum()
            dev_right_sum += (y_pred_dev == y_dev).sum()
            test_right_sum += (y_pred_test == y_test).sum()

            train_len_sum += X_train.shape[0]
            dev_len_sum += X_dev.shape[0]
            test_len_sum += X_test.shape[0]
        
        # calculate the accuracies
        train_accuracy = train_right_sum / train_len_sum
        dev_accuracy = dev_right_sum / dev_len_sum
        test_accuracy = test_right_sum / test_len_sum

        return train_accuracy, dev_accuracy, test_accuracy