import time
import numpy as np
import joblib
from . import nn_layers
from . import node_funcs
from . import no_resources
from . import learning_funcs

import matplotlib.pyplot as plt

class ChunkNN:
    """Simple NN class"""

    def __init__(self):
        self.layers = []
        self.loss_layer = None
        self._last_output_layer = None

        self._learning_rates = []
        self._reg_strengths = []
        self._batch_sizes = []

        self._has_fit = False
        self._has_dropout = False
        self._loaded_model = False

        self._stable_constant = 10e-8

    def add_layer(self, layer_object):

        if self.loss_layer is not None:
            raise Exception("Cannot add another layer after Loss layer")

        if len(self.layers) == 0:
            if isinstance(layer_object, nn_layers.Web):
                if layer_object.input_shape is None:
                    raise Exception("First layer input shape must be set in layer")
                
                layer_object.input_layer_flag = True
                self._last_output_layer = layer_object
            else:
                raise Exception("First layers must be a web layer")

        if len(self.layers) > 0:
            if isinstance(layer_object, (nn_layers.Web, nn_layers.Dropout, nn_layers.BatchNorm)):
                if layer_object.input_shape is not None:
                    if layer_object.input_shape != self._last_output_layer.output_shape:
                        raise Exception("Input shape must equal the output shape of the last web")
                else:
                    layer_object.input_shape = self._last_output_layer.output_shape
                
                if isinstance(layer_object, (nn_layers.Web)):
                    self._last_output_layer = layer_object

            if isinstance(layer_object, nn_layers.BatchNorm):
                self._last_output_layer.feeds_into_norm = True

            if isinstance(self.layers[-1], nn_layers.Web):
                self.layers[-1].output_layer = layer_object

        if isinstance(layer_object, nn_layers.Loss):
            if layer_object.loss_func == node_funcs.BCE:

                if self._last_output_layer.output_shape != 1:
                    raise Exception("Should use output layer of size 1 when using binary cross entropy loss,\
                                    decrease layer size to 1 or use CE (regular cross entropy)")

            if layer_object.loss_func == node_funcs.CE:
                if self._last_output_layer.output_shape < 2:
                    raise Exception("Should use cross entropy loss for multi-class classification, increase layer-size or use BCE")

            self.loss_layer = layer_object
        
        if isinstance(layer_object, nn_layers.Dropout):
            self._has_dropout = True

        self.layers.append(layer_object)
            
    def _initialize_params(self):

        if not self.loss_layer:
            raise Exception("Neural net not complete, add loss layer before initializing params")
        
        for layer in self.layers:
            if layer.learnable:
                layer.initialize_params()

    def _set_epoch_dropout_masks(self, epoch):

        for layer in self.layers:
            if isinstance(layer, nn_layers.Dropout):
                layer.set_epoch_dropout_mask(epoch)

    def fit(self,
            train_chunk,
            dev_chunk=None,
            learning_scheduler=learning_funcs.ConstantRate(1),
            reg_strength=0,
            num_epochs=30,
            epoch_gap=5,
            batch_prob=.01,
            batch_seed=100,
            model_path=None,
            display_path=None,
            verbose=True, 
            display=True):

        # validate inputs def validate_structure
        self._val_structure()
        
        # clear data for the first fit
        self._has_fit = True
        self._epoch = 0

        self._train_costs = []
        self._dev_costs = []

        self._learning_scheduler = learning_scheduler

        self._learning_rates = []
        self._reg_strengths = []
        self._batch_sizes = []

        self._batch_prob = batch_prob
        self._batch_seed = batch_seed

        self._rounds = []

        self._train_collection = None
        self._dev_collection = None

        # get the initial weights and biases
        if verbose:
            print("WARNING: Creating new set of params")

        self._initialize_params()

        self.refine(train_chunk=train_chunk,
                    dev_chunk=dev_chunk,
                    reg_strength=reg_strength,
                    num_epochs=num_epochs,
                    epoch_gap=epoch_gap,
                    model_path=model_path,
                    display_path=display_path,
                    verbose=verbose, 
                    display=display)
        
    def _val_structure(self):
        """validate the structure of the the NN"""
        if len(self.layers) == 0:
            raise Exception("No layers in network")

        if self.loss_layer is None:
            raise Exception("Please add a loss function")
        
    def refine(self,
            train_chunk,
            dev_chunk=None,
            reg_strength=None,
            num_epochs=15,
            epoch_gap=5,
            model_path=None,
            display_path=None,
            verbose=True, 
            display=True):
        
        if not self._has_fit:
            raise Exception("Please fit the model before refining")

        if not train_chunk._train_chunk:
            raise Exception("Given train chunk must be a valid train chunk (_train_chunk attribute set to True)")
        
        self.train_chunk = train_chunk

        # set chunks
        batch_size = train_chunk.chunk_size

        self._dev_flag = False
        if dev_chunk is not None:
            self.dev_chunk = dev_chunk
            self._dev_flag = True

        # add rounds
        self._rounds.append(self._epoch)

        # update attributes if needed
        if reg_strength is not None:
            self.reg_strength = reg_strength
        
        # initialize the display
        if display:
            fig, ax = self._initialize_epoch_plot()

        start_epoch = self._epoch
        end_epoch = self._epoch + num_epochs

        for epoch in range(start_epoch, end_epoch):
            print(f"Epoch: {epoch}")
            epoch_start_time = time.time()

            # update learning rate
            self.learning_rate = self._learning_scheduler.get_learning_rate(epoch)

            # update epoch data
            self._learning_rates.append(self.learning_rate)
            self._reg_strengths.append(self.reg_strength)
            self._batch_sizes.append(batch_size)

            if self._has_dropout:
                self._set_epoch_dropout_masks(epoch=epoch)

            # set a seed for sampling batches for loss
            rng2 = np.random.default_rng(self._batch_seed)
            #start_gen = time.time()
            for X_train, y_train in train_chunk.generate():
                #end_gen = time.time()
                #print(f"gen time {end_gen-start_gen}")

                #start_batch = time.time()
                self._batch_total_pass(X_train, y_train)
                #end_batch = time.time()
                #print(f"batch time: {end_batch-start_batch}")
                
                if verbose:
                    if rng2.binomial(1, self._batch_prob):
                        sampled_batch_loss = self.cost(X_train, y_train)

                        print(f"\t Sampled batch loss: {sampled_batch_loss}")
                
                #start_gen = time.time()

            epoch_end_time = time.time()
            if verbose:
                print(f"Epoch completion time: {(epoch_end_time-epoch_start_time) / 3600} Hours")
            
            # record costs after each epoch gap
            if epoch % epoch_gap == 0:
                gap_start_time = time.time()
                
                epoch_train_cost = self.chunk_cost(self.train_chunk)
                self._train_costs.append(epoch_train_cost)

                if verbose:
                    
                    print(f"\t Training cost: {epoch_train_cost}")

                if self._dev_flag:
                    epoch_dev_cost = self.chunk_cost(self.dev_chunk)
                    self._dev_costs.append(epoch_dev_cost)
                    if verbose:
                        print(f"\t Dev cost: {epoch_dev_cost}")
                else:
                    self._dev_costs.append(None)

                if display:
                    self._update_epoch_plot(fig, ax, epoch, end_epoch)

                gap_end_time = time.time()

                if verbose:
                    print(f"Gap completion time: {(gap_end_time-gap_start_time) / 3600} Hours")
            else:
                self._train_costs.append(None)
                self._dev_costs.append(None)

            self._epoch += 1
            if model_path:
                self.save_model(model_path, train_chunk, dev_chunk)

            if display and display_path:
                fig.savefig(display_path)

        plt.close()

    # PROP

    def _batch_total_pass(self, X_train_batch, y_train_batch):
        """forward propagation, backward propagation and parameter updates for gradient descent"""

        # forward pass
        self._forward_prop(X_train_batch)

        # perform back prop to obtain gradients and update
        self._back_prop(X_train_batch, y_train_batch)

    def _forward_prop(self, X_train):
        
        input = X_train
        for layer in self.layers:
            input = layer.advance(input, forward_prop_flag=True)

    def _back_prop(self, X_train, y_train):
        
        loss_layer = True
        for layer in reversed(self.layers):
            if loss_layer:
                input_grad_to_loss = layer.back_up(y_train)
                loss_layer = False
            else:
                if layer.learnable:
                    input_grad_to_loss = layer.back_up(input_grad_to_loss, learning_rate=self.learning_rate, reg_strength=self.reg_strength)
                else:
                    input_grad_to_loss = layer.back_up(input_grad_to_loss)
    
    # EPOCH PLOT 
    def _initialize_epoch_plot(self):
        plt.close()
        fig, ax = plt.subplots()
        ax.set_title(f"Average Loss ({self.loss_layer.loss_func.name}) vs. Epoch")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Average Loss")
        if len(self._rounds) > 1:
            for round, pos in enumerate(self._rounds):
                ax.axvline(x=pos, linestyle="--", alpha=.5, c="grey")
                ax.text(x=pos+.1, y=.75, 
                        color="grey",
                        s=f"Round {round}", 
                        rotation=90, 
                        verticalalignment='top', 
                        horizontalalignment='left',
                        transform=ax.get_xaxis_transform())
        return fig, ax
    
    def _update_epoch_plot(self, fig, ax, epoch, num_epochs):
        """Updates training plot to display average losses

        Args:
            fig (matplotlib.pyplot.figure) : figure containing plot axis
            ax (matplotlib.pyplot.axis) : axis that contains line plots
            epoch (int) : epoch number to graph new data for
            num_epochs (int) : total number of epochs
        """
        if not self._train_collection:
            self._train_collection = ax.scatter(range(0,epoch+1), self._train_costs, marker="x", c="red", alpha=.5, label="Average training loss")
            ax.legend()
        else:
            self._update_scatter(self._train_collection, range(0,epoch+1), self._train_costs)

        if self._dev_flag:
            if not self._dev_collection:
                self._dev_collection = ax.scatter(range(0,epoch+1), self._dev_costs, marker="x", c="blue", alpha=.5, label="Average dev loss")
                ax.legend()
            else:
                self._update_scatter(self._dev_collection, range(0,epoch+1), self._dev_costs)

        max_val = max([d for d in self._dev_costs if d is not None] + [t for t in self._train_costs if t is not None])
        ax.set(xlim=[-.5,num_epochs], ylim=[min(0, max_val*2),max(0, max_val*2)])
        plt.pause(.2)

    @staticmethod
    def _update_scatter(collection, new_x, new_y):
        offsets = np.c_[new_x, new_y]
        collection.set_offsets(offsets)

    # COST/LOSS

    def cost(self, X, y_true):
        """Calculate the average loss depending on the loss function
        
        Args:
            X (numpy array) : examples to predict on (num_examples x num_features)
            y (numpy array) : true labels (num_examples x 1)

        Returns:
            cost (numpy array) : average loss given predictions on X and truth y
        
        """
        # calculate activation values for each layer (includes predicted values)
        y_pred = self.predict_prob(X)

        cost = self.loss_layer.get_cost(y_pred, y_true)

        # L2 regularization loss with Frobenius norm
        if self.reg_strength != 0: 
            cost = cost + self.reg_strength * sum(np.sum(layer._weights ** 2) for layer in self.layers if isinstance(layer, nn_layers.Web))

        return cost
    
    def chunk_cost(self, eval_chunk):
        """"Calculate loss on the eval chunk"""
        
        loss_sum = 0
        length = 0

        for X_data, y_data in eval_chunk.generate():
            y_probs = self.predict_prob(X_data)

            chunk_cost_sum = np.sum(self.loss_layer.get_total_loss(y_probs, y_data))
            chunk_length = X_data.shape[0]

            loss_sum += chunk_cost_sum
            length += chunk_length

        return loss_sum / length
    
    def predict_prob(self, X):
        """Obtain output layer activations
        
        Args: 
            X (numpy array) : examples (num_examples x num_features)
        
        Returns:
            cur_a (numpy array) : probabilities of output layer
        """

        # set the data as "a0"
        input = X

        # go through the layers and save the activations
        for layer in self.layers:
            input = layer.advance(input, forward_prop_flag=False)

        return input
    
    def predict_labels(self, X):
        """Predict labels of given X examples
        
        Args:
            X (numpy array) : array of examples by row

        Returns:
            predictions (numpy array) : array of prediction for each example by row
        
        """
        # calculate activation values for each layer (includes predicted values)
        final_activations = self.predict_prob(X)

        if isinstance(self.loss_layer.loss_func, node_funcs.BCE):
            predictions = final_activations > .5
        else:
            predictions = np.argmax(final_activations, axis=1)

        return predictions
    
    # PERFORMANCE

    def accuracy(self, eval_chunk):
        
        eval_right_sum = 0
        eval_len_sum = 0

        for X_eval, y_eval in eval_chunk.generate():
            y_pred = self.predict_labels(X_eval)

            if not isinstance(self.loss_layer.loss_func, node_funcs.BCE):
                y_eval = np.argmax(y_eval, axis=1)

            eval_right_sum += (y_pred == y_eval).sum()
            eval_len_sum += X_eval.shape[0]
        
        accuracy = eval_right_sum / eval_len_sum

        return accuracy
    
    def save_model(self, path, train_chunk, dev_chunk):
        self.train_chunk = None
        self.dev_chunk = None
        joblib.dump(self, path)
        self.train_chunk = train_chunk
        self.dev_chunk = dev_chunk
    


