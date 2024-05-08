import numpy as np
import csv

class Chunk:
    """Chunk object to deal with parsing large datasets, 
    feeds data chunk by chunk without reading whole dataset into memory.
    
    Attributes:
        data_csv_path (str) : csv file path
        chunk_size (int) : size of the chunks
        seed (int) : seed for train,dev,test split in chunks

    Methods:
        see method docstrings
    
    """

    def __init__(self, data_csv_path, chunk_size, seed=100):
        """
        Args:
            data_csv_path (str) : csv file path
            chunk_size (int) : size of the chunks
            seed (int) : seed for train,dev,test split in chunks
        """
        self.data_csv_path = data_csv_path
        self.chunk_size = chunk_size
        self.seed = seed

    def generate(self, 
                 make_sparse=False,
                 sparse_dim=None,
                 tdt_sizes=(.95,.04,.01)):
        """Generator to produce chunks from large data set split into train, dev, test
        
        Args:
            make_sparse (bool, default=True) : whether or not to convert 
            tdt_sizes (3_tuple, default=(.95,.04,.01)) : split of the data into (train_share, dev_share, test_share) as proportion of 1 ex. (.95,.04,.01)

        Yields:
            X_data (numpy array) : 
            y_data (numpy array) :
        """
        # validation
        val_list = np.array(tdt_sizes)

        if val_list.sum() != 1:
            raise Exception("tdt split must sum to 1")
        
        if np.any(val_list < 0) or np.any(val_list > 1):
            raise Exception("tdt splits must be between 0 and 1 inclusive")

        # set the seed for consistent results
        # setting the seed once in teh beginning for every epoch ensures same training, dev, and test data
        np.random.seed(self.seed)

        with open(self.data_csv_path, "r") as csv_file:
            csv_reader = csv.reader(csv_file)
            next(csv_reader)
            end_flag = False

            while not end_flag:
                # to gather examples in current chunk
                chunk, end_flag = self.get_current_chunk(csv_reader, end_flag)
                shuffled_idxs = np.random.permutation(len(chunk))
                
                X_data, y_data = self.data_label_split(chunk)
                if make_sparse:
                    X_data = OneHotArray(shape=(len(X_data),sparse_dim),idx_array=X_data) # how to deal with splitting this
                X_train, y_train, X_dev, y_dev, X_test, y_test = self.tdt_split(X_data, y_data, shuffled_idxs, tdt_sizes)

                yield X_train, y_train, X_dev, y_dev, X_test, y_test

    def get_current_chunk(self, csv_reader, end_flag):
        """Gets current chunk (section of examples) based on current state of csv_reader
        
        Args:
            csv_reader (reader object) : reader object for the data csv
            end_flag (bool) : whether or not the end of the csv file has been reached
        
        Returns:
            chunk (numpy array) : current chunk of examples in numpy array form
            end_flag (bool) : whether or not the end of the csv file has been reached
        
        """
        i = 0 
        chunk = []
        while i < self.chunk_size and not end_flag:
            try:
                line = next(csv_reader)
                chunk.append(line)
            except: # reached the end of the csv file
                end_flag = True
            i+=1

        chunk = np.array(chunk)

        return chunk, end_flag
    
    def data_label_split(self, chunk):
        """split chunk into data and labels
        
        Args:
            chunk (numpy array) : complete data with labels as last column

        Returns:
            X_data (numpy array) : inputs, each row is an entry
            y_data (numpy array) : labels/outputs
        """

        X_data = chunk[:,:-1]
        y_data = chunk[:,-1].reshape(-1,1)

        return X_data, y_data

    
    def tdt_split(self, X_data, y_data, shuffled_idxs, tdt_sizes):
        """ Splits the data into train dev and test sets based on tdt_split, splits into entries and labels
        
        Args:
            X_data (numpy array) :
            y_data (numpy array)
            shuffled_idxs (numpy array) : permutation of non-negative integers up to but not including the chunk size
            tdt_split (3_tuple) : split of the data into (train_share, dev_share, test_share) as proportion of 1 ex. (.95,.04,.01)

        Returns:
            X_train (numpy array) : training entries (num_training_examples x num_features) 
            y_train (numpy array) : training labels (num_training_examples x num _features)
            X_dev (numpy array) : development "..."
            y_dev (numpy array) : development "..."
            X_test (numpy array) : test "..."
            y_test (numpy array) : test "..."
        """
        
        # divide the data into train, dev, test
        train_share = tdt_sizes[0]
        dev_share = tdt_sizes[1]

        train_upper = round(train_share * len(X_data))
        dev_upper = train_upper + round(dev_share * len(X_data))

        train_idxs = shuffled_idxs[:train_upper]
        dev_idxs = shuffled_idxs[train_upper:dev_upper]
        test_idxs = shuffled_idxs[dev_upper:]

        X_train = X_data[train_idxs]
        y_train = y_data[train_idxs]

        X_dev = X_data[dev_idxs]
        y_dev = y_data[dev_idxs]

        X_test = X_data[test_idxs]
        y_test = y_data[test_idxs]

        return X_train, y_train, X_dev, y_dev, X_test, y_test



class OneHotArray:
    """Sparse array for maximizing storage efficiency

    Attributes:

    """

    def __init__(self, shape, idx_array=None, oha_dict=None):
        """
        Args:
            shape (tuple) : dimensions of the array
            idx_array (array, default=None) : array where each row corresponds to a row vector in the OneHotArray
                integers in the array correspond to column indices of the 1 entries
            oha_dict (dict) : {row:col_idxs} dict

        """
        self.shape = shape
        self.ndim = 2
        
        if idx_array and not oha_dict:
            if self.shape[0] < len(idx_array):
                raise Exception("Number of row vectors in array must be greater than amount given")
            self.idx_rel = {row_idx:col_idxs for row_idx, col_idxs in enumerate(idx_array)}
        elif oha_dict and not idx_array:
            if self.shape[0] < max(oha_dict.keys()) + 1:
                raise Exception("Number of row vectors in array must be greater than max row index plus one")
            self.idx_rel = oha_dict
        else:
            raise Exception("Must either instantiate OneHotArray with an idx_array or oha_dict")

        self.validate_col_idxs()
        
    def validate_col_idxs(self):
        for col_idxs in self.idx_rel.values():
            for col_idx in col_idxs:
                self.validate_idx(col_idx, axis=1)
        
    def __matmul__(self, other):
        # validation
        if other.ndim != 2:
            raise Exception("Dimensions of composite transformations must be 2")

        if isinstance(other, np.ndarray): #sparse - dense multiplication
            # validation
            if self.shape[1] != other.shape[0]:
                    raise Exception("Inner dimensions must match")
            outside_dims = (self.shape[0], other.shape[1])

            product = np.zeros(outside_dims)

            for row_idx, col_idxs in self.idx_rel.items():
                product[row_idx] = other[col_idxs].sum(axis=0)

            return product
        elif isinstance(other, OneHotArray):
            pass
        else:
            raise Exception("OneHotArray can only matrix multiply with numpy array or another OneHotArray")

    def __rmatmul__(self, other):
        # (b x s) (s x next layer) will become O(b) 
        # validation
        if other.ndim != 2:
            raise Exception("Dimensions of composite transformations must be 2")
        
        if isinstance(other, np.ndarray): # dense-sparse multiplication
            outside_dims = (other.shape[0], self.shape[1])

            product = np.zeros(outside_dims)
            
            if other.shape[1] != self.num_vectors:
                raise Exception("Inner dimensions must match")
            
            transposed = self.T

            for row_idx, col_idxs in transposed.idx_rel:
                product[:,row_idx] = other[:,col_idxs]

            return product
        elif isinstance(other, OneHotArray):
            pass
        else:
            raise Exception("OneHotArray can only matrix multiply with numpy array")

    def __getitem__(self, key):
        """Defining getitem to duck type with numpy arrays for 0th axis slicing and indexing"""

        # define dimensions and n_rows placeholder
        n_rows = 0
        n_cols = self.shape[1]

        gathered = {}
        if isinstance(key, int):
            n_rows = self.add_int_key(key, gathered, n_rows)
        elif isinstance(key, slice):
            n_rows = self.add_slice_key(key, gathered, n_rows)
        elif isinstance(key, list): #
            for sub_key in key:
                if isinstance(sub_key, int):
                    n_rows = self.add_int_key(sub_key, gathered, n_rows)
                else:
                    raise SyntaxError
        else:
            raise SyntaxError
        
        return OneHotArray(shape=(n_rows,n_cols), oha_dict=gathered)
        
    def add_int_key(self, int_idx, gathered, n_rows):
        """Get integer index value 
        Args:
            int_idx (int) : integer row idx of the oha
            gathered (dict) : current gathered values of the indexed oha
            n_rows (int) : counter for amount of rows 

        Returns:
            n_rows (int) : number of rows in new oha
        """
        self.validate_idx(int_idx)
        if int_idx < 0:
            int_idx = self.convert_neg_idx(int_idx)
        if int_idx in self.idx_rel:
            gathered[n_rows] = self.idx_rel[int_idx]
        
        n_rows += 1

        return n_rows

    def convert_neg_idx(self, idx):
        """Converts negative idxs for __getitem__ to positive idx
        Args:
            idx (int) : negative int to convert to positive
        """
        return self.shape[0] + idx
    
    def validate_idx(self, idx, axis=0):
        """See if the idx is out of bounds or not
        
        Args:
            idx (int) :  index to validate
            axis (int, default=0)

        """
        indexed_rows = self.shape[axis]
        if idx < -indexed_rows or idx > indexed_rows-1:
            raise IndexError(f"Given index {idx} does not exist")
        
    def add_slice_key(self, slice_obj, gathered, n_rows):
        """Add corresponding valid index values in slice to gathered
        
        Args:
            slice (slice) : key slice object
            gathered (dict) : current gathered values of the indexed oha
            n_rows (int) : counter for amount of rows 
        
        """
        if slice_obj.step is None:
            step = 1
        else:
            step = slice_obj.step
        for idx in range(slice_obj.start, slice_obj.stop, step):
            n_rows = self.add_int_key(idx, gathered, n_rows)

        return n_rows
    
    @property
    def T(self):
        """create a transpose of the one-hot array"""
        
        transpose_idx_rel = {}

        new_shape = (self.shape[1], self.shape[0])

        for row_idx, row in self.idx_rel.items():
            for col_idx in row:
                if col_idx in transpose_idx_rel:
                    transpose_idx_rel[col_idx].append(row_idx)
                else:
                    transpose_idx_rel[col_idx] = [row_idx]
        
        #transpose_idx_vals =  [transpose_idx_rel[idx] for idx in range(len(transpose_idx_rel))]
        new = OneHotArray(shape=new_shape, oha_dict=transpose_idx_rel)

        return new
    
    def __str__(self):
        return str(self.idx_rel)
