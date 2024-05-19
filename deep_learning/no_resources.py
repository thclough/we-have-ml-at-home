import numpy as np
import csv
import gzip
import re
import time
import warnings
import traceback

#TODO
# how to standardize data

## could change the opener attrs to kwargs, or make an opener object (# jar opener)
## could extract tdt_split, make separate files for tdt_split to avoid leakage, makes object simpler

## one hot encoding fro y_data just to OHA
## optimizing chunk size based on operations (but would have to peek at operations)

## "like" arg, np.loadtxt __array_function__ protocol for oha?

def chop_up_data(source_path, tdt_split):
    """Section given data source into train, test, and dev data sets"""
    
    pass

class JarOpener:
    """A file opener to read a file at a given path
    
    Attributes:
        _opener (function)) : opener function for file
        _open_kwargs (dict)) : kwargs for the opener
    """

    def __init__(self, source_path) -> None:
        self._opener, self._opener_kwargs = self.get_opener_attrs(source_path)
    
    @staticmethod
    def get_file_extension(source_path):
        """Get the file extension of a file path
        
        Args:
            source_path (str) : file path string

        Returns:
            (str) The file extension 
        """
        # Define the regex pattern to match the file extension
        pattern = r'\.(.+)$'
        
        # Use re.search to find the pattern in the file path
        match = re.search(pattern, source_path)
        
        # If a match is found, return the matched file extension
        if match:
            return match.group(1)
        else:
            return None  # Return None if no extension is found

    def get_opener_attrs(self, source_path):
        """Return the correct open function for the file extension
        
        Args:
            source_path (str) : file path string
        
        Returns:
            opener (function) : function to open file
            opener_kwargs (dict) : args for the opener function
        """
        file_extension = self.get_file_extension(source_path)

        if file_extension == "csv":
            opener = open
            opener_kwargs = {"file": source_path, "mode": "r", "encoding": None}
        elif file_extension == "csv.gz":
            opener = gzip.open
            opener_kwargs =  {"filename": source_path, "mode": "rt", "encoding":"utf-8"}
        else:
            raise Exception("File extension not supported")
        
        return opener, opener_kwargs

    def __enter__(self):
        # return the open file
        self.open_source = self._opener(**self._opener_kwargs)
        return self.open_source

    def __exit__(self, exc_type, exc_value, exc_traceback):
        if self.open_source:
            self.open_source.close()
        if exc_type and exc_type != GeneratorExit:
            print(f"An exception occurred: {exc_value}")
            traceback.print_exception(exc_type, exc_value, exc_traceback)
        return False   

class Chunk:
    pass

class ChunkBlock:
    """Chunk object to deal with parsing large datasets, 
    feeds data chunk by chunk without reading whole dataset into memory.
    ChunkBlock is different from Chunk because ChunkBlock does not need separate 
    train, dev, and test data sources and will perform this data split on the fly 
    (per chunk, hence the name block, because there is a multinomial draw per block)
    
    Attributes:
        data_csv_path (str) : csv file path
        chunk_size (int) : size of the chunks
        seed (int) : seed for train,dev,test split in chunks
        tdt_split (3_tuple) : split of the data into (train_share, dev_share, test_share) as proportion of 1 ex. (.95,.04,.01)

    Methods:
        see method docstrings
    
    """

    def __init__(self, 
                 chunk_size,
                 tdt_sizes=(.95,.04,.01),
                 seed=100):
        """
        Args:
            chunk_size (int) : size of the chunks
            tdt_sizes (3_tuple, default=(.95,.04,.01)) : split of the data into (train_share, dev_share, test_share) as proportion of 1 ex. (.95,.04,.01)
            seed (int) : seed for train,dev,test split in chunks
        """
        self.chunk_size = chunk_size
        self.tdt_sizes = tdt_sizes
        self.seed = seed
        # whether or not input or output data specified yet
        self._input_flag = False
        self._output_flag = False

    def set_data_input_props(self, input_csv_path, data_selector=np.s_[:], skip_rows=0, sparse_dim=None, standardize=False):
        """Set the data/input properties for the chunk object
        
        Args:
            input_csv_path (str) : csv file path of data
            data_selector (IndexExpression, default=None) : 1D index expression to select certain columns, if none specified will select all columns
            skip_rows (int, default=0) : number of rows to skip
            sparse_dim (int, default=None) : dimensions of the sparse vectors, if applicable
            standardize (bool, default=False) : whether or not to standardize data
        """

        # get the opener function
        self._input_jar = JarOpener(input_csv_path)

        # select all columns if no data columns
        self._data_input_selector = data_selector

        self._sparse_dim = sparse_dim

        self._input_skip_rows = skip_rows

        self.set_input_dim()

        self._standardize=standardize
        
        # calculate mean and standard deviation of training data if standardizing
        if self._standardize:
            if self._sparse_dim is not None:
                raise Exception("ChunkyBlock does not support standardization for sparse dims")
            self.set_training_data_mean()
            self.set_training_data_std()

        self._input_flag=True

    def set_input_dim(self):
        """Set the dimension of the input data by peeking inside the input data file"""

        if self._sparse_dim:
            dim = self._sparse_dim
        else:
            dim = self.get_selector_dim(self._input_jar, self._data_input_selector)

        self.input_dim = dim

    def set_training_data_mean(self):
        """Retrieve the mean of the training data"""

        # set the seed
        np.random.seed(self.seed)

        # open the file
        with self._input_jar as input_file:
            
            # skip rows
            for _ in range(self._input_skip_rows):
                next(input_file)

            train_sum = np.zeros(self.input_dim)
            train_count = 0

            # obtain the chunk of X_data
            while True:
                start_time = time.time()
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    X_data = np.loadtxt(input_file, delimiter=",", max_rows=self.chunk_size)

                data_len = X_data.shape[0]

                # split the data into training data
                if data_len == 0:
                    break

                X_data = X_data[:,self._data_input_selector]
            
                train_idxs, _, _ = self.get_tdt_idxs(data_len)

                X_train = X_data[train_idxs]

                train_sum += X_train.sum(axis=0)
                train_count += data_len

        self._train_mean = train_sum / train_count

    def set_training_data_std(self):
        # set the seed
        np.random.seed(self.seed)

        # open the file
        with self._input_jar as input_file:
            
            # skip rows
            for _ in range(self._input_skip_rows):
                next(input_file)

            sum_dev_sqd = np.zeros(self.input_dim)
            train_count = 0

            # obtain the chunk of X_data
            while True:
                start_time = time.time()
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    X_data = np.loadtxt(input_file, delimiter=",", max_rows=self.chunk_size)

                data_len = X_data.shape[0]

                # split the data into training data
                if data_len == 0:
                    break

                X_data = X_data[:,self._data_input_selector]

                train_idxs, _, _ = self.get_tdt_idxs(data_len)

                X_train = X_data[train_idxs]

                sum_dev_sqd += ((X_train - self._train_mean) ** 2).sum(axis=0)
                train_count += data_len
        
        self._train_std = np.sqrt(sum_dev_sqd / train_count)

    def set_data_output_props(self, output_csv_path, data_selector=np.s_[:], skip_rows=0, one_hot_width=None):
        """Set the label properties for the chunk object
        
        Args:
            output_csv_path (str) : csv file path of data
            data_selector (IndexExpression, default=None) : 1D index expression to select certain columns, if none specified will select all columns
            skip_rows (int, default=0) : number of rows to skip
            one_hot_width (list, default=None) : number of categories for one hot encoding
        """

        # get the opener function
        self._output_jar = JarOpener(output_csv_path)

        self._data_output_selector = data_selector

        # handle one hot encoding
        self.one_hot_width = one_hot_width

        self._output_skip_rows = skip_rows

        self.set_output_dim()

        self._output_flag = True

    @property
    def one_hot_width(self):
        return self._one_hot_width
    
    @one_hot_width.setter
    def one_hot_width(self, one_hot_width_cand):
        """Validate one hot encoding for set_data_output_props and set if appropriate
        
        Args:
            one_hot_width_cand (int) : candidate for one_hot_width
        """
        if one_hot_width_cand is None:
            self._one_hot_width = one_hot_width_cand
        elif isinstance(one_hot_width_cand, int):
            dim = self.get_selector_dim(self._output_jar, self._data_output_selector)

            if dim == 1:
                self._one_hot_width = one_hot_width_cand
            else:
                raise AttributeError("Cannot set one_hot_width and one hot encode if dimensions of raw output is not 1")
        else:
            return AttributeError(f"one_hot_width must be an integer or None, value of {one_hot_width_cand} given")

    def set_output_dim(self):
        """Set the output dimension (After one hot encoding)"""
        if self.one_hot_width:
            dim = self.one_hot_width
        else:
            dim = self.get_selector_dim(self._output_jar, self._data_output_selector)

        self.output_dim = dim

    @staticmethod
    def get_selector_dim(jar_opener, data_selector):
        """Return the dimension of selected data within file"""
        with jar_opener as file:
            reader = csv.reader(file) 
            line = np.array(next(reader))
            selector_dim = len(line[data_selector])

        return selector_dim

    def generate(self):
        """Generator to produce chunks from large data set split into train, dev, test
        
        Yields:
            X_train (numpy array) : 
            y_train (numpy array) :
            X_dev (numpy array) :
            y_dev (numpy array) :
        """
        # check if input and output are set
        if not self._input_flag:
            raise Exception("Please set data input properties (set_data_input_props)")

        if not self._output_flag:
            raise Exception("Please set data output properties (set_data_output_properties)")

        # set the seed for consistent results
        # setting the seed once in the beginning for every epoch ensures same training, dev, and test data
        # use numpy load text skiprows and max_rows=chunk_size

        np.random.seed(self.seed)

        # open data and label files as file objects to decrease open and close overhead
        with self._input_jar as input_file, self._output_jar as output_file:
            
            for _ in range(self._input_skip_rows):
                next(input_file)
            for _ in range(self._output_skip_rows):
                next(output_file)

            while True:
                start_time = time.time()
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    X_data = np.loadtxt(input_file, delimiter=",", max_rows=self.chunk_size)
                    y_data = np.loadtxt(output_file, delimiter=",", max_rows=self.chunk_size, ndmin=2)
                end_time = time.time()

                if len(X_data) == 0 and len(y_data) == 0:
                    break

                X_data = X_data[:,self._data_input_selector]
                y_data = y_data[:,self._data_output_selector]
                
                #print(end_time-start_time)
                
                # datasets not the same size
                if len(X_data) != len(y_data):
                    raise Exception("Input file and output file are not the same length")
                
                if self._sparse_dim:
                    X_data = OneHotArray(shape=(len(X_data),self._sparse_dim),idx_array=X_data)

                if self._one_hot_width:
                    y_data = self.one_hot_labels(y_data)

                train_idxs, dev_idxs, test_idxs = self.get_tdt_idxs(X_data.shape[0])

                X_train = X_data[train_idxs]
                y_train = y_data[train_idxs]

                X_dev = X_data[dev_idxs]
                y_dev = y_data[dev_idxs]

                X_test = X_data[test_idxs]
                y_test = y_data[test_idxs]

                if self._standardize:
                    X_train = (X_train - self._train_mean) / self._train_std
                    X_dev = (X_dev - self._train_mean) / self._train_std
                    X_test = (X_test - self._train_mean) / self._train_std

                yield X_train, y_train, X_dev, y_dev, X_test, y_test

    def one_hot_labels(self, y_data):
        """One hot labels from y_data
        
        Args:
            y_data (numpy array)

        Returns:
            one_hot_labels
        
        """
        one_hot_labels = np.zeros((y_data.size, self._one_hot_width))
        one_hot_labels[np.arange(y_data.size), y_data.astype(int).flatten()] = 1

        return one_hot_labels
    
    def get_tdt_idxs(self, data_length):
        """Retrieve indexes of train, dev, and test set for data of given length
        
        Args:
            data_length (int) : length of the data to retrieve idxs for

        Returns:
            train_idxs (numpy ndarray) 
            dev_idxs (numpy ndarray) 
            test_idxs (numpy ndarry) 
        
        """

        shuffled_idxs = np.random.permutation(data_length)

        train_share = self.tdt_sizes[0]
        dev_share = self.tdt_sizes[1]

        train_upper = round(train_share * data_length)
        dev_upper = train_upper + round(dev_share * data_length)

        train_idxs = shuffled_idxs[:train_upper]
        dev_idxs = shuffled_idxs[train_upper:dev_upper]
        test_idxs = shuffled_idxs[dev_upper:]

        return train_idxs, dev_idxs, test_idxs

    @property
    def tdt_sizes(self):
        return self._tdt_sizes

    @tdt_sizes.setter
    def tdt_sizes(self, tdt_tuple_cand):
        """Validate and set tdt_sizes """

        # validation
        val_list = np.array(tdt_tuple_cand)

        train_share = tdt_tuple_cand[0]

        if np.any(val_list < 0) or np.any(val_list > 1):
            raise AttributeError("tdt splits must be between 0 and 1 inclusive")

        if train_share == 0:
            raise AttributeError("Training share must be greater than 0")

        if val_list.sum() != 1:
            raise AttributeError("tdt split must sum to 1")
    
        self._tdt_sizes = tdt_tuple_cand
        
class OneHotArray:
    """Sparse array for maximizing storage efficiency

    Attributes:

    """
    def __init__(self, shape, idx_array=None, oha_dict=None):
        """
        Args:
            shape (tuple) : dimensions of the array
            idx_array (array, default=None) : array where each row corresponds to a row vector in the OneHotArray
                integers in the array correspond to column indices of the 1 entries, 
                only positive integers allowed, except for -1 which counts as null space
            oha_dict (dict) : {row:col_idxs} dict

        """
        self.shape = shape
        self.ndim = 2

        # instantiate cand_idx_rel dict to hold sparse array
        cand_idx_rel = {}

        if isinstance(idx_array, (np.ndarray, list)) != None and oha_dict == None:
            if self.shape[0] < len(idx_array):
                raise Exception("Number of row vectors in array must be greater than amount given")
            for row_idx, col_idxs in enumerate(idx_array):
                filtered_col_idxs = self.filter_col_idxs(col_idxs)
                cand_idx_rel[row_idx] = filtered_col_idxs

        elif oha_dict != None and idx_array == None:
            
            if oha_dict.keys():
                if self.shape[0] < max(oha_dict.keys()) + 1:
                    raise Exception("Number of row vectors in array must be greater than max row index plus one")
                
            for row_idx, col_idxs in oha_dict.items():
                self.validate_idx(row_idx, axis=0)
                filtered_col_idxs = self.filter_col_idxs(col_idxs)
                cand_idx_rel[row_idx] = filtered_col_idxs
        else:
            raise Exception("Must either instantiate OneHotArray with an idx_array or oha_dict")

        self.idx_rel = cand_idx_rel
    
    def filter_col_idxs(self, raw_col_idxs):
        """Add valid column idxs to list and return valid idxs (in range of reference matrix when 0-indexed)

        Args:
            raw_col_idxs (array-like): list of possible column idxs
        
        Returns: 
            filtered_col_idxs (list): valid col idxs for a given row
        """
        filtered_col_idxs = []
        for col_idx in raw_col_idxs:
            if col_idx >= 0:
                self.validate_idx(col_idx, axis=1)
                filtered_col_idxs.append(int(col_idx))
            if col_idx < -1:
                raise Exception("No negative indices allowed (besides -1 which represents null space)")

        return filtered_col_idxs
    
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
        elif isinstance(key, (list, np.ndarray)):
            for sub_key in key:
                if isinstance(sub_key, tuple([int] + np.sctypes["int"])):
                    n_rows = self.add_int_key(sub_key, gathered, n_rows)
                else:
                    raise SyntaxError
        else:
            raise SyntaxError
        
        # for empty 
        if n_rows == 0:
            n_cols = 0

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
        # only need to gather rows in oha, 
        # if not in oha array (all zeroes) then should not be part of oha
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
    
    def __eq__(self, other):
        if isinstance(other, OneHotArray):
            return self.shape == other.shape and self.idx_rel == other.idx_rel
        elif isinstance(other, np.ndarray):
            if self.shape != other.shape:
                return False
            for i in range(other.shape[0]):
                for j in range(other.shape[1]):
                    val = other[i][j]
                    if val != 0 and val != 1:
                        return False
                    elif val == 1:
                        if j not in self.idx_rel[i]:
                            return False
                    elif val == 0:
                        if j in self.idx_rel[i]:
                            return False
                        
            return True

    def __str__(self):
        return str(self.idx_rel)
