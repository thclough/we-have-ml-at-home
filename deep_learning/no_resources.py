import numpy as np
import csv
import gzip
import re

#TODO
## could change the opener attrs to kwargs

class Chunk:
    """Chunk object to deal with parsing large datasets, 
    feeds data chunk by chunk without reading whole dataset into memory.
    
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

    def set_data_input_props(self, input_csv_path, data_selector=np.s_[:], sparse_dim=None):
        """Set the data/input properties for the chunk object
        
        Args:
            input_csv_path (str) : csv file path of data
            data_selector (IndexExpression, default=None) : 1D index expression to select certain columns, if none specified will select all columns
            sparse_dim (int, default=None) : dimensions of the sparse vectors, if applicable
        """
        self.input_csv_path = input_csv_path

        # get the opener function
        self._input_opener, self._input_read_mode, self._input_encoding = self.get_opener_attrs(self.input_csv_path)

        # select all columns if no data columns
        self._data_input_selector = data_selector

        self._sparse_dim = sparse_dim

        self.set_input_dim()
        
        self._input_flag=True

    def set_input_dim(self):
        """Set the dimension of the input data by peeking inside the input data file"""

        if self._sparse_dim:
            dim = self._sparse_dim
        else:
            with self._input_opener(self.input_csv_path, mode=self._input_read_mode, encoding=self._input_encoding) as file:
                reader = csv.reader(file) 
                line = np.array(next(reader))
                dim = len(line[self._data_input_selector])

        self.input_dim = dim

    def set_data_output_props(self, output_csv_path, data_selector=np.s_[:], one_hot_width=None):
        """Set the label properties for the chunk object
        
        Args:
            input_csv_path (str) : csv file path of data
            data_selector (IndexExpression, default=None) : 1D index expression to select certain columns, if none specified will select all columns
            one_hot_width (list, default=None) : number of categories for one hot encoding
        """

        self.val_set_data_output_props(output_csv_path, data_selector, one_hot_width)

        self.output_csv_path = output_csv_path

        # get the opener function
        self._output_opener, self._output_read_mode, self._output_encoding = self.get_opener_attrs(self.output_csv_path)

        self._data_output_selector = data_selector

        # handle one hot encoding
        self._one_hot_width = one_hot_width

        self._output_flag = True

    @staticmethod
    def val_set_data_output_props(output_csv_path, data_selector, one_hot_width):
        """Validate output properties input for set_data_output_props
        
        Args:
            see set_data_output_props
        """
        if type(data_selector) is not int and one_hot_width:
            raise Exception("Cannot one hot encode multiple columns")
        
    def get_opener_attrs(self, file_path):
        """
        Args:
            file_path (str) : file path string

        Returns:
            opener (function) : function to open file
            read_mode (str) : read mode to use for opener
            encoding (str) : encoding type
        """
        extension = self.get_file_extension(file_path)
        opener, read_mode, encoding = self.extension_to_opener(extension)

        return opener, read_mode, encoding

    @staticmethod
    def extension_to_opener(file_extension):
        """Return the correct open function for the file extension
        
        Args:
            file_extension (str) : file extension string
        
        Returns:
            opener (function) : function to open file
            read_mode (str) : read mode to use for opener
            encoding (str) : encoding type
        """
        if file_extension == "csv":
            opener = open
            read_mode = "r"
            encoding = None
        elif file_extension == "csv.gz":
            opener = gzip.open
            read_mode = "rt"
            encoding = "utf-8"
        else:
            raise Exception("File extension not supported")
        
        return opener, read_mode, encoding

    @staticmethod
    def get_file_extension(file_path):
        """Get the file extension of a file path
        
        Args:
            file_path (str) : file path string

        Returns:
            (str) The file extension 
        """
        # Define the regex pattern to match the file extension
        pattern = r'\.(.+)$'
        
        # Use re.search to find the pattern in the file path
        match = re.search(pattern, file_path)
        
        # If a match is found, return the matched file extension
        if match:
            return match.group(1)
        else:
            return None  # Return None if no extension is found

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
        np.random.seed(self.seed)

        with open(self.input_csv_path, "r") as input_file, open(self.output_csv_path, "r") as output_file:
            input_reader = csv.reader(input_file)
            output_reader = csv.reader(output_file)

            next(input_reader)
            next(output_reader)

            input_end_flag = False
            output_end_flag = False

            while not (input_end_flag and output_end_flag):
                # to gather examples in current chunk
                X_data, input_end_flag = self.get_current_chunk(input_reader, self._data_input_selector, input_end_flag)
                y_data, output_end_flag = self.get_current_chunk(output_reader, self._data_output_selector, output_end_flag)

                # datasets not the same size
                if len(X_data) != len(y_data):
                    raise Exception("Input file and output file are not the same length")

                shuffled_idxs = np.random.permutation(len(X_data))

                if self._sparse_dim:
                    X_data = OneHotArray(shape=(len(X_data),self._sparse_dim),idx_array=X_data)

                if self._one_hot_width:
                    y_data = self.one_hot_labels(y_data)

                X_train, y_train, X_dev, y_dev, X_test, y_test = self.tdt_split(X_data, y_data, shuffled_idxs)

                yield X_train, y_train, X_dev, y_dev, X_test, y_test

    def one_hot_labels(self, y_data):
        """One hot labels from 
        
        Args:
            y_data (numpy array) : 
        
        """
        one_hot_labels = np.zeros((y_data.size, self._one_hot_width))
        one_hot_labels[np.arange(y_data.size), y_data.astype(int)] = 1

        return one_hot_labels
    
    def get_current_chunk(self, csv_reader, data_selector, end_flag):
        """Gets current chunk (section of examples) based on current state of csv_reader
        
        Args:
            csv_reader (reader object) : reader object for the data csv
            data_selector (numpy index expression) : index expression to select data from
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
                np_line = np.array(line)
                chunk.append(np_line[data_selector]) #select for each line feed for efficiency
            except: # reached the end of the csv file
                end_flag = True
            i+=1

        chunk = np.array(chunk).astype(float)

        return chunk, end_flag
    
    def tdt_split(self, X_data, y_data, shuffled_idxs):
        """ Splits the data into train dev and test sets based on tdt_split, splits into entries and labels
        
        Args:
            X_data (numpy array) :
            y_data (numpy array)
            shuffled_idxs (numpy array) : permutation of non-negative integers up to but not including the chunk size
        
        Returns:
            X_train (numpy array) : training entries (num_training_examples x num_features) 
            y_train (numpy array) : training labels (num_training_examples x num _features)
            X_dev (numpy array) : development "..."
            y_dev (numpy array) : development "..."
            X_test (numpy array) : test "..."
            y_test (numpy array) : test "..."
        """
        m, _ = X_data.shape

        # divide the data into train, dev, test
        train_share = self.tdt_sizes[0]
        dev_share = self.tdt_sizes[1]

        train_upper = round(train_share * m)
        dev_upper = train_upper + round(dev_share * m)

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

    @property
    def tdt_sizes(self):
        return self._tdt_sizes

    @tdt_sizes.setter
    def tdt_sizes(self, tdt_sizes):

        # validation
        val_list = np.array(tdt_sizes)

        train_share = tdt_sizes[0]

        if np.any(val_list < 0) or np.any(val_list > 1):
            raise AttributeError("tdt splits must be between 0 and 1 inclusive")

        if train_share == 0:
            raise AttributeError("Training share must be greater than 0")

        if val_list.sum() != 1:
            raise AttributeError("tdt split must sum to 1")
    
        self._tdt_sizes = tdt_sizes
        
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
