
#!/usr/bin/python3
"""
This module contains tabular dataset classes.

The purpose of these classes is to enable models to load a variety of 
generic tabular datasets with any number of columns of mixed data
types.
"""

#################
# Session Setup #
#################

# Standard Library
# ----------------
 
import os
import random
from pathlib import Path

from typing import Protocol
from typing import Optional

import warnings

# Numerical & Data 
# ----------------

import numpy as np
import polars as pl
import pandas as pd

# Local Modules
# -------------

# Functions for categorical variables
from .argn_encoder_decoder import encode_categorical, generate_categorical_encoding_mappings, generate_categorical_decoding_mappings

# Functions for numerical discrete
from .argn_encoder_decoder import discrete_float_into_int, generate_numerical_discrete_encoding_mappings, generate_numeric_discrete_decoding_mappings
from .argn_encoder_decoder import encode_numerical_discrete

# Functions for numerical BINNED and numerical DIGIT
from .argn_encoder_decoder import BinDesign, get_bin_designs, generate_numerical_binned_encoding_mappings, generate_numeric_binned_decoding_mappings, encode_numerical_binned

from .argn_encoder_decoder import encode_numerical_digit

# Functions for datetime

from .argn_encoder_decoder import encode_datetime

#########################
# Functions and Classes #
#########################

class TabularDatasetProtocol(Protocol):

    _raw_data: pd.DataFrame
    
    def load_data(self, df_pd: pd.DataFrame) -> pl.DataFrame:
        ...


class ArgnDataset(TabularDatasetProtocol):
    """
    An object to store a table of any size with mixed data types.
    
    The class has methods to preprocess the data frame in 
    accordance with the specifications of the TabularARGN model.

    Assumes that columns with open-ended text have been dropped 
    by the client. Columns with strings will be assumed to have 
    string levels. A warning will be given for columns where the
    number of unique integer levels is greater than 1/3 of the
    number of rows.

    Attributes:
    ----------

    # --- Parameters ---
    
    clip_cols : bool, optional, default_value = True
        If True outliers in integer and float columns 
        are clipped to preset percentiles
    
    set_seed : int, optional, default_value = True
        An integer to set a random seed for random
        and numpy libraries

    # --- Raw Data ---
    
    _raw_data : pd.DataFrame
        A pandas data frame unprocessed

    self.nrow : int
        Number of rows in original data frame

    self.ncol : int
        Number of columns in the original data frame

    self.table_dim : tuple[int, int]
        Dimentions of the original data frame in the
        (number of rows, number of columns) format

    self.col_names : list[str]
        Column names in the original data frame

    self.dtypes : list[str]
        The data types of each column in the original 
        data frame
    
    # --- Processed Data ---

    self._table : pl.DataFrame
        The postprocessed data frame after undergoing all
        transformation and encodings per TabularARGN paper

    table : pd.DataFrame
        A copy of the _table casted into a pandas data frame
        this object will be be passed to the PyTorch Dataset class    

    # --- Columns Metadata ---

    self._categorical_columns : list[tuple[str,int]]
        A list of tuples in the (column name, column index) format
        that contain categorical data encoded as string levels
        in the original data frame

    self._numerical_discrete_columns : list[tuple[str,int]]
        A list of tuples in the (column name, column index) format
        that contain integer data encoding integer levels in the 
        original data frame

    self._float_columns_to_cast_to_integer : list[tuple[str,int]]
        A list of tuples in the (column name, column index) format
        that originaly had foat values with no relevant decimals
        (e.g. 10.0) and were casted as integers by the preprocessing 
        pipeline. Indeces correspond the original data frame

    self._numerical_float_columns : list[tuple[str,int]]
        A list of tuples in the (column name, column index) format
        that contain float values in the original data frame

    self._datetime_columns : list[tuple[str,int]]
        A list of tuples in the (column name, column index) format
        that contain date time, date, or time data type values 
        in the original data frame

    self._duration_columns : list[tuple[str,int]]
        A list of tuples in the (column name, column index) format
        that contain duration type values in the original data frame.
        Duration columns are all casted into seconds

    self._bool_columns : list[tuple[str,int]]
        A list of tuples in the (column name, column index) format
        of columns that contain boolean values in the original data 
        frame.

    self._incompatible_columns : list[tuple[str,int]]
        A list of tuples in the (column name, column index) format
        of columns in the original data set that contain data types.
        Not suported types include: binary, list, array, and struct.
        These columns are dropped in the preprocessing pipeline.

    self._numerical_binned_columns : list[tuple[str,int]]
        A list of tuples in the (column name, column index) format
        that were encoded by the preprocessing pipeline using the
        BINNED strategy as described by the TabularARGN paper.

    self._numerical_digit_columns : list[tuple[str,int]]
        A list of tuples in the (column name, column index) format
        that were encoded by the preprocessing pipeline using the
        DIGIT strategy as described by the TabularARGN paper.

    # --- Encoding Maps ---

    self.categorical_encoding_maps : dict[str,dict[str,int]]
        A dictionary with dictionaries that map uniques levels 
        to integers

    self.categorical_decoding_maps : dict[dict[int,str]]
        A decoding map to restore encoded data back into its 
        original form

    self.numerical_discrete_encoding_maps : dict[dict[int,int]]
        A dictionary with dictionaries that map uniques levels to integers

    self.numerical_discrete_decoding_maps : dict[dict[int,int]]
        A decoding map to restore encoded data back into its original form

    self.numerical_binned_encoding_maps : dict[dict[tuple[float,float], int]]
        A dictionary containing dictionaries that map intervals to integers.
        Outer key: column name
        Inner dict: maps (lower_edge, upper_edge) tuples to int category index
                    None mapped to  0  (reserved for missing values)

    self.numerical_binned_decoding_maps : dict[dict[int, tuple[float,float]]]
        A decoding map to restore encoded data back into its intermediate form.
        Outer key: column name
        Inner dict: maps integer level back to (lower_edge, upper_edge) tuples 
                    0 mapped to  None  (reserved for missing values)

    self.numerical_digit_encoding_maps : dict[str, tuple[int,int]]
        The encoding scheme for each column, with the column name as key
        and a tuple with the number of decimal and digits as values 

    self.datetime_encoding_map : dict[str,str]
        A helper dict mapping the specific datetime type of each 
        date/time columns

    self._numeric_strategy : dict[str,str]
        A dictionary with the column name as key and encoding strategy as value
        for each column containing float

    """

    def __init__(
            self, 
            _raw_data: pd.DataFrame, 
            clip_cols: Optional[bool] = True,
            encode_datetime_as_discrete: Optional[bool] = True,
            set_seed: Optional[int] = 42
            ):

        """
        Parameters:
        ----------

        _raw_data: pd.DataFrame, 
            A pandas dataframe to be processed
        clip_cols: Optional[bool], default_value = True,
            A boolean parameter, if True outlier values are clipped
            to prevent identification of observations with unusual
            or extreme values
        encode_datetime_as_discrete: Optional[bool], default_value = True,
            A boolean parameter, if True features in the family of datetime 
            data types are encoded twice: first into sub-columns and then
            as integer levels. If False encoding consist only in sub-columns 
        set_seed: int, default_value = 42
            Sets a seed for reproducibility

        Raises:
        -------

        TypeError
            If data frame is not pd.DataFrame

        """
        if not isinstance(_raw_data, pd.DataFrame):
            raise TypeError(f"Instances of {self.__class__.__name__} can only be initiated with pandas.DataFrame objects...")
        
        random.seed(set_seed)
        np.random.seed(set_seed)

        self._raw_data = _raw_data
        self._table_pd = None
        self.clip_cols = clip_cols
        self.encode_datetime_as_discrete = encode_datetime_as_discrete
        self._table = self.load_data(self._raw_data)
        

    @property
    def table(self):
        return self._table.to_pandas()
        
        
    def load_data(self, df_pd: pd.DataFrame) -> pl.DataFrame:
        """
        Preprocess _raw_data. The data frame is casted into polars for fast transformation.
        Converting a the polars oject back into pandas is nearly instantaneus because they  
        often just pass "pointers" to the data in memory rather than duplicating the entire 
        dataset. Strickly speaking, polars is faster with tables with number of rows > 100k,
        however, since the purpose is building an universal pipeline, every data set will
        be casted into polars.

        Parameters:
        ----------

        df_pd : pd.DataFrame
            A pandas dataframe stores in _raw_data
        """

        df_pl = pl.from_pandas(df_pd).clone()

        # Data Frame Metadata
        # -------------------

        # Data frame dimensions
        self.nrow = df_pl.height
        
        self.ncol = df_pl.width

        self.table_dim = (self.nrow, self.ncol)

        # Data frame column name and data types
        self.col_names = df_pl.columns

        self.dtypes = [str(df_pl[name].dtype.base_type()) for name in self.col_names]

        # Data frame metadata
        (self._categorical_columns,
        self._numerical_discrete_columns,
        self._float_columns_to_cast_to_integer,
        self._numerical_float_columns,
        self._datetime_columns,
        self._bool_columns,
        self._incompatible_columns) = column_types_sieve(df_pl, self.dtypes, self.col_names)

        # Transformations
        # ---------------

        # Separating datetime columns from duration columns, then casting duration as a scalar
        self._duration_columns = [
            (col_name, col_idx) 
            for col_name, col_idx in self._datetime_columns 
            if isinstance(df_pl[col_name].dtype, pl.Duration)
        ]
        
        if len(self._duration_columns) > 0:
            self._datetime_columns = [
                (col_name, col_idx) 
                for col_name, col_idx in self._datetime_columns 
                if (col_name, col_idx) not in self._duration_columns
            ]
            
            for col_name, _ in self._duration_columns:
                df_pl = df_pl.with_columns(
                    pl.col(col_name).dt.total_seconds().cast(pl.Int64).alias(col_name) # Assumes duration columns smaller component are seconds
                )

        # Casting into Int64 float columns that contain integers
        for col, _ in self._float_columns_to_cast_to_integer:
            df_pl = df_pl.with_columns(
                    pl.col(col).cast(pl.Int64, strict=True)
                    )
            
        # Removing non-compatible columns
        if len(self._incompatible_columns) > 0:
            columns_to_drop, _ = zip(*self._incompatible_columns)
            df_pl = df_pl.drop(columns_to_drop)

        # Clipping outliers to preserve privacy [Optional]

        if self.clip_cols:
            # Integer
            # Not needed, clipping ZIP codes will result in missing information

            # Float
            df_pl = clip_columns(df_pl, self._numerical_float_columns, lower_quantile = 0.01, upper_quantile = 0.99)   

        # Mappings
        # --------

        # Generating mappings for categorial values coded as strings
        self.categorical_encoding_maps = generate_categorical_encoding_mappings(df_pl, self._categorical_columns, self.nrow)

        self.categorical_decoding_maps = generate_categorical_decoding_mappings(self.categorical_encoding_maps)

        # Generating mappings for numerical discrete columns 
        self.numerical_discrete_encoding_maps = generate_numerical_discrete_encoding_mappings(df_pl, self._numerical_discrete_columns)

        self.numerical_discrete_decoding_maps = generate_numeric_discrete_decoding_mappings(self.numerical_discrete_encoding_maps)

        # Initializing mappings for datetime types. Set to a value other than None if encode_datetime_as_discrete == True
        self.datetime_discretized_encoding_maps = {}

        self.datetime_discretized_decoding_maps = {}

        self._datetime_discretized_subcols_names = []

        # Encoding Strategy
        # -----------------

        # Selecting encoding strategy for float columns

        self._numeric_strategy = select_numeric_strategy(df_pl, self._numerical_float_columns)
        
        # Sorting float columns that will use BINNED strategy and those that will use DIGIT
        binned_strategy_target_cols = [col_name for col_name, strategy in self._numeric_strategy.items() if strategy == "BINNED"]
        digit_strategy_target_cols = [col_name for col_name, strategy in self._numeric_strategy.items() if strategy == "DIGIT"]

        self._numerical_binned_columns = [(col_name, col_index) for col_name, col_index in self._numerical_float_columns if col_name in binned_strategy_target_cols]
        self._numerical_digit_columns = [(col_name, col_index) for col_name, col_index in self._numerical_float_columns if col_name in digit_strategy_target_cols]
        
        # BINNED strategy
        self._column_binned_designs = get_bin_designs(df_pl, self._numerical_binned_columns)

        self.numerical_binned_encoding_maps = generate_numerical_binned_encoding_mappings(self._numerical_binned_columns, self._column_binned_designs)
        self.numerical_binned_decoding_maps = generate_numeric_binned_decoding_mappings(self.numerical_binned_encoding_maps)

        # DIGIT
        # Encoded directly in the preprocessing step

        # Preprocessing Data Frame
        # ------------------------

        df_pl = self.argn_preprocessing(
            df_pl = df_pl,
            # Categorical
            cat_encoding_map = self.categorical_encoding_maps, 
            cat_cols = self._categorical_columns,
            # Numerical Discrete
            float_to_int_cols = self._numerical_discrete_columns,
            num_discrete_encoding = self.numerical_discrete_encoding_maps,
            numerical_discrete_cols = self._float_columns_to_cast_to_integer,
            # Numerical Binned
            num_binned_encoding = self.numerical_binned_encoding_maps,
            numerical_binned_cols = self._numerical_binned_columns,
            binned_strategy_design = self._column_binned_designs, 
            # Numerical Digit
            numerical_digit_cols = self._numerical_digit_columns,
            # Datetime
            datetime_cols = self._datetime_columns,
            # Bool
            bool_cols = self._bool_columns
            )

        return df_pl

        
    def argn_preprocessing(
            self, 
            df_pl: pl.DataFrame,
            cat_encoding_map: dict[str,dict[str,int]], 
            cat_cols: list[tuple[str,int]],
            float_to_int_cols: list[tuple[str,int]],
            num_discrete_encoding: dict[str,dict[int,int]],
            numerical_discrete_cols: list[tuple[str,int]],
            num_binned_encoding: dict[str, dict[tuple[float, float] | None, int]],
            numerical_binned_cols: list[tuple[str,int]],
            binned_strategy_design: dict[str,BinDesign],
            numerical_digit_cols: list[tuple[str,int]],
            datetime_cols: list[tuple[str,int]],
            bool_cols: list[tuple[str,int]]
            ) -> pl.DataFrame:
        """
        Method to preprocess a polars data frame in accordance to tabularARGN
        specifications, listed in Appendix A of the paper:

        TabularARGN models exclusively operate on categorical columns, all other data 
        types are converted into one or more categorical sub-columns using specific 
        encoding strategies.

        Columns lists in the functions take a list of strings as arguments. Note
        that the data structure for columns lists are a list of a tuples with 
        (column name, column index). Thus list comprenhension must be passed to
        processing columns.
        
        Parametrs:
        ---------

        df_pl : pl.DataFrame
            A polars data frame
        cat_encoding_map : dict[str,dict[str,int]]
            Mapping to encode string levels as integer levels of categorical variables
        cat_cols : list[str]
            A list of columns with categorical variables coded as strings
        float_to_int_cols:list[tuple[str,int]],
            A list of columns with float values that should be discrete values
        num_discrete_encoding: dict[str,dict[str,int]]
            Encoding mappings for numerical discrete columns
        numerical_discrete_cols:list[tuple[str,int]]
            A list of columns with numerical discrete values
        num_binned_encoding: dict[str,dict[str,int]]
            Encoding maps for columns encoded using the BINNED strategy
        numerical_binned_cols: list[tuple[str,int]]
            A list of the columns following the BINNED strategy
        binned_strategy_design: dict[str,BinDesign]
            A dict storing the encoding strategy for columns encoded using BINNED
        numerical_digit_cols : list[tuple[str,int]]
            A list of columns following the DIGIT encoding strategy
        datetime_cols : list[tuple[str,int]]
            A list of columns with datetime columns to be processesed 
        
        Returns:
        -------

        A polars dataframe that went through the preprocessing phase
        """

        # Rare categorical, geospatial, and text mapping not implemented yet.
        # Some of these transformations could be implemented outside this class          

        # Recoding columns with categorial values coded as strings
        if len(cat_cols) > 0:
            df_pl = encode_categorical(df_pl, cat_encoding_map, [item[0] for item in cat_cols])
        
        # Casting float columns with discrete values as Int64
        if len(float_to_int_cols) > 0:
            df_pl = discrete_float_into_int(df_pl, float_to_int_cols)

        # Recoding columns with integer data types
        if len(numerical_discrete_cols) > 0:
            df_pl = encode_numerical_discrete(df_pl, num_discrete_encoding, [item[0] for item in numerical_discrete_cols])

        # Recoding columns with float data types
        
        # BINNED
        if len(numerical_binned_cols) > 0:
            df_pl = encode_numerical_binned(df_pl, num_binned_encoding, [item[0] for item in numerical_binned_cols], binned_strategy_design)

        # DIGIT
        if len(numerical_digit_cols) > 0:
            df_pl, self.numerical_digit_encoding_maps = encode_numerical_digit(df_pl, [item[0] for item in numerical_digit_cols])

        # Encoding columns with datetime data types
        if len(datetime_cols) > 0:
            df_pl, self.datetime_encoding_map = encode_datetime(df_pl, [item[0] for item in  datetime_cols])

        if self.encode_datetime_as_discrete:

            date_time_col_prefixes = [item[0] for item in self._datetime_columns + self._duration_columns]
            
            date_time_subcols = []

            duration_col_names = {dur_col_name for dur_col_name, _ in self._duration_columns}

            for col_name_prefix in date_time_col_prefixes:
                
                curr_prefix_subcols = []
                
                for i, subcol_name in enumerate(df_pl.columns): 
                    
                    if subcol_name.startswith(f"{col_name_prefix}_") or (subcol_name in duration_col_names):

                        curr_prefix_subcols.append((subcol_name, i))

                date_time_subcols = date_time_subcols + curr_prefix_subcols

            self._datetime_discretized_subcols_names = date_time_subcols

            self.datetime_discretized_encoding_maps = generate_numerical_discrete_encoding_mappings(df_pl, date_time_subcols)

            self.datetime_discretized_decoding_maps = generate_numeric_discrete_decoding_mappings(self.datetime_discretized_encoding_maps)

            df_pl = encode_numerical_discrete(df_pl, self.datetime_discretized_encoding_maps, [item[0] for item in date_time_subcols])

        # Casting boolean columns into integers

        if len(bool_cols) > 0: 
            boolean_columns_list, _ = zip(*bool_cols)
            for col_name in  boolean_columns_list:
                df_pl = df_pl.with_columns(
                    pl.col(col_name).cast(pl.Int8)
                )
                
        return df_pl
    

    def __repr__(self):
        return f"ArgnDataset(original_shape={self.table_dim}) - transformed_shape={self._table.shape}"

    def __str__(self):
        raise NotImplementedError

    def __eq__(self,other):
        """
        Two data sets are equal if their raw dataset and the trasformed
        data set are equal
        """
        return (
            self._raw_data.equals(other._raw_data) and
            self._table.equals(other._table)
        )

    def __len__(self):
        """
        Returns the number of rows in the dataset
        """
        return self.nrow
    

def select_numeric_strategy(df_pl: pl.DataFrame, float_cols: list[tuple[str,int]]) -> dict[str,str]:
    """
    Selects the encoding strategy for columns with float

    Parameters:
    ----------
    
    df_pl : pl.DataFrame
        A polars data frame
    float_cols : list[tuple[str,int]]
        A list with the columns with float data types

    Returns:
    -------

    col_encoding_strategy : dict[str,str]
        A dictionary with the column name as key and encoding strategy as value
        for each column containing floats
    """

    # Guard to avoid vValueError on empty list
    if not float_cols:
        return {}

    float_col_names, _ = zip(*float_cols)
    col_encoding_strategy = {}

    for col in float_col_names:

        numbers_to_analyze = df_pl[col].drop_nulls().to_list() # .drop_nulls() to avoid TypeError due to propagation of nulls

        integer_lengths = []
        decimal_lengths = []

        for number in numbers_to_analyze:

            formatted = f"{abs(number):.10f}"
            int_part, dec_part = formatted.split('.')

            dec_part = dec_part.rstrip('0') or '0'
            int_part = int_part.lstrip('0') or '0'
             
            integer_lengths.append(len(int_part))
            decimal_lengths.append(len(dec_part))

        max_num_digits = max(integer_lengths)
        max_decimal_places = max(decimal_lengths)

        if max_decimal_places <=2:
            col_encoding_strategy[col] = "BINNED"
        elif max_num_digits > 6 or max_decimal_places > 3:
            col_encoding_strategy[col] = "DIGIT"
        else:
            col_encoding_strategy[col] = "BINNED"

    return col_encoding_strategy

def column_types_sieve(
        df_pl: pl.DataFrame, 
        df_dtypes:list[str], 
        col_names:list[str],  
        ) -> tuple[
            list[tuple[str,int]], 
            list[tuple[str,int]], 
            list[tuple[str,int]], 
            list[tuple[str,int]], 
            list[tuple[str,int]],
            list[tuple[str,int]],
            list[tuple[str,int]]
            ]:
    """
    Creates list of columns by data type to enable targeted encoding for each column.

    Parameters:
    ----------

    df_pl: pl.DataFrame
        The data frame in the object instance
    df_dtypes : list[str]
        A list of the data types 
    col_names : list[str]
        A list with the column names
    
    Returns:
    -------

    A tuple with seven lists of tuples
        Each list contains information the names of the columns and its column number
    """

    # Supported data types
    cat_types = ["String", "Categorical", "Categories", "Enum", "Utf8"]
    int_types = ["Int8", "Int16", "Int32", "Int64", "UInt8", "UInt16", "UInt32", "UInt64"]
    float_types = ["Float16", "Float32", "Float64"]
    datetime_types = ["Date", "Time", "Datetime", "Duration"]
    bool_types = ["Boolean"]

    # Not supported data types
    not_compatiple_types = ["Binary", "List", "Array", "Struct"]

    categorical_columns = []
    numerical_discrete_columns = []
    float_columns_to_cast_to_integer = []
    numerical_float_columns = []
    datetime_columns = []
    bool_columns = []
    incompatible_columns = []
    
    for i,dtype in enumerate(df_dtypes):
        if dtype in cat_types:
            categorical_columns.append((col_names[i],i))
        elif dtype in int_types:
            numerical_discrete_columns.append((col_names[i],i))
        elif dtype in float_types:
            is_integer = (df_pl[col_names[i]].drop_nulls() == df_pl[col_names[i]].drop_nulls().floor()).all() # Added drop nulls to avoid null propagation as False
            if is_integer:
                float_columns_to_cast_to_integer.append((col_names[i],i))
                numerical_discrete_columns.append((col_names[i],i))
            else:
                numerical_float_columns.append((col_names[i],i))
        elif dtype in bool_types:
            bool_columns.append((col_names[i],i))
        elif dtype in datetime_types:
            datetime_columns.append((col_names[i],i))
        elif dtype in not_compatiple_types:
            incompatible_columns.append((col_names[i],i))
        else:
            incompatible_columns.append((col_names[i],i))

    return (
        categorical_columns,
        numerical_discrete_columns,
        float_columns_to_cast_to_integer,
        numerical_float_columns,
        datetime_columns,
        bool_columns,
        incompatible_columns
    )  

def clip_columns(df_pl: pl.DataFrame, cols_to_process: list[tuple[str,int]], lower_quantile: float = 0.01, upper_quantile: float = 0.99) -> pl.DataFrame:
    """
    Clips columns with numeical values.

    Parameters:
    ----------

    df_pl: pl.DataFrame
        A polars data aframe
    cols_to_process: list[tuple[str,int]]
        Lis of columns with either integer or float values to beclipped
    lower_quantile: int, default_value = 0.01
        Lower quantile threshold. Values below this quantile are replaced 
        with the value at this quantile. 
        Setting the lower cutoff at 0.01 selects the value below
        which the lowest 1% of data points fall
    upper_quantile: int, default_value = 0.99
        Upper quantile threshold. Values above this quantile are replaced 
        with the value at this quantile.
        Setting the upper cutoff at 0.99 selects the value below
        which the lowest 99% of data points fall. It also means selecting
        a value above which the highest 1% of data points fall.

    Returns:
    -------

    processed_df : pl.DataFrame
        Polars data frame with specified columns clipped to the given quantile range
    """

    # Guard to avoid ValueError on empty list
    if len(cols_to_process) == 0:
        return df_pl
    
    processed_df = df_pl
    cols_to_process_names, _ = zip(*cols_to_process)

    for col_name in cols_to_process_names:
        col_values = df_pl[col_name].to_numpy()
        lower = np.nanquantile(col_values, lower_quantile)
        upper = np.nanquantile(col_values, upper_quantile)
        processed_df = processed_df.with_columns(pl.col(col_name).clip(lower, upper))

    return processed_df


