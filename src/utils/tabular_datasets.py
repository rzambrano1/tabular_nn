
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
from .argn_encoder_decoder import BinDesign, get_bin_designs, generate_numerical_binned_encoding_mappings, generate_numeric_binned_decoding_mappings

#########################
# Functions and Classes #
#########################

class TabularDatasetProtocol(Protocol):

    _raw_data: pd.DataFrame
    
    def load_data(self, df: pd.DataFrame) -> pd.DataFrame:
        ...


class ArgnDataset(TabularDatasetProtocol):
    """
    An object to store a table of any size with mixed data types.
    
    The class has methods to preprocess the data frame in 
    accordance with the specifications of the TabularARGN model.

    Attributes:
    ----------

    _raw_data : pd.DataFrame
        A pandas data frame unprocessed
    clip_cols : bool, optional, default_value = True
        If True outliers in integer and float columns 
        are clipped to preset percentiles
    
    table : pd.DataFrame
        A copy of the raw data that will be preprocessed and passed
        to the PyTorch Dataset class
    """

    def __init__(self, _raw_data: pd.DataFrame, clip_cols: Optional[bool] = True):

        """
        Raises:
        -------

        TypeError
            If data frame is not pd.DataFrame

        """
        if not isinstance(_raw_data, pd.DataFrame):
            raise TypeError(f"Instances of {self.__class__.__name__} can only be initiated with pandas.DataFrame objects...")
        
        self._raw_data = _raw_data
        self._table_pd = None
        self.clip_cols = clip_cols
        self._table = self.load_data(self._raw_data)


    @property
    def table(self):
        if self._table_pd is None:
            self._table_pd = self._table.to_pandas()
        return self._table_pd
        
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

        # Data fraame dimensions
        self.nrow = df_pl.height
        
        self.ncol = df_pl.width

        self.table_dim = (self.nrow, self.ncol)

        # Data frame column name and data types
        self.col_names = df_pl.columns

        self.dtypes = [str(df_pl[name].dtype) for name in self.col_names]

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
            df_pl = clip_columns(df_pl, self._numerical_discrete_columns, lower_quantile = 0.01, upper_quantile = 0.99) 
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
            # Numerical Binned and Digit
            num_float_encoding = None,
            numerical_float_cols = self._numerical_float_columns
            # Need to add parameters to float, datetime types
            )

        return df_pl

        
    def argn_preprocessing(
            self, 
            df_pl: pl.DataFrame,
            cat_encoding_map: dict[dict[str,int]], 
            cat_cols: list[tuple[str,int]],
            float_to_int_cols: list[tuple[str,int]],
            num_discrete_encoding: dict[dict[str,int]],
            numerical_discrete_cols: list[tuple[str,int]],
            num_float_encoding: dict[dict[str,int]],
            numerical_float_cols: list[tuple[str,int]]
            ) -> pl.DataFrame:
        """
        Method to preprocess a polars data frame in accordance to tabularARGN
        specifications, listed in Appendix A of the paper:

        TabularARGN models exclusively operate on categorical columns, all other data 
        types are converted into one or more categorical sub-columns using specific 
        encoding strategies.
        
        Parametrs:
        ---------

        df_pl : pl.DataFrame
            A polars data frame
        encode_map : dict[dict[str,int]]
            Mapping to encode string levels as integer levels of categorical variables
        cat_cols : list[str]
            A list of columns with categorical variables coded as strings
        float_to_int_cols:list[tuple[str,int]],
            A list of columns with float values that should be discrete values
        mumerical_discrete_cols:list[tuple[str,int]]
            A list of columns with numerical discrete values
        
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
        
        # Recoding columns with datetime data types

        return df_pl

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
        string_numbers = [str(abs(number)) for number in numbers_to_analyze] 
        splited_numbers = [tuple(str_number.split('.')) for str_number in string_numbers]
        
        integers_in_numbers, decimals_in_numbers = zip(*splited_numbers)

        integer_lenghs = [len(integer) for integer in integers_in_numbers]
        decimal_lenghts = [len(decimal) for decimal in decimals_in_numbers]

        max_decimal_places = max(integer_lenghs)

        max_num_digits = max(decimal_lenghts)

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

