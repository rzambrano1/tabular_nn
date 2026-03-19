
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

#########################
# Functions and Classes #
#########################

class tabular_dataset_protocol(Protocol):

    _raw_data: pd.DataFrame
    
    def load_data(self, df: pd.DataFrame) -> pd.DataFrame:
        ...


class argn_dataset(tabular_dataset_protocol):
    """
    An object to store a table of any size with mixed data types.
    
    The class has methods to preprocess the data frame in 
    accordance with the specifications of the TabularARGN model.

    Attributes:
    ----------

    _raw_data : pd.DataFrame
        A pandas data frame unprocessed
    
    table : pd.DataFrame
        A copy of the raw data that will be preprocessed and passed
        to the PyTorch Dataset class
    """

    def __init__(self, _raw_data: pd.DataFrame):

        """
        Raises:
        -------

        TypeError
            If data frame is not pd.DataFrame

        """
        if not isinstance(_raw_data, pd.DataFrame):
            raise TypeError(f"Instances of {self.__class__.__name__} can only be initiated with pandas.DataFrame objects...")
        
        self._raw_data = _raw_data
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

        # Casting into Int64 float columns that contain integers
        for col, _ in self._float_columns_to_cast_to_integer:
            df_pl = df_pl.with_columns(
                    pl.col(col).cast(pl.Int64, strict=True)
                    )
            
        # Removing non-compatible columns
        if len(self._incompatible_columns) > 0:
            columns_to_drop, _ = zip(*self._incompatible_columns)
            df_pl = df_pl.drop(columns_to_drop)

        # Generating mappings for categorial values coded as strings
        self.categorical_encoding_maps = generate_categorical_encoding_mappings(df_pl, self._categorical_columns, self.nrow)

        self.categorical_decoding_maps = generate_categorical_decoding_mappings(self.categorical_encoding_maps)

        # Generating mappings for numerical discrete columns 
        self.numerical_discrete_encoding_maps = generate_numerical_discrete_encoding_mappings(df_pl, self._numerical_discrete_columns)

        self.numerical_discrete_decoding_maps = generate_numeric_discrete_decoding_mappings(self.numerical_discrete_encoding_maps)

        # Selecting encoding strategy for float columns

        self._numeric_strategy = select_numeric_strategy(df_pl, self._numerical_float_columns)

        df_pl = self.argn_preprocessing(
            df_pl = df_pl,
            # Categorical
            cat_encoding_map = self.categorical_encoding_maps, 
            cat_cols = self._categorical_columns,
            # Numerical Discrete
            float_to_int_cols = self._numerical_discrete_columns,
            num_discrete_encoding = self.numerical_discrete_encoding_maps,
            mumerical_discrete_cols = self._float_columns_to_cast_to_integer
            # Numerical Binned and Digit

            # Need to add parameters to preprocess float, datetime types
            )

        return df_pl

        
    def argn_preprocessing(
            self, df_pl: pl.DataFrame, 
            cat_encoding_map:dict[dict[str,int]], 
            cat_cols:list[tuple[str,int]],
            float_to_int_cols:list[tuple[str,int]],
            num_discrete_encoding:dict[dict[str,int]],
            mumerical_discrete_cols:list[tuple[str,int]],
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
            A polars data frame.
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

        # Rare categorical, clipping of outliera, geospatial, and text mapping 
        # not implemented yet.Some of these transformations could be implemented
        # outside this class

        # Recoding columns with categorial values coded as strings
        if len(cat_cols) > 0:
            df_pl = encode_categorical(df_pl, cat_encoding_map, [item[0] for item in cat_cols])
        
        # Casting float columns with discrete values as Int64
        if len(float_to_int_cols) > 0:
            df_pl = discrete_float_into_int(df_pl, float_to_int_cols)
        # Recoding columns with integer data types
        if len(mumerical_discrete_cols) > 0:
            df_pl = encode_numerical_discrete(df_pl, num_discrete_encoding, [item[0] for item in mumerical_discrete_cols])
        # Recoding columns with float data types
        
        # Recoding columns with datetime data types

        return df_pl

def select_numeric_strategy(df_pl: pl.DataFrame, float_cols: list[tuple[str,int]]) -> list[str]:
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

    col_encoding_strategy : list[str]
        A list with the encoding strategy for each column with floats
    """

    float_col_names, _ = zip(*float_cols)
    col_encoding_strategy = []

    for col in float_col_names:

        numbers_to_analyze = df_pl[col].to_list()
        string_numbers = [str(abs(number)) for number in numbers_to_analyze] 
        splited_numbers = [tuple(str_number.split('.')) for str_number in string_numbers]
        len_int_and_dec = [(len(x), len(y)) for x, y in splited_numbers]
        
        max_decimal_places = max(x[1] for x in len_int_and_dec)

        max_num_digits = max((x[0]+x[1]) for x in len_int_and_dec)

        if max_decimal_places <=2:
            col_encoding_strategy.append("BINNED")
        elif max_num_digits > 6 or max_decimal_places > 3:
            col_encoding_strategy.append("DIGIT")
        else:
            col_encoding_strategy.append("BINNED")

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
            is_integer = (df_pl[col_names[i]] == df_pl[col_names[i]].floor()).all()
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

