
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

# Numerical & Data 
# ----------------

import numpy as np
import polars as pl
import pandas as pd

# Local Modules
# -------------

from argn_encoder_decoder import encode_categorical

#########################
# Functions and Classes #
#########################

class tabular_dataset_protocol(Protocol):

    _raw_data: pd.DataFrame

    @property
    def table(self):
        if self._table is None:
            self._table = self.load_data(self._raw_data)
        return self._table
    
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
        if isinstance(_raw_data, pd.DataFrame) == False:
            raise TypeError(f"Instances of {self.__class__.__name__} can only be initiated with pandas.DataFrame objects...")
        
        self._raw_data = _raw_data

        self.load_data(self._raw_data)

    def load_data(self, df_pd: pd.DataFrame) -> pd.DataFrame:
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

        self.nrow = df_pl.height
        
        self.col_names = df_pl.columns

        self.ncol = len(self.col_names)

        self.table_dim = (self.nrow, self.ncol)

        self.dtypes = [str(df_pl[name].dtype) for name in self.col_names]

        # Generating mappings for categorial values coded as strings
        self.categorical_encoding_maps, self.categorical_cols = generate_categorical_encoding_mappings(self.dtypes, self.col_names, df_pl, self.nrow)

        self.categorical_decoding_maps = generate_categorical_decoding_mappings(self.categorical_encode_maps)

        
    def argn_preprocessing(df_pl: pl.DataFrame, encode_map:dict[dict[str,int]], cat_cols:list[str]) -> pl.DataFrame:
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
        """

        # Rare categorical mapping not implemented yet

        # Recoding columns with categorial values coded as strings

        if len(cat_cols) > 0:
            df_pl = encode_categorical(df_pl, encode_map, cat_cols)


def generate_categorical_encoding_mappings(df_dtypes:list[str], col_names:list[str], df_pl:pl.DataFrame, nrow:int) -> dict[dict[str,int]]:
    """
    Creates a mapping for encoding categorical variables. Assumes columns with string data types are categorical variables
    coded with levels as strings.

    Note the missing values (null) will be casted as None 

    Parameters:
    ----------

    df_dtypes : list[str]
        A list of data types of each column in df_pl
    col_names : list[str]
        A list of column names in df_pl
    df_pl : pl.DataFrame
        A polars data frame
    nrow : int
        Number of rows in df_pl

    Returns:
    -------

    categorical_encode_maps : dict[dict[str,int]]
        A dictionary with dictionaries that map uniques levels to integers
    categorical_columns : list[str]
        A list of columns with categorical strings

    Warns:
    -----

    Warning:
        If the number of levels in a given column are more than a third of the number of 
        rows, the client receives a warning to make sure a column with open ended text 
        was not passed to the model.

    """
    
    categorical_encode_maps = {}
    categorical_columns = []

    for i,dtype in enumerate(df_dtypes):
        if dtype in ["String", "Categorical", "Categories", "Enum", "Utf8"]:

            unique_vals = df_pl[:, i].unique().to_list()
            col = col_names[i]
            
            categorical_columns.append(col)
            if len(unique_vals) > nrow/3:
                Warning(f"Check {col} does not contain open ended values. This implementation only process categorical levels...")

            map_name = col

            categorical_encode_maps[map_name] = {
                val: idx for idx, val in enumerate(unique_vals)
            }

    return categorical_encode_maps, categorical_columns


def generate_categorical_decoding_mappings(encode_maps: dict[dict[str,int]]) -> dict[dict[int,str]]:
    """
    Assumes a mapping for encoding categorical variables. Returns a mapping for decoding categorical
    variables back into strings 

    Parameters:
    ----------

    encode_maps : dict[dict[str,int]]
        An encoding map generated with generate_categorical_encoding_mappings()

    returns:
    -------

    decode_map : dict[dict[int,str]]
        A decoding map to restore encoded data back into its original form
    """

    decode_map = {
            outer_key: {v: k for k, v in inner_dict.items()}
            for outer_key, inner_dict in encode_maps.items()
        }
    
    return decode_map

        
        