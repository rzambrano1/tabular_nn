#!/usr/bin/python3
"""
This module contains the engine to encode and decode data to be fed to the TabularARGN architecture.
"""

#################
# Session Setup #
#################

# Standard Library
# ----------------
 
import os
import random
from pathlib import Path

from dataclasses import dataclass

import re

from typing import Protocol

import warnings
import logging

# Numerical & Data 
# ----------------

import numpy as np
import polars as pl
import pandas as pd

#########################
# Functions and Classes #
#########################

class encoding_decoding_engine_protocol(Protocol):

    def encode_categorical(self, df: pl.DataFrame):
        ...
    
    def decode_categorical(self, encode_map:dict):
        ...
    
    # Not fully implemented


def encode_categorical(df_pl:pl.DataFrame, encode_map:dict[dict[str,int]],cat_cols:list[str]):
    """
    Assumes a polars data frame and transform the categorical columns into integer levels.

    Parameters:
    ----------

    df_pl : pl.DataFrame
        A polars data frame with categorical data encoded as strings
    
    encode_map : dict[dict[str,int]] 
        A dict of dicts with encoding mapping of string levels into integer levels
    
    cat_cols : list[str]
        List of columns to encode

    Returns:
    -------

    df_pl_encoded : pl.DataFrame
        A polars data frame with categorical variables encoded as integers

    """

    df_pl_encoded = df_pl

    for col_name in cat_cols:
        df_pl_encoded = df_pl_encoded.with_columns(
            pl.col(col_name)
            .replace(encode_map[col_name])
            .cast(pl.Int64)
        )

    return df_pl_encoded


def decode_categorical(df_pl:pl.DataFrame, decode_map:dict[dict[int, str]], cat_cols:list[str]):
    """
    Assumes a polars data frame and transform the encoded data in categorical columns 
    back into string levels.

    Parameters:
    ----------

    df_pl : pl.DataFrame
        A polars data frame with categorical data encoded as strings
    
    decode_map : dict[dict[int, str]] 
        A dict of dicts with decoding mapping of integer levels into string levels
        -----
        Outer key: column name
        Inner dict: key, integer levels : value, string levels 
    
    cat_cols : list[str]
        List of columns to encode

    Returns:
    -------

    df_pl_encoded : pl.DataFrame
        A polars data frame with categorical variables encoded as integers

    """

    df_pl_encoded = df_pl

    for col_name in cat_cols:
        curr_map = decode_map[col_name]
        df_pl_encoded = df_pl_encoded.with_columns(
            pl.col(col_name)
            .map_elements(
                lambda x, m = curr_map: m.get(x, None),
                return_dtype = pl.String
            )
            .alias(col_name)
        )

    return df_pl_encoded


def generate_categorical_encoding_mappings(df_pl:pl.DataFrame, cat_cols: list[tuple[str,int]], nrow:int) -> dict[dict[str,int]]:
    """
    Creates a mapping for encoding categorical variables. Assumes columns with string data types are categorical variables
    coded with levels as strings.

    Note the missing values (null) will be casted as None 

    Parameters:
    ----------

    df_pl : pl.DataFrame
        A polars data frame
    cat_cols : list[tuple[str,int]]
        A list with tuples containing the name and the column index for categorical variables
    nrow : int
        Number of rows in df_pl

    Returns:
    -------

    categorical_encode_maps : dict[dict[str,int]]
        A dictionary with dictionaries that map uniques levels to integers

    Warns:
    -----

    Warning:
        If the number of levels in a given column are more than a third of the number of 
        rows, the client receives a warning to make sure a column with open ended text 
        was not passed to the model.

    """
    
    categorical_encode_maps = {}

    for col_name, i in cat_cols:
        unique_vals = df_pl[col_name].unique().sort().to_list() # Adding sort to make the mappings more deterministic
        
        if len(unique_vals) > nrow/3:
            warnings.warn(f"Check {col_name} does not contain open ended values. This implementation only process categorical levels...")

        map_name = col_name

        categorical_encode_maps[map_name] = {
            val: idx for idx, val in enumerate(unique_vals)
        }

    return categorical_encode_maps


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


def discrete_float_into_int(df_pl: pl.DataFrame, columns_to_fix: list[tuple[str,int]]) -> pl.DataFrame:
    """
    Casts float columns flagged as discrete as Int64

    Parameters:
    ----------

    df_pl : pl.DataFrame 
        A polars data frame
    columns_to_fix: list[tuple[str,int]]
        A list with tuples containing the column name and column index 
        of columns to modify

    Returns:
    -------

    A polars dataframe modified 
    """

    for col, i in columns_to_fix:
        df_pl = df_pl.with_columns(
            pl.col(col).cast(pl.Int64)
            )
    
    return df_pl


def generate_numerical_discrete_encoding_mappings(df_pl:pl.DataFrame, discrete_cols: list[tuple[str,int]]) -> dict[dict[int,int]]:
    """
    Creates a mapping for encoding numerical discrete variables. 

    Note the missing values (null) will be casted as None 

    Parameters:
    ----------

    df_pl : pl.DataFrame
        A polars data frame
    cat_cols : list[tuple[str,int]]
        A list with tuples containing the name and the column index for categorical variables

    Returns:
    -------

    categorical_encoding_maps : dict[dict[int,int]]
        A dictionary with dictionaries that map uniques levels to integers
    """
    
    numerical_discrete_encodng_maps = {}

    for col_name, i in discrete_cols:
        unique_vals = df_pl[col_name].unique().sort().to_list() # Adding sort to make the mappings more deterministic

        map_name = col_name

        numerical_discrete_encodng_maps[map_name] = {
            val: idx for idx, val in enumerate(unique_vals)
        }

    return numerical_discrete_encodng_maps


def generate_numeric_discrete_decoding_mappings(encoding_maps: dict[dict[int,int]]) -> dict[dict[int,int]]:
    """
    Assumes a mapping for encoding numerical discrete variables. Returns a mapping for decoding numerical
    discrete variables back into integers 

    Parameters:
    ----------

    encoding_maps : dict[dict[str,int]]
        An encoding map generated with generate_numerical_discrete_encoding_mappings()

    returns:
    -------

    decoding_map : dict[dict[int,int]]
        A decoding map to restore encoded data back into its original form
    """

    decoding_map = {
            outer_key: {v: k for k, v in inner_dict.items()}
            for outer_key, inner_dict in encoding_maps.items()
        }
    
    return decoding_map


def encode_numerical_discrete(df_pl:pl.DataFrame, encoding_map:dict[dict[int,int]], discrete_cols:list[str]) -> pl.DataFrame:
    """
    Assumes a polars data frame and transform the numerical discrete columns into integer levels.

    Parameters:
    ----------

    df_pl : pl.DataFrame
        A polars data frame with categorical data encoded as strings
    
    encoding_map : dict[dict[int,int]] 
        A dict of dicts with encoding mapping of discrete values into integer levels
    
    cat_cols : list[str]
        List of columns to encode

    Returns:
    -------

    df_pl_encoded : pl.DataFrame
        A polars data frame with numerical discrete variables encoded as integers

    """

    df_pl_encoded = df_pl

    for col_name in discrete_cols:
        
        df_pl_encoded = df_pl_encoded.with_columns(
            pl.col(col_name)
            .replace(encoding_map[col_name])
            .cast(pl.Int64)
        )

    return df_pl_encoded


def decode_numerical_discrete(df_pl:pl.DataFrame, decode_map:dict[dict[int, int]], discrete_cols:list[str]):
    """
    Assumes a polars data frame and transform the encoded data in numerical discrete columns 
    back into integer levels.

    Parameters:
    ----------

    df_pl : pl.DataFrame
        A polars data frame with numerical integers (like ZIP codes) data encoded as integer levels
    
    decode_map : dict[dict[int, int]] 
        A dict of dicts with decoding mapping of integer levels back into numerical integers 
        -----
        Outer key: column name
        Inner dict: key, integer levels : value, numerical integers
    
    discrete_cols : list[str]
        List of columns to encode

    Returns:
    -------

    df_pl_encoded : pl.DataFrame
        A polars data frame with integer levels variables decoded back to their original 
        form as numerical discrete

    """

    df_pl_encoded = df_pl

    for col_name in discrete_cols:

        curr_map = decode_map[col_name]

        df_pl_encoded = df_pl_encoded.with_columns(
            pl.col(col_name)
            .map_elements(
                lambda x, m = curr_map: m.get(x, None),
                return_dtype = pl.Int64
            )
            .alias(col_name)
        )

    return df_pl_encoded


@dataclass(frozen=True)
class BinDesign:
    """
    Stores discretization metadata for a single numerical column where
    BINNED encoding is used.

    Attributes:
    ----------

    n_bins : int
        Number of bins in design
    edges : list[float]
        Percentile-based bin edge values, length = n_bins + 1
    """
    n_bins: int
    edges: list[float]


def get_bin_designs(df_pl: pl.DataFrame, float_cols: list[tuple[str,int]]) -> dict[str,BinDesign]:
    """
    Assumes a polars data frame and calculates the number of bins and the corresponding edges  
    required for BINNED encoding for each column in float_cols. Returns this design as a dict. 

    Parameters:
    ----------

    df_pl : pl.DataFrame
        A polars data frame 
    float_cols : list[tuple[str,int]]
        A list of (column_name, column_index) pairs. The columns
        are a subset of the existing columns in df_pl 

    Returns:
    -------

    columns_bin_designs : dict[str,BinDesign]
        A dictionary with column name as key and a BinDesign 
        as a corresponding value. BinDesign is a container for
        the number of bins and the edges for a given column
    """

    if len(float_cols) == 0:
        warnings.warn("get_n_bins() received an empty list as an argument...")
        return {}

    MAX_BINS = 100

    cols_to_process_names, _ = zip(*float_cols)

    columns_bin_designs = {}
    
    for col_name in cols_to_process_names:
        n_unique_vals = len(df_pl[col_name].unique().to_list())
        n_bins = min(MAX_BINS, n_unique_vals)
        percentiles = np.linspace(0, 100, n_bins + 1)
        edges = np.percentile(df_pl[col_name].drop_nulls().to_numpy(), percentiles) # .drop_nulls() to avoid TypeError due to propagation of nulls
        
        unique_edges = np.unique(edges)  # remove duplicates from highly skewed data
        edges_diff = len(edges) - len(unique_edges)
        if edges_diff > 0:
            edges = unique_edges
            n_bins = len(unique_edges) - 1 # n_bins = n_bins - edges_diff 

        columns_bin_designs[col_name] = BinDesign(n_bins, edges)
    
    return columns_bin_designs


def generate_numerical_binned_encoding_mappings(
        binned_cols: list[tuple[str,int]], 
        bin_designs: dict[str,BinDesign]) -> dict[dict[tuple[float,float], int]]:
    """
    Creates a mapping for encoding collumns following the numerical binned strategy. 

    Note the missing values (null) will be casted as None 

    Parameters:
    ----------

    binned_cols : list[tuple[str,int]]
        A list with tuples containing the name and index of each column to 
        be processed as key-vaue pairs
    bin_designs : dict[str,BinDesign[int, list[float]]]
        A dict with column names as keys and bin designs as values for each column 
        passed in binned_cols.
        The designs are abstrated in BinDesign instances, each of which stores 
        the number of bins in the n_bins attribute and a list of edges in the
        edges attribute  

    Returns:
    -------

    categorical_encoding_maps : dict[dict[tuple[float,float], int]]
        A dictionary containing dictionaries that map intervals to integers.
        Outer key: column name
        Inner dict: maps (lower_edge, upper_edge) tuples to int category index
                    None mapped to  0  (reserved for missing values)

    Raises:
    ------

    ValueError
        If the number of bin designs does not match the number of columns
        passed in binned_cols there will be columns that will not have 
        rules to be processed
    """

    if len(binned_cols) == 0:
        logging.info("No columns following the BINNED encoding strategy")
        return {}
    
    if len(binned_cols) > 0 and (len(bin_designs) != len(binned_cols)):
        raise ValueError(
            f"The number of bin designs ({len(bin_designs)}) does not match "
            f"number of columns following the BINNED strategy ({len(binned_cols)})"
            )
    

    numerical_binned_encoding_maps = {}

    for col_name, _ in binned_cols:
        # Extracting design 
        curr_col_bin_design = bin_designs[col_name]
        curr_n_bins = curr_col_bin_design.n_bins
        curr_edges = curr_col_bin_design.edges

        map_name = col_name

        # Initializing inner dict

        inner_mapping = {}
        
        inner_mapping[None] = 0 

        for i in range(curr_n_bins):
            interval = (curr_edges[i], curr_edges[i+1])
            inner_mapping[interval] = i + 1  # The first entry of the dict is None mapped to 0. Thus mapping is shifted by 1

        # Assigning the inner dict to the outer dict

        numerical_binned_encoding_maps[col_name] = inner_mapping
    
    return numerical_binned_encoding_maps


def generate_numeric_binned_decoding_mappings(encoding_maps: dict[dict[tuple[float,float], int]]) -> dict[dict[int, tuple[float,float]]]:
    """
    Assumes a mapping for encoding columns with the BINNED strategy. Returns a mapping for decoding numerical
    binned variables back into bins with float edges 

    Parameters:
    ----------

    encoding_maps : dict[dict[tuple[float,float], int]]
        An encoding map generated with generate_numerical_binned_encoding_mappings()

    returns:
    -------

    decoding_map : dict[dict[int, tuple[float,float]]]
        A decoding map to restore encoded data back into its intermediate form.
        Outer key: column name
        Inner dict: maps integer level back to (lower_edge, upper_edge) tuples 
                    0 mapped to  None  (reserved for missing values)
    """

    decoding_map = {
            outer_key: {v: k for k, v in inner_dict.items()}
            for outer_key, inner_dict in encoding_maps.items()
        }
    
    return decoding_map


def find_bin(num_to_bin: float, edges: list[float]) -> tuple[float,float] | None:
    """
    Heper function to find the bin to sort a given number using binary search

    Parameters:
    ----------

    num_to_bin : float 
        A given value of a column to be uncoded using the BINNED strategy
    edges : list[float]
        A list of edges in the BinDesign for a given column

    Returns:
    -------

    key_for_mapping : tuple[float,float]
        The key to query the dictionary mapping for encoding a given column
    """

    if num_to_bin is None:
        return None  # None maps to 0 in encoding_map

    low = 0
    high = len(edges)

    # Binary Search to find the index
    while low < high:
        mid = (low + high) // 2
        if edges[mid] <= num_to_bin: # The binary search uses < but since the interval is open on the right (upper edge) <= is necessary
            low = mid + 1
        else:
            high = mid
    
    # Including bounds protection in case num_to_bin is below the first edge or above the last edge
    upper_edge = min(low, len(edges) - 1)
    lower_edge = max(upper_edge - 1, 0)

    return (float(edges[lower_edge]), float(edges[upper_edge]))


def encode_numerical_binned(
        df_pl:pl.DataFrame, 
        encoding_map:dict[dict[tuple[float,float],int]], 
        binned_cols:list[str], 
        bin_designs: dict[str,BinDesign]) -> pl.DataFrame:
    """
    Assumes a polars data frame and applies a transformation to the columns following BINNED encoding
    strategy.

    Parameters:
    ----------

    df_pl : pl.DataFrame
        A polars data frame with float columns to be encoded using the BINNED strategy
    
    encoding_map : dict[dict[tuple[float,float],int]]
        A dict of dicts with the encoding mapping for each column that follow the
        BINNED strategy. The mappig uses the key of the inner dict to first bin 
        float values and then map them into aninteger level.
        ----------------------
        Outer key: column name
        Inner dict: maps (lower_edge, upper_edge) tuples to int category index
                    None mapped to  0  (reserved for missing values)
    
    binned_cols : list[str]
        List of columns to encode

    Returns:
    -------

    df_pl_encoded : pl.DataFrame
        A polars data frame with numerical variables encoded as integers levels
        following the BINNED strategy

    """

    df_pl_encoded = df_pl

    for col_name in binned_cols:
        curr_col_enc_design = bin_designs[col_name]
        curr_col_edges = curr_col_enc_design.edges
        curr_col_mapping = encoding_map[col_name]

        df_pl_encoded = df_pl_encoded.with_columns(
            pl.col(col_name)
            .map_elements(
                lambda x, edges = curr_col_edges, col_map = curr_col_mapping: (
                    col_map[find_bin(x, edges)]
                ),
                return_dtype = pl.Int64 
            )
            .alias(col_name)
        )

    return df_pl_encoded


def decode_numerical_binned(
        df_pl:pl.DataFrame, 
        decoding_map:dict[dict[int, tuple[float,float]]], 
        binned_cols:list[str]) -> pl.DataFrame:
    """
    Assumes a polars data frame and applies a transformation to the generated values in
    columns encoded using the BINNED strategy.

    Parameters:
    ----------

    df_pl : pl.DataFrame
        A polars data frame with float columns to be encoded using the BINNED strategy
    
    decoding_map : dict[dict[int, tuple[float,float]]]
        A dict of dicts with the decoding mapping for each column that followed the
        BINNED strategy. The mappig uses the key of the inner dict to map the integer 
        levels into a tupple with a lower and upper bound.
        ----------------------
        Outer key: column name
        Inner dict: maps integer levels to (lower_edge, upper_edge) tuples 
                    0 is mapped to None for missing values generated by the NN
    
    binned_cols : list[str]
        List of columns to decode

    Returns:
    -------

    df_pl_encoded : pl.DataFrame
        A polars data frame with generated integer levels decoded as float values for
        columns encoded following the BINNED strategy

    """

    df_pl_encoded = df_pl

    for col_name in binned_cols:

        curr_col_mapping = decoding_map[col_name]

        df_pl_encoded = df_pl_encoded.with_columns(
            pl.col(col_name)
            .map_elements(
                lambda x, col_map = curr_col_mapping: (
                    None if x == 0
                    else float(np.random.uniform(low = col_map[x][0], high = col_map[x][1]))
                ),
                return_dtype = pl.Float64 
            )
            .alias(col_name)
        )

    return df_pl_encoded


def pad_numeric_digit_col(string_numbers: list[str], max_len: int, direction: str) -> list[str]:
   """
   Assumes a list of numbers casted into strings and returns a list of string 
   containing the same numbers bud padded with zeros on the left for digits
   [direction == "left"] or right for decimals [direction == "right"]

   Parameters:
   ----------

   string_numbers : list[str]
        A list of numbers casted as strings to process
   max_len : int
        The maximun length of the string numbers. The every string number in 
        the list will be paddded with zeros up until the length of the padded
        number is equal to max_len
   direction : str, ["left", "right"]
        The direction of the padding. Left adds zeroes to the left of the 
        string number (digits case) and right adds the zeros to the right of
        the number (decimal case).
    
   Returns:
   -------

   processed_number_strings : list[str]
        A list of padded string numbers

   Warns:
   -----

   If string_numbers is empty

   Raises:
   ------

   ValueError
      If the parameter direction is not equal to either 'right' or 'left'

   """

   if direction not in ["right", "left"]:
        raise ValueError(f"The direction parameter only accepts 'right' or 'left values. Client provided {direction}...")
    
   if len(string_numbers) == 0:
        warnings.warn("No numbers were provided to the padding step...")
        return []
   
   processed_number_strings = []

   for item in string_numbers:
       if pd.isna(item):
           padded_value = '0'*max_len
           processed_number_strings.append(padded_value)
       elif len(item) == max_len:
           processed_number_strings.append(item)
       else:
           pad_needed = max_len - len(item)
           generated_pad = '0'*pad_needed
           if direction == 'left':
               padded_value = generated_pad + item
               processed_number_strings.append(padded_value)
           else:
               padded_value = item + generated_pad
               processed_number_strings.append(padded_value)

   return processed_number_strings


def generate_sub_column_values(df_pl: pl.DataFrame, col_name: str) -> tuple[int, int, int, list[str]]:
    """
    Assumes a polars data frame and a column name with float values to be
    encoded following the DIGIT strategy, which generates sub columns.

    Parameters:
    ----------

    df_pl : pl.DataFrame
        A polars data fra,e
    col_name : str
        The name of the column with float values to process
    
    Returns:
    -------

    n_sub_cols, n_digit_sub_cols, n_decimal_sub_cols, sub_column_values_as_string : tuple[int, int, int, list[str]]
        
        n_sub_cols, the number of sub-columns required to encode col_name 
        following the DIGIT strategy

        n_digit_sub_cols, the number of digits in the encoding scheme
        
        n_decimal_sub_cols, the number of decimls in the encoding scheme
        
        sub_column_values_as_string, the values for each sub-column, starting
        with a binary column for the sign of the float, the float number 
        padded with zeros in both sides, and finally a binary column 
        to indicate missing values
    """

    list_to_process = df_pl[col_name].to_list()

    # The first sub column will have a vocabulary of {1, 0}. A value equal to 0 for positive numbers and 1 for negative numbers.
    # In this step we also collect the missing values. There values will have a column with a vocabulary of {0, 1} in the 
    # sub-columns scheme. The scheme for the sub-columns will be [sign, digit, ..., digit, decimal, ..., missing] with vocabularies
    # equal to [{0,1}, {0-9}, ..., {0-9}, {0-9}, {0, 1}] 
    signs_of_items =  [
    '0'     if pd.isna(item)           else
    '0'     if not np.isfinite(item)   else   
    '0'     if item >= 0               else
    '1'
    for item in list_to_process
    ]

    missing_in_items =  [
    '1'     if pd.isna(item)           else
    '1'     if not np.isfinite(item)   else   
    '0'     
    for item in list_to_process
    ]

    # The next step is convert ing the float values into strings for processing. Missing values and np.inf are both treated as missing 
    col_vals_as_strings = [
        f"{abs(item):.10f}" # Used f strings to about scientific notation issue that str(abs(item)) brings. Choosing 10f was arbitrary. 
        if not (pd.isna(item) or np.isinf(item)) 
        else None 
        for item in list_to_process
        ] 

    # Separating digints from decimals in two distinct lists
    digits_in_numbers, decimals_in_numbers = zip(*[
        tuple(item.split(".")) if item is not None else (None,None)
        for item in col_vals_as_strings
        ])

    # Claeaning trailing zeroes introduced in decimals by the f-string method
    decimals_in_numbers = [
        item.rstrip('0') or '0'
        if item is not None
        else None 
        for item in decimals_in_numbers
        ]

    # Computing length of digits
    len_digits_in_numbers = [
        len(item) 
        if item is not None
        else 0
        for item in digits_in_numbers
        ]
    n_digit_sub_cols = max(len_digits_in_numbers)

    # Computing length of decimals
    len_decimals_in_numbers = [
        len(item)
        if item is not None
        else 0 
        for item in decimals_in_numbers
        ]
    n_decimal_sub_cols = max(len_decimals_in_numbers)

    # Computing number of required sub-columns for the variable being processed
    n_sub_cols = n_digit_sub_cols + n_decimal_sub_cols + 1  + 1 # Adding one for a sub-column to store the sign and another to store NAs

    # Padding digits
    padded_digits = pad_numeric_digit_col(digits_in_numbers, n_digit_sub_cols, 'left')

    # Padding decimals
    padded_decimals = pad_numeric_digit_col(decimals_in_numbers, n_decimal_sub_cols, 'right')

    # Generating the values for all sub-columns as strings
    signs_and_pad_digits = [f"{prefix}{value}" for prefix, value in zip(signs_of_items, padded_digits)]
    signs_and_pad_digits_and_decimals = [f"{prefix}{value}" for prefix, value in zip(signs_and_pad_digits, padded_decimals)]
    sub_column_values_as_string = [f"{prefix}{value}" for prefix, value in zip(signs_and_pad_digits_and_decimals, missing_in_items)]

    return n_sub_cols, n_digit_sub_cols, n_decimal_sub_cols, sub_column_values_as_string  


def encode_numerical_digit(df_pl: pl.DataFrame, digit_cols: list[str]) -> tuple[pl.DataFrame, dict[str, tuple[int,int]]]:
    """"
    Assumes a polars data frame and a list of columns to process with float values
    to encode using the DIGIT strategy

    Parameters:
    ----------

    df_pl : pl.DataFrame 
        A polars data frame
    digit_cols: list[str]
        Columns to encode following the DIGIT strategy
    
    Returns:
    -------

    processed_df : pl.DataFrame
        A data frame with float columns encoded using the DIGIT strategy. The
        encoded column is removed and multiple sub-cloumns have been added 
        in its place
    
    digit_encodings: dict[str, tuple[int,int]]
        The encoding scheme for each column, with the column name as key
        and a tuple with the number of decimal and digits as values  
    """

    if len(digit_cols) == 0:
        return df_pl, {}
    
    processed_df = df_pl

    digit_encodings = {}

    for col_name in digit_cols:
        
        # First generate the padded values for a given column
        n_sub_cols, n_digit_sub_cols, n_decimal_sub_cols, sub_column_values_as_string = generate_sub_column_values(processed_df, col_name)

        # An important step is saving the number of digits and decimals in the encoding
        digit_encodings[col_name] = (n_digit_sub_cols, n_decimal_sub_cols)

        # Then insert that column in the polars data frame temporarily
        temp_col = f"_{col_name}_tmp"
        processed_df = processed_df.with_columns(
            pl.Series(name = temp_col, values = sub_column_values_as_string)
        )

        # This step generates the subcolumns from the temporal column
        processed_df = processed_df.with_columns([
            pl.col(temp_col)
            .str.slice(i, 1)
            .cast(pl.Int64)
            .alias(f"{col_name}_{i + 1}")
            for i in range(n_sub_cols)
        ])

        # Dropping the original column and the temp column
        processed_df = processed_df.drop([col_name, temp_col])
    
    return processed_df, digit_encodings


def decode_numerical_digit(
        df_pl: pl.DataFrame,
        digit_cols: list[str],
        digit_encodings: dict[str, tuple[int, int]]) -> pl.DataFrame:
    """
    Reconstructs float columns from their digit sub-column encoding.

    Encoding schema per original column:
        col_1:            sign indicator          (0 = positive, 1 = negative)
        col_2...col_n-1:  digit sub-columns       (integer digits then decimal digits - The vocabulary of each column is 0-9)
        col_n:            missing indicator       (1 = null, 0 = valid value)

    Parameters:
    ----------

    df_pl : pl.DataFrame
        A polars data frame with digit-encoded sub-columns
    digit_cols: list[str]
        Columns to decode following the DIGIT strategy
    digit_encodings : dict[str, tuple[int, int]]
        The encoding scheme for each column, with the column name as key
        and a tuple with the number of decimal and digits as values 

    Returns:
    -------
    df_decoded : pl.DataFrame
        Dataframe with digit sub-columns replaced by reconstructed float columns
    """
    
    df_decoded = df_pl

    for col_name in digit_cols:
        
        # Extracting the encoding information
        curr_col_encoding = digit_encodings[col_name]

        # Collecting all columns needed to decode the float value
        # all_columns_in_encoding = [item for item in df_decoded.columns if re.search(col_name, item, re.IGNORECASE)]
        all_columns_in_encoding =  [item for item in df_decoded.columns if item.startswith(f"{col_name}_")] # trying this approach that seems to be more precise

        # Extracting column name for sign and missing
        sign_col = all_columns_in_encoding[0]
        missing_col = all_columns_in_encoding[-1]

        # Pinpont the columns that are digits and those that are decimals
        digit_columns_in_encoding = all_columns_in_encoding[1:(curr_col_encoding[0] + 1)]
        decimal_columns_in_encoding = all_columns_in_encoding[curr_col_encoding[1] : -1]

        # Joining Digits
        digits_in_magnitude = [str(item) for item in df_decoded[digit_columns_in_encoding[0]].to_list()]

        for digit_sub_col in digit_columns_in_encoding[1:]:
            next_digit_list = [str(item) for item in df_decoded[digit_sub_col].to_list()]
            digits_in_magnitude = [f"{digit}{next_digit}" for digit, next_digit in zip(digits_in_magnitude, next_digit_list)]

        # Joining Decimals
        decimals_in_magnitude = [str(item) for item in df_decoded[decimal_columns_in_encoding[0]].to_list()]

        for decimal_sub_col in decimal_columns_in_encoding[1:]:
            next_decimal_list = [str(item) for item in df_decoded[decimal_sub_col].to_list()]
            decimals_in_magnitude = [f"{decimal}{next_decimal}" for decimal, next_decimal in zip(decimals_in_magnitude, next_decimal_list)]
        
        # Joining the string numbers
        magnitudes_as_str = [f"{digits}.{decimals}" for digits, decimals in zip(digits_in_magnitude, decimals_in_magnitude)]

        # Casting string numbers into float
        magnitudes = [float(item) for item in magnitudes_as_str]

        # Inserting magnitudes in temporal column
        temp_col = f"_{col_name}_tmp"
        df_decoded = df_decoded.with_columns(
            pl.Series(name = temp_col, values = magnitudes)
        )

        # Reconstructing float column
        df_decoded = df_decoded.with_columns(
            pl.when(pl.col(missing_col) == 1)
            .then(pl.lit(None).cast(pl.Float64))
            .when(pl.col(sign_col) == 0)
            .then(pl.col(temp_col).cast(pl.Float64))
            .otherwise((pl.col(temp_col) * -1.0).cast(pl.Float64))
            .alias(col_name)
        )

        # Drop all digit sub-columns
        df_decoded = df_decoded.drop(all_columns_in_encoding + [temp_col])

    return df_decoded


def encode_datetime(df_pl: pl.DataFrame, datetime_cols: list[str]) -> tuple[pl.DataFrame, dict[str,str]]:
    """
    Assumes a polars data frame and a list of columns to process with datetime values
    returns a processed data frame with encoded datetime types into subcolumns.

    It does not process pl.Duration columns.

    Parameters:
    ----------

    df_pl : pl.DataFrame 
        A polars data frame
    datetime_cols: list[str]
        Columns to encode into subcolumns
    
    Returns:
    -------

    processed_df : pl.DataFrame
        A data frame with datetime columns encoded into subcolumns. 
    datetime_type_mapping : dict[str,str]
        A helper dict mapping the specific datetime type of each 
        date/time columns
    """

    if len(datetime_cols) == 0:
        return df_pl, {}

    processed_df = df_pl
    datetime_type_mapping = {}

    COMPONENTS_BY_DTYPE = {
        pl.Datetime: ["year", "month", "day", "hour", "minute", "second", "millisecond"],
        pl.Date:     ["year", "month", "day"],
        pl.Time:     ["hour", "minute", "second", "millisecond"],
        # pl.Duration: ["total_seconds"], # Handle as a scalar value
    }
    
    for col_name in datetime_cols:
    
        # Step 1: Extract current column data type and recording in helper mapping
        col_datetime_type = type(processed_df[col_name].dtype)
        datetime_type_mapping[col_name] = col_datetime_type

        # Step 2: Find the valid components for a given column with a given date/time [sub]type
        if col_datetime_type == pl.Duration: # Guard in case a pl.Duration made it here
            continue
        elif col_datetime_type in COMPONENTS_BY_DTYPE:
            curr_col_valid_components = COMPONENTS_BY_DTYPE[col_datetime_type]
        else:
            raise ValueError("datetime_columns contains non-datetime data types...")
        
        # Step 3: Collecting components 

        components_in_curr_col = []

        for valid_component in curr_col_valid_components:

            component_unique_values = processed_df[col_name].dt.__getattribute__(valid_component)().drop_nulls().n_unique()
            
            if component_unique_values >= 1 and component_unique_values != 0:
                components_in_curr_col.append(valid_component)
        
        # Step 4: Creating sub columns

        if len(components_in_curr_col) > 0:

            processed_df = processed_df.with_columns([
                getattr(pl.col(col_name).dt, component)()
                .cast(pl.Int32)
                .alias(f"{col_name}_{component}")
                    for component in components_in_curr_col
            ])

            processed_df = processed_df.drop(col_name)
    
    return processed_df, datetime_type_mapping


def decode_datetime(
        df_pl: pl.DataFrame,
        datetime_columns: list[str],
        datetime_encoding_map: dict[str, str]) -> pl.DataFrame:
    """
    Reconstructs float columns from their digit sub-column encoding.

    Encoding schema per original column:
        col_1:            sign indicator          (0 = positive, 1 = negative)
        col_2...col_n-1:  digit sub-columns       (integer digits then decimal digits - The vocabulary of each column is 0-9)
        col_n:            missing indicator       (1 = null, 0 = valid value)

    Parameters:
    ----------

    df_pl : pl.DataFrame
        A polars data frame with datetime-encoded sub-columns
    datetime_columns : list[str]
        Columns to dencode by reconstructing data/time data spread into
         multiple sub-columns components
    datetime_encoding_map : dict[str, str]
        The encoding scheme for each column, with the column name as key
        and a the data type  of the column in the original data set  

    Returns:
    -------
    df_decoded : pl.DataFrame
        Dataframe with pl.Datetime and pl.Date sub-columns have been reconstructed 
        and replaced by individual columns
    """

    df_decoded = df_pl

    for col_name in datetime_columns:

        encoding_map = datetime_encoding_map
        curr_col_type = encoding_map[col_name] 
        
        curr_subcol_names = {c: pl.col(c) for c in df_decoded.columns if c.startswith(f"{col_name}_")} 

        def get(name, default):
            """Helper function to retrive each component value of datetime / time data types"""
            return curr_subcol_names.get(f"{col_name}_{name}", pl.lit(default))
        
        if curr_col_type == pl.Datetime:

            df_decoded = df_decoded.with_columns(
                pl.datetime(
                    get("year", 1970),
                    get("month", 1),
                    get("day", 1),
                    get("hour", 0),
                    get("minute", 0),
                    get("second", 0),
                    get("millisecond", 0),
                ).alias(col_name)
            )

        elif curr_col_type == pl.Date:

            df_decoded = df_decoded.with_columns(
                pl.date(
                    get("year", 1970),
                    get("month", 1),
                    get("day", 1),
                ).alias(col_name)
            )

        elif curr_col_type == pl.Time:

            df_decoded = df_decoded.with_columns(
                pl.time(
                    get("hour", 0),
                    get("minute", 0),
                    get("second", 0),
                    get("millisecond", 0),
                ).alias(col_name)
            )

        df_decoded = df_decoded.drop(list(curr_subcol_names.keys()))
    
    return df_decoded
