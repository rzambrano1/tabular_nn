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

from typing import Protocol
from typing import Generic, TypeVar

T = TypeVar("T")
U = TypeVar("U")

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

    for col, i in cat_cols:
        unique_vals = df_pl[:, i].unique().to_list()
        
        if len(unique_vals) > nrow/3:
            warnings.warn(f"Check {col} does not contain open ended values. This implementation only process categorical levels...")

        map_name = col

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

def generate_numerical_discrete_encoding_mappings(df_pl:pl.DataFrame, discrete_cols: list[tuple[str,int]]) -> dict[dict[str,int]]:
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

    categorical_encoding_maps : dict[dict[str,int]]
        A dictionary with dictionaries that map uniques levels to integers
    """
    
    numerical_discrete_encodng_maps = {}

    for col_name, i in discrete_cols:
        unique_vals = df_pl[:, i].unique().to_list()

        map_name = col_name

        numerical_discrete_encodng_maps[map_name] = {
            val: idx for idx, val in enumerate(unique_vals)
        }

    return numerical_discrete_encodng_maps

def generate_numeric_discrete_decoding_mappings(encoding_maps: dict[dict[str,int]]) -> dict[dict[int,str]]:
    """
    Assumes a mapping for encoding numerical discrete variables. Returns a mapping for decoding numerical
    discrete variables back into integers 

    Parameters:
    ----------

    encoding_maps : dict[dict[str,int]]
        An encoding map generated with generate_numerical_discrete_encoding_mappings()

    returns:
    -------

    decoding_map : dict[dict[int,str]]
        A decoding map to restore encoded data back into its original form
    """

    decoding_map = {
            outer_key: {v: k for k, v in inner_dict.items()}
            for outer_key, inner_dict in encoding_maps.items()
        }
    
    return decoding_map

def encode_numerical_discrete(df_pl:pl.DataFrame, encoding_map:dict[dict[str,int]], discrete_cols:list[str]):
    """
    Assumes a polars data frame and transform the numerical discrete columns into integer levels.

    Parameters:
    ----------

    df_pl : pl.DataFrame
        A polars data frame with categorical data encoded as strings
    
    encoding_map : dict[dict[str,int]] 
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
            n_bins = n_bins - edges_diff

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