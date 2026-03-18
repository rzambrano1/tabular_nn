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

from typing import Protocol

import warnings

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