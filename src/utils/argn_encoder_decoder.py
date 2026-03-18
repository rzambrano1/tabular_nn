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

# Numerical & Data 
# ----------------

import numpy as np
import polars as pl
import pandas as pd

#########################
# Functions and Classes #
#########################

class encoding_decoding_engine_protocol:

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

