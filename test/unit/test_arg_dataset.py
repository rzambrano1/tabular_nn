#!/usr/bin/python3
"""
Unit testing of encoding-decofing functions.
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

# Testing Modules 
# ----------------

import pytest

# Local Modules
# -------------

# Functions for categorical variables
from utils.argn_encoder_decoder import encode_categorical, generate_categorical_encoding_mappings, generate_categorical_decoding_mappings, decode_categorical

# Functions for numerical discrete
from utils.argn_encoder_decoder import discrete_float_into_int, generate_numerical_discrete_encoding_mappings, generate_numeric_discrete_decoding_mappings
from utils.argn_encoder_decoder import encode_numerical_discrete

# Functions for numerical BINNED and numerical DIGIT
from utils.argn_encoder_decoder import BinDesign, get_bin_designs, generate_numerical_binned_encoding_mappings, generate_numeric_binned_decoding_mappings, encode_numerical_binned

from utils.argn_encoder_decoder import encode_numerical_digit

# Functions for datetime

from utils.argn_encoder_decoder import encode_datetime

#########
# Tests #
#########

def test_generate_categorical_encoding_mappings(categorical_table):
    # Loads a correctly specified table with categorical variables
    test_cat_df, test_cat_cols = categorical_table

    test_mapping = generate_categorical_encoding_mappings(df_pl = test_cat_df, cat_cols = test_cat_cols, nrow = 5)

    expected_mapping = {
        'sex' : {'Female':0,'Male':1},
        'region' : {'Midwest':0,'Northeast':1, 'South':2, 'West':3},
        'educ' : {
            "Associate Degree":0, 
            "Bachelor's Degree":1, 
            "Doctorate":2,
            "High School":3, 
            "Master's Degree":4
        }
    }

    assert test_mapping == expected_mapping

    

def test_encode_categorical(categorical_table):
    # Loads a correctly specified table with categorical variables
    cat_df = categorical_table



# def test_categorical_encode_decode_roundtrip(categorical_table):
#     test_cat_df, test_cat_cols = categorical_table
#     cat_col_names = [item[0] for item in test_cat_cols]

#     encode_map = generate_categorical_encoding_mappings(test_cat_df, test_cat_cols, nrow=5)
#     decode_map = generate_categorical_decoding_mappings(encode_map)

#     encoded = encode_categorical(test_cat_df, encode_map, cat_col_names)
#     decoded = decode_categorical(encoded, decode_map, cat_col_names)

#     assert decoded.equals(test_cat_df)