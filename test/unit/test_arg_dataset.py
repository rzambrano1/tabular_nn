#!/usr/bin/python3
"""
Unit testing of encoding-decofing functions.
"""

#################
# Session Setup #
#################

# Standard Library
# ----------------


import warnings

# Numerical & Data
# ----------------
import polars as pl

# Testing Modules
# ----------------
# Local Modules
# -------------
# Functions for categorical variables
from utils.argn_encoder_decoder import (
    encode_categorical,
    generate_categorical_decoding_mappings,
    generate_categorical_encoding_mappings,
)

# Functions for numerical discrete

# Functions for numerical BINNED and numerical DIGIT


# Functions for datetime


#########
# Tests #
#########


def test_generate_categorical_encoding_mappings(categorical_table):
    # Loads a correctly specified table with categorical variables
    test_cat_df, test_cat_cols = categorical_table

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        test_mapping = generate_categorical_encoding_mappings(
            df_pl=test_cat_df, cat_cols=test_cat_cols, nrow=5
        )

    # These assertions test the warnings
    assert len(caught) == 3
    warning_messages = [str(w.message) for w in caught]
    assert any("sex" in msg for msg in warning_messages)
    assert any("region" in msg for msg in warning_messages)
    assert any("educ" in msg for msg in warning_messages)

    expected_mapping = {
        "sex": {"Female": 0, "Male": 1},
        "region": {"Midwest": 0, "Northeast": 1, "South": 2, "West": 3},
        "educ": {
            "Associate Degree": 0,
            "Bachelor's Degree": 1,
            "Doctorate": 2,
            "High School": 3,
            "Master's Degree": 4,
        },
    }

    assert test_mapping == expected_mapping


def test_generate_categorical_decoding_mappings(categorical_table):

    test_cat_df, test_cat_cols = categorical_table

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        encode_mapping = generate_categorical_encoding_mappings(
            df_pl=test_cat_df, cat_cols=test_cat_cols, nrow=5
        )

    test_mapping = generate_categorical_decoding_mappings(encode_mapping)

    expected_mapping = {
        "sex": {0: "Female", 1: "Male"},
        "region": {0: "Midwest", 1: "Northeast", 2: "South", 3: "West"},
        "educ": {
            0: "Associate Degree",
            1: "Bachelor's Degree",
            2: "Doctorate",
            3: "High School",
            4: "Master's Degree",
        },
    }

    assert test_mapping == expected_mapping


def test_encode_categorical(categorical_table):

    test_cat_df, test_cat_cols = categorical_table

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        encode_mapping = generate_categorical_encoding_mappings(
            df_pl=test_cat_df, cat_cols=test_cat_cols, nrow=5
        )

    test_result = encode_categorical(
        df_pl=test_cat_df,
        encode_map=encode_mapping,
        cat_cols=[col_name for col_name, _ in test_cat_cols],
    )

    expedted_results = pl.DataFrame(
        [
            {"sex": 1, "region": 1, "educ": 0},
            {"sex": 0, "region": 2, "educ": 3},
            {"sex": 0, "region": 0, "educ": 1},
            {"sex": 0, "region": 2, "educ": 2},
            {"sex": 1, "region": 3, "educ": 4},
        ]
    )

    assert test_result.equals(expedted_results)


# def test_categorical_encode_decode_roundtrip(categorical_table):
#     test_cat_df, test_cat_cols = categorical_table
#     cat_col_names = [item[0] for item in test_cat_cols]

#     encode_map = generate_categorical_encoding_mappings(test_cat_df, test_cat_cols, nrow=5)
#     decode_map = generate_categorical_decoding_mappings(encode_map)

#     encoded = encode_categorical(test_cat_df, encode_map, cat_col_names)
#     decoded = decode_categorical(encoded, decode_map, cat_col_names)

#     assert decoded.equals(test_cat_df)
