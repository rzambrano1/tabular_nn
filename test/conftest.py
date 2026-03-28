#!/usr/bin/python3
"""
Shared test fixtures
"""

# Boilerplate Modules
# -------------------


import polars as pl
import pytest

# Model Modules
# -------------

# -- None

# Fixture Definitions
# -------------------


@pytest.fixture(scope="session")
def categorical_table():
    cat_df = pl.DataFrame(
        [
            {"sex": "Male", "region": "Northeast", "educ": "Associate Degree"},
            {"sex": "Female", "region": "South", "educ": "High School"},
            {"sex": "Female", "region": "Midwest", "educ": "Bachelor's Degree"},
            {"sex": "Female", "region": "South", "educ": "Doctorate"},
            {"sex": "Male", "region": "West", "educ": "Master's Degree"},
        ]
    )

    cat_cols = [("sex", 0), ("region", 1), ("educ", 2)]

    return cat_df, cat_cols


# Helper Functions
# ----------------
