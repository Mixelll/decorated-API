import logging

import numpy as np
import pandas as pd
from sqlalchemy.types import UserDefinedType
from sqlalchemy.dialects.postgresql import JSON
from sqlalchemy.dialects.postgresql.base import ischema_names

import base_functions as bf


def pd_determine_sqlalchemy_type(series):
    """
    Determine the SQLAlchemy column type for a given pandas series.

    Args:
        series (pd.Series): The pandas series for which to determine the SQLAlchemy column type.

    Returns:
        SQLAlchemy Type: The SQLAlchemy column type that best fits the data in the series.
    """
    # Check for non-empty series and handle possible empty series
    if not series.dropna().empty:
        first_element = series.dropna().iloc[0]

        if isinstance(first_element, np.ndarray):
            # Assuming all lists or arrays in the column are of the same length and represent vectors
            vector_length = len(first_element)
            return PGVector(dimensions=vector_length)
        elif bf.contains_dict(first_element):
            return JSON
    return None  # Fallback to Text for any types not explicitly handled


class PGVector(UserDefinedType):
    """
    Custom SQLAlchemy type for interfacing with the PostgreSQL pgvector extension.
    """
    def __init__(self, dimensions=None):
        # The dimensions can be optional depending on how pgvector is used/setup.
        self.dimensions = dimensions

    def get_col_spec(self):
        """
        Return the PostgreSQL column specification using the pgvector type.
        """
        if self.dimensions:
            return f"VECTOR({self.dimensions})"
        return "VECTOR"

    def bind_expression(self, bindvalue):
        """
        Directly use the bind value without converting to array.
        This assumes that bindvalue is already a sequence of floats.
        """
        return bindvalue

    def result_processor(self, dialect, coltype):
        """
        Process the result value from SQL to Python when reading from the database.
        """
        def process(value):
            if value is not None:
                return list(value)  # Convert to a Python list if not already one.
            return value
        return process


ischema_names['vector'] = PGVector
