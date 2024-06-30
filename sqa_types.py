import logging
import numpy as np
import pandas as pd
import sqlalchemy as sa
from sqlalchemy.types import UserDefinedType
from sqlalchemy.dialects.postgresql import JSON
from sqlalchemy.dialects.postgresql.base import ischema_names

import base_functions as bf

import datetime
import decimal


def determine_sqlalchemy_type(python_type_or_object, allowed=None, conversion=False):
    """
    Maps Python types to SQLAlchemy types which automatically translates to
    PostgreSQL types during schema creation.

    Args:
        python_type_or_object: Python type or object to determine the SQLAlchemy type for
        allowed: List of allowed Python types to consider. If None, all types are allowed.

    :return: SQLAlchemy type object
    """
    inverse_string_mapping = {'pg_vector': [np.ndarray], 'json': [dict, list]}
    allowed_set = set()
    if allowed is not None:

        for key in allowed:
            if key in inverse_string_mapping:
                allowed_set.update(inverse_string_mapping[key])
            else:
                allowed_set.add(key)

    python_object = None
    if isinstance(python_type_or_object, type):
        python_type = python_type_or_object
    else:
        python_object = python_type_or_object
        python_type = type(python_type_or_object)
    list_lambda = lambda: sa.ARRAY(sa.String) if python_object is None else JSON() if bf.contains_dict(python_object) else sa.ARRAY(sa.String) if all(isinstance(i, str) for i in python_object) else sa.ARRAY(sa.Float)
    # pg_vector_lambda: PGVector() if python_object is None else PGVector(len(python_object)) if all(isinstance(i, float) for i in python_object) else sa.ARRAY(sa.Float)
    ndarray_lambda = lambda: PGVector(dimensions=len(python_object)) if python_object is not None else PGVector()
    type_mapping = {
        int: sa.Integer,
        float: sa.Float,
        str: sa.String,
        bytes: sa.LargeBinary,
        bool: sa.Boolean,
        datetime.datetime: sa.DateTime,
        datetime.date: sa.Date,
        decimal.Decimal: sa.Numeric,
        list: list_lambda,
        np.ndarray: ndarray_lambda,
        dict: JSON,
    }
    convert_dict = {
        np.ndarray: list,
    }
    if allowed_set and python_type not in allowed_set:
        return (None, convert_dict.get(python_type)) if conversion else None
    out = type_mapping.get(python_type, sa.String)  # Default to String if type is unknown
    if callable(out):
        out = out()
    if conversion:
        return out, convert_dict.get(python_type)
    return out


def pd_determine_sqlalchemy_special_types(series, **kwargs):
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
        # check for pg_vector or JSON
        return determine_sqlalchemy_type(first_element, allowed=['pg_vector', 'json'], **kwargs)


def pd_determine_sqlalchemy_types(series, **kwargs):
    # Check for non-empty series and handle possible empty series
    if not series.dropna().empty:
        first_element = series.dropna().iloc[0]
        return determine_sqlalchemy_type(first_element, **kwargs)


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
