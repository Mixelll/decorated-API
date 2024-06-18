import json
import inspect
import numpy as np
import pandas as pd
import functools as ft
import itertools as it
import decortools as dt

import postgresql_db as pgdb

from ib_insync import IB
from pytz import timezone

# from m_objects import JSONEncoderCustom
from credentials import postgres_db
from postgresql_db import connect as psql_connect, create_engine as psql_create_engine, composed_from_join
from inspect import signature


def create_db_connection(**kwargs):
    return psql_connect(**postgres_db.__dict__, **kwargs)


def create_db_engine(**kwargs):
    return psql_create_engine(**postgres_db.__dict__, **kwargs)


def ib_connect(**kwargs):
    ib = IB()
    ib.connect('127.0.0.1', 7497, clientId=1, **kwargs)
    return ib


postgres_inject_kw = dict(inject={'engine': create_db_engine}, execute_on_call=True)  # {0: 'df}
df_postgres_inject_kw = dict(**postgres_inject_kw, output_to_input_map={0: 1})  # {0: 'df}

upsert_df2db_decorator = dt.generate_decorator([pgdb.update_table_schema_sqa, pgdb.upsert_dataframe_sqa], **df_postgres_inject_kw)
return_df_rows_not_in_table = dt.generate_decorator(pgdb.return_df_rows_not_in_table, **df_postgres_inject_kw, overwrite_output=True)
get_table_as_df = dt.inject_execute_on_call(pgdb.get_table_as_df, **postgres_inject_kw)

"""superseded by dt.inject_execute_on_call"""
# @dt.inject_inputs(engine=dt.ExecuteFunctionOnCall(create_db_engine))
# @dt.ExecuteFunctionOnCall.decorator
# @dt.copy_signature(pgdb.get_table_as_df)
# def get_table_as_df(*args, **kwargs):
#     return pgdb.get_table_as_df(*args, **kwargs)

"""superseded by dt.generate_decorator"""
# @dt.inject_inputs(engine=dt.ExecuteFunctionOnCall(create_db_engine))
# @dt.ExecuteFunctionOnCall.decorator
# @dt.copy_signature(pgdb.upsert_df_add_columns_decorator)
# def upsert_df2db_decorator(*args, **kwargs):
#     pgdb.update_table_schema_sqa(*args, **kwargs)
#     pgdb.upsert_dataframe_sqa(*args, **kwargs)


