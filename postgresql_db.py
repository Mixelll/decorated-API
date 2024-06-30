import functools as ft
import inspect
import io
import itertools as it
import logging
from uuid import uuid4

import numpy as np
import pandas as pd
import psycopg as psg
import sqlalchemy as sqa
from psycopg import sql
from sqlalchemy import MetaData
from sqlalchemy.exc import SQLAlchemyError, NoSuchTableError, InvalidRequestError
import sqlalchemy.dialects.postgresql as pg

import extra_functions as ef
from sqa_types import pd_determine_sqlalchemy_special_types, pd_determine_sqlalchemy_types


def connect(*, dbname=None, user=None, password=None, **kwargs):
    return psg.connect(dbname=dbname, user=user, password=password, **kwargs)


def create_engine(*, dbname, user, password, **kwargs):
    return sqa.create_engine(f'postgresql+psycopg://{user}:{password}@localhost:5432/{dbname}', **kwargs)


def upsert_dataframe_sqa(engine, df, table, schema=None, primary_keys=None):
    """
    Perform an upsert operation (from a df) on a PostgreSQL table using SQLAlchemy.

    Args:
    engine (Engine): SQLAlchemy engine instance connected to the database.
    table_name (str): Name of the table to perform the upsert operation on.
    df (DataFrame): Pandas DataFrame containing the data to upsert.
    schema (str, optional): Schema name if the table is not in the default public schema.
    """

    if not isinstance(df, pd.DataFrame):
        raise ValueError("The DataFrame must be a Pandas DataFrame.")
    if df.empty:
        logging.warning("The DataFrame is empty.")
        return
    table_name = table
    metadata = MetaData(schema=schema)
    try:
        metadata.reflect(bind=engine, only=[table_name], schema=schema)
        table = metadata.tables.get(f"{schema}.{table_name}" if schema else table_name)
    except Exception as e:
        logging.warning(f"Failed to reflect table: {e}, creating table.")
        table = None

    if table is None:
        dtype_mapping = {}
        logging.info(f"Table '{table_name}' does not exist. Creating table...")
        for col_name, series in df.items():
            sql_type, convert_fn = pd_determine_sqlalchemy_special_types(series, conversion=True)
            if convert_fn is not None:
                df[col_name] = series.apply(convert_fn)
            if sql_type is not None:
                dtype_mapping[col_name] = sql_type
        df.to_sql(table_name, engine, schema=schema, index=False, if_exists='replace', dtype=dtype_mapping)
        if primary_keys:
            add_primary_key(engine, table_name, primary_keys)
        return

    keys = [key.name for key in table.primary_key]

    stmt = pg.insert(table).values(df.to_dict(orient='records'))
    update_dict = {c.name: c for c in stmt.excluded if c.name in df.columns}
    if not update_dict:
        raise ValueError("No columns from DataFrame found in the table for updating.")

    if keys:
        # Prepare update statement for conflict resolution
        stmt = stmt.on_conflict_do_update(
            index_elements=keys,
            set_=update_dict
        )

    with engine.connect() as conn:
        with conn.begin() as t:
            result = conn.execute(stmt)
            logging.info(f"Rows inserted/updated: {result.rowcount}")


def return_df_rows_not_in_table(engine, df, table_name, schema=None, primary_keys=None, suppress_error_no_table_exists=False):
    """
    Returns rows from a DataFrame that are not present in the specified database table.

    Args:
        engine (sqlalchemy.Engine): Database engine.
        df (pandas.DataFrame): DataFrame to compare against the database table.
        table_name (str): Name of the database table.
        schema (str, optional): Schema of the database table.
        primary_keys (list, optional): List of column names to consider as primary keys for matching rows.
        suppress_error_no_table_exists (bool, optional): Suppress or raise error if the table does not exist.

    Returns:
        pandas.DataFrame: Rows from the input DataFrame that are not found in the database table.
    """
    if not table_exists(engine, table_name, schema=schema):
        if suppress_error_no_table_exists:
            logging.warning(f"Table '{table_name}' does not exist in the database schema '{schema}'.")
            return df
        else:
            raise ValueError(f"Table '{table_name}' does not exist in the database schema '{schema}'.")
    convert_type = None
    if not isinstance(df, pd.DataFrame):
        obj_cls = df.__class__
        convert_type = lambda x: ef.convert_df_fn(x, obj_cls)
        df = pd.DataFrame(df)

    if primary_keys is None:
        metadata = MetaData(schema=schema)
        try:
            metadata.reflect(bind=engine, only=[table_name], schema=schema)
            table_sqa = metadata.tables.get(f"{schema}.{table_name}" if schema else table_name)
            keys = [key.name for key in table_sqa.primary_key]
        except Exception as e:
            keys = None
            if suppress_error_no_table_exists:
                logging.warning(f"Failed to reflect table: {e}")
            else:
                raise e
        primary_keys = keys if keys else [c for c in df.columns]

    if not isinstance(primary_keys, (list, tuple)):
        primary_keys = [primary_keys]

    df_keys = pd.DataFrame(df[primary_keys])
    temp_table = table_name + '_temp'
    table_names = [(schema, table_name), (schema, temp_table)]
    temp_table_id = I_2(table_names[1])
    upsert_dataframe_sqa(engine, df_keys, temp_table, schema=schema, primary_keys=primary_keys)
    comp = S('SELECT {} ').format(composed_columns(it.product([table_names[1]], primary_keys)))
    comp += composed_from_join(tables=table_names, using=primary_keys)
    try:
        result = pg_execute(engine, comp, commit=False, mogrify_print=False)
    except psg.errors.UndefinedTable as e:
        if suppress_error_no_table_exists:
            logging.warning(f"Table '{table_name}' does not exist in the database schema '{schema}'.")
            return df if convert_type is None else convert_type(df)
        else:
            raise e
    finally:
        pg_execute(engine, S('DROP TABLE {}').format(temp_table_id))

    df_result = pd.DataFrame(result, columns=primary_keys)
    merged_df = df_keys.merge(df_result, on=primary_keys, how='left', indicator=True)
    df_not_in_db = merged_df[merged_df['_merge'] == 'left_only'].drop(columns=['_merge'])
    df_final = df_not_in_db.merge(df, on=primary_keys, how='left')
    if convert_type is not None:
        df_final = convert_type(df_final)
    return df_final


# @ef.post_decorator_factory(output_to_input_map={0: 1}, overwrite_output=True)  # {0: 'df'}
# @ef.copy_signature(return_df_rows_not_in_table)
# def return_df_rows_not_in_table_decorator(*args, **kwargs):
#     return return_df_rows_not_in_table(*args, **kwargs)


def upsert_df_add_columns_sqa_old(engine, table, schema=None, primary_keys=None):

    def decorator_upsert(func):
        @ft.wraps(func)
        def wrapper(*args, **kwargs):
            # Call the function
            df = func(*args, **kwargs)
            if not isinstance(df, pd.DataFrame):
                raise ValueError("The function did not return a DataFrame.")

            update_table_schema_sqa(engine, df, table, schema=None)
            upsert_dataframe_sqa(engine, df, table, schema=schema, primary_keys=primary_keys)
            return df
        return wrapper
    return decorator_upsert


def update_table_schema_sqa(engine, df, table_name, schema=None, primary_keys=None):
    metadata = MetaData()
    try:
        metadata.reflect(bind=engine, only=[table_name], schema=schema)
        table = metadata.tables.get(f"{schema}.{table_name}" if schema else table_name)
    except (NoSuchTableError, InvalidRequestError) as e:
        logging.warning(f"Unable to reflect table '{table_name}' in the database schema '{schema}'. Error: {e}")
        return

    inspector = sqa.inspect(engine)
    existing_columns = {col['name']: col['type'] for col in inspector.get_columns(table_name, schema=schema)}

    for column, series in df.items():
        if column not in existing_columns:
            sql_type, convert_fn = pd_determine_sqlalchemy_types(series, conversion=True)
            if convert_fn is not None:
                df[column] = series.apply(convert_fn)
            alter_command = f"ALTER TABLE {table_name} ADD COLUMN {column} {sql_type.compile(dialect=engine.dialect)}"
            with engine.connect() as conn:
                with conn.begin() as t:
                    conn.execute(sqa.text(alter_command))
                    logging.info(f"Added new column '{column}' with type '{sql_type}' to table '{table_name}' in schema '{schema}'.")


def add_primary_key(engine, table_name, primary_keys):
    """
    Add a composite primary key to an existing table using a list of column names.

    Args:
    engine (Engine): SQLAlchemy engine instance connected to the database.
    table_name (str): Name of the table to alter.
    primary_keys (list): List of strings representing the column names that make up the primary key.
    """
    if isinstance(primary_keys, str):
        primary_keys = [primary_keys]
    sql_command = f"""ALTER TABLE {table_name} ADD CONSTRAINT {table_name}_pk PRIMARY KEY ({', '.join(primary_keys)}); """.replace('\n', ' ')

    with engine.connect() as conn:
        with conn.begin() as t:
            conn.execute(sqa.text(sql_command))


def pg_execute(conn_obj, query, params=None, commit=True, autocommit=False, as_string=False, mogrify_print=False, mogrify_return=False, return_cursor=False):
    """
    Execute a psycopg2 composable, possibly containing a placeholder -
    *sql.Placeholder* or `'%s'` for the `params`.

    :param conn_obj:
        :psycopg.connection: or :sqlalchemy.engine.base.Engine:,
        connection object to execute the query with

    :param query:
        :Composable:, query awaiting params

    :param params:
        :[str or NestedIterable]:, params to be passed to the query

    :param as_string:
        Whether to print the output of `conn.cursor.as_string(...)`

    :param mogrify_print:
        Whether to print the output of `conn.cursor.mogrify(...)`

    :param mogrify_return:
        Whether to return the output of `conn.cursor.mogrify(...)`

    :return:
        :[(...)]:, returns array of tuples (rows)
    """
    conn = get_conn_if_engine(conn_obj)
    # conn.autocommit = autocommit
    mogrify = mogrify_print or mogrify_return
    cur = psg.ClientCursor(conn) if mogrify else conn.cursor()
    # cur = psg.ClientCursor(conn)
    if as_string:
        print(query.as_string(conn))
    if mogrify:
        mog = cur.mogrify(query, params)
        if mogrify_print:
            print(mog)
        if mogrify_return:
            return mog
    cur.execute(query, params)
    if commit:
        conn.commit()
    if return_cursor:
        return cur
    if cur.description is not None:
        return cur.fetchall()


def vacuum_table(conn_or_engine, table, schema=None, analyze=False):
    conn = get_conn_if_engine(conn_or_engine)
    if schema:
        table = schema, table
    if analyze:
        query = S('VACUUM FULL ANALYZE {}').format(I_2(table))
    else:
        query = S('VACUUM FULL {}').format(I_2(table))
    pg_execute(conn, query, commit=True)


def get_table_as_df(engine, table_name, schema=None, columns=None, params=None, index=None, between=None, except_columns=None, limit=None):
    tbl = [schema, table_name]
    if params is None:
        params = []
    where_between = None if index is None or between is None else [index, between]
    fn_query = lambda *_a, **_k: composed_select_from_table(tbl, params=params, where_between=where_between, *_a, **_k) + (S(' LIMIT {}').format(limit) if limit else S(''))
    if columns is not None:
        if isinstance(columns, str):
            columns = [columns]
        query = fn_query(columns=columns)
    elif except_columns is not None:
        if isinstance(except_columns, str):
            except_columns = [except_columns]
        tbl_columns = get_table_column_names(engine, tbl)
        request_columns = [x for x in tbl_columns if x not in except_columns]
        query = fn_query(columns=request_columns)
    else:
        query = fn_query()
    return query_get_df(engine, query, psg_params=params, index=index)


def column_prev_date_part_ratio(sql_conn, ident, column, index='end_date', date_part='D', is_null=True,
                                denominator=None,
                                window_fn='FIRST_VALUE', orth=False, run_return_new_column=False):
    new_col = column + f'_pct_{date_part}' + (f'_{window_fn}' if window_fn != 'FIRST_VALUE' else '')
    if run_return_new_column:
        return new_col
    column_denominator = denominator if denominator else column
    # ALTER TABLE {ident} ADD COLUMN IF NOT EXISTS {new_col} double precision;
    # ALTER TABLE {ident} ALTER COLUMN {new_col} TYPE double precision;
    comp = S("""
    ALTER TABLE {ident} ADD COLUMN IF NOT EXISTS {new_col} double precision;
    UPDATE {ident} AS t
    SET {new_col} = 1 - t.{column} / CTE.closing_val
    FROM (
        WITH CTE AS (
            SELECT DISTINCT
                date_trunc({date_part}, {index} AT TIME ZONE 'America/New_York') AS closing_date_part,
                {window_fn}({column2}) OVER (
                    PARTITION BY date_trunc({date_part}, {index} AT TIME ZONE 'America/New_York')
                    ORDER BY {index} DESC
                    ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
                ) AS closing_val
            FROM {ident}
            {where}
            ORDER BY closing_date_part
        )
        SELECT
            LEAD(closing_date_part) OVER (ORDER BY closing_date_part) AS next_day,
            closing_val
        FROM CTE
        ORDER BY closing_date_part
    ) AS CTE
    WHERE {is_null} CTE.next_day = date_trunc({date_part}, t.{index} AT TIME ZONE 'America/New_York');

    """)
    is_null = S('{} is null AND'.format(new_col)) if is_null is None else S('')
    where = S('') if orth else S(
        'WHERE extract(hour from {} AT TIME ZONE \'America/New_York\') BETWEEN 9 AND 16').format(I(index))
    comp = comp.format(is_null=is_null, where=where, ident=I_2(ident), column=I(column),
                       column2=I(column_denominator), index=I(index), date_part=date_part,
                       new_col=I(new_col), window_fn=S(window_fn))
    return pg_execute(sql_conn, comp)


def divide_columns(sql_conn, tbl, columns, divider, schema=None):
    if schema is not None:
        if isinstance(tbl, (list, tuple)):
            tbl = list(tbl)
            tbl.insert(0, schema)
        else:
            tbl = [schema, tbl]
    if isinstance(columns, str):
        columns = [columns]
    for col in columns:
        cmp_update = composed_update(tbl, [[col, col, '$/$', '%s'], ])
        pg_execute(sql_conn, cmp_update, [divider], commit=True)


def join_indexed_tables(conn, table_names, columns, index, join=None, schema=None, start_date=None, end_date=None):
    if not isinstance(schema, list):
        schema = [schema] * len(table_names)

    comp = S('SELECT {}.{}, {} ').format(I(table_names[0]), I(index),
                                         composed_columns(it.product(table_names, columns), AS=True))
    comp += composed_from_join(tables=zip(schema, table_names), using=[index] * len(table_names), join=join)

    cb, execV = composed_between(start=start_date, end=end_date)
    comp += S('WHERE {} ').format(I(index)) + cb
    return pd.read_sql_query(conn.cursor().mogrify(comp, execV), conn)

def append_df_to_db_stdin(engine, ident, df, index=True):
    schema, tbl = ident
    conn = engine.raw_connection()
    df.head(0).to_sql(tbl, engine, if_exists='append', index=index, schema=schema)
    cur = conn.cursor()
    copy_df2db(cur, ident, df, index=index)
    conn.commit()


def copy_df2db(cur, df, ident, index=True):
    output = io.StringIO()
    df.to_csv(output, sep='\t', header=True, index=index)
    output.seek(0)
    ident = I_2(ident)
    with output as f:
        with cur.copy(S("COPY {} FROM STDIN DELIMITER '\t' CSV HEADER;").format(ident)) as copy:
            while data := f.read(100):
                copy.write(data)


coerce = lambda x, typ: x if isinstance(x, typ) else typ(x)


def c_(x, typ):

    def create(y):
        if isinstance(y, (tuple, list)):
            return typ(*y[1:]) if len(y) >= 2 and y[0] is None else typ(*y)
        else:
            return typ(y)

    return x if isinstance(x, typ) else create(x)


def query_get_df(conn_or_engine, query, index=None, psg_params=None):
    if 'sqlalchemy' in str(type(conn_or_engine)).lower():
        if isinstance(query, sql.Composable):
            query = pg_execute(conn_or_engine.raw_connection(), query, params=psg_params, mogrify_return=True)
        return pd.read_sql_query(query, con=conn_or_engine, index_col=index)
    else:
        df = pd.DataFrame(pg_execute(conn_or_engine, query, params=psg_params))
        df.set_index(index, inplace=True)
        return df


def get_conn_if_engine(conn_or_engine):
    if all(map(lambda x: x in str(type(conn_or_engine)).lower(),
               ['sqlalchemy', 'engine'])):
        return conn_or_engine.raw_connection()
    else:
        return conn_or_engine


#
"""
This section of the module is intended for the creation and execution of `psycopg3`
composables, representing queries-pending-values. The composables are further
composed together to produce PostgreSQL queries. It leverages nesting, parsing
and keywords to generate complex queries with properly escaped names.

Parameters are denoted by :param_or_type:, brackets in conjunction with this
notation have the same meaning as when used as a Python literal.

The list of major parameters recurring in the module:

:param columns:
    :[str or [str]]: List of strings[] representing column names
    passed to `composed_columns`.

:param schema:
    :str: representing schema name

:param returning:
    :[str or NestedIterable]: parsable by `composed_parse` representing
    returned expressions.

:param conflict:
    :[str]: representing column names following SQL's `ON CONFLICT`

:param nothing:
    Whether to `DO NOTHING` following SQL's `ON CONFLICT`

:param set:
    :[str: or Iterable]:, parsables passed to `composed_set`

:param parse:
    :Callable: passed to `composed_set` and `composed_separated`
    as `parse=parse`, usually `composed_parse` is used.

:param where:
    :[str or NestedIterable]:, parsable passed to `composed_parse`
"""


S = sql.SQL
I = sql.Identifier
P = sql.Placeholder()


def I_2(x):
    return c_(x, I)


def psg_operators():
    return S, I, P, I_2


approvedExp = ['primary key', 'foreign key', 'references', 'default', 'uuid',
                'on delete restrict', 'unique', 'timestamp with time zone',
                'on delete cascade', 'current_timestamp','double precision',
                'not null', 'json', 'bytea', 'timestamp', 'like', 'and',
                'or', 'join', 'inner', 'left', 'right', 'full', '', '.', ',',
                 '=', '||','++','||+', 'min', 'max', 'least', 'greatest', 'like',
                '/']


def composed_columns(columns, enclose=False, parse=None, literal=None, **kwargs):
    s = psg_operators()[0]
    if parse is None:
        parse = lambda x: composed_separated(x, '.', **kwargs)
    if isinstance(columns, str):
        columns = [columns]

    if columns is None:
        return s('*')
    else:
        comp = s(', ').join(map(parse, columns))
        if enclose:
            return s('({})').format(comp)
        return comp


def composed_parse(exp, safe=False, tuple_parse=composed_columns, **kwargs):
    """
    Parse a nested container of strings by recursively pruning according
    to the following rules:

    - Enclose expression with `'$'` to parse the string raw into the quary,
        only selected expressions are allowed. *
    - If exp is  `'%s'` or *sql.Placeholder* to parse into *sql.Placeholder*.
    - If exp is a tuple it will be parsed by `composed_columns`.
    - If exp is a dict the keys will be parsed by `composed_columns` only if
        exp[key] evaluates to True.
    - Else (expecting an iterable) `composed_parse` will be applied to
        each element in the iterable.

    :param safe:
        Wether to disable the raw parsing in *

    :param exp:
        :[:str: or :Iterable:] or str: List (or string) of parsables
        expressions (strings or iterables).

    :param enclose:
        :Bool: passed to `composed_columns` as `enclose=enclose`

    :param tuple_parse:
        :Callable: passed to `composed_columns` as `parse=tuple_parse`,
        usually `composed_columns` is used.

    :return:
        :Composable:
    """

    if isinstance(exp, str):
        if not safe and exp[0]=='$' and exp[-1]=='$':
            exp = exp.replace('$','')
            if exp.strip(' []()').lower() in approvedExp:
                returned =  S(exp)
            elif exp.strip(' []()') == '%s':
                e = exp.split('%s')
                returned = S(e[0]) + P + S(e[1])
            else:
                raise ValueError(f'Expression: {exp.strip(" []()")} not found in allowed expressions')
        elif exp.strip('$ []()') == '%s':
            e = exp.replace('$','').split('%s')
            returned = S(e[0]) + P + S(e[1])
        else:
            returned = I(exp)
    elif isinstance(exp, sql.Placeholder):
        returned = exp
    elif isinstance(exp, tuple):
        returned = tuple_parse(filter(ef.mbool, exp), **kwargs)
    elif isinstance(exp, dict):
        returned = tuple_parse(filter(ef.mbool, [k for k in exp.keys() if exp[k]]), **kwargs)
    else:
        expPrev = exp[0]
        for x in exp[1:]:
            if x == expPrev:
                raise ValueError(f"Something's funny going on - {x,x} pattern is repeated ")
            else:
                expPrev = x

        return sql.Composed([composed_parse(x, safe=safe, tuple_parse=tuple_parse) for x in filter(ef.mbool, exp)])

    return S(' {} ').format(returned)


def composed_insert(tbl, columns, returning=None, conflict=None,
                    nothing=False, set=None, parse=composed_parse):
    """
    Construct query with value-placeholders to insert a row into `"tbl"` or `"tbl[0]"."tbl[1]"`

    :return:
        :Composable:, query awaiting params
    """
    comp = S('INSERT INTO {} ({}) VALUES ({}) ').format(I_2(tbl), composed_separated(columns),
                                                        composed_separated(len(columns) * [P]))

    if conflict:
        if nothing:
            comp += S('ON CONFLICT ({}) DO NOTHING').format(I(conflict))
        else:
            if set is None:
                set = columns
            comp += S('ON CONFLICT ({}) DO UPDATE').format(I(conflict)) + composed_set(set, parse=parse)

    if returning is not None:
        comp += S(' RETURNING {}').format(composed_separated(returning, parse=parse))

    return comp


def composed_update(tbl, columns, returning=None, where=None,
                    parse=composed_parse):
    """
    Construct query with value-placeholders to insert a row into `"schema"."tbl"`

    :return:
        :Composable:, query awaiting params
    """

    comp = S('UPDATE {}').format(I_2(tbl)) + composed_set(columns)

    if where is not None:
        comp += S(' WHERE {}').format(parse(where))

    if returning is not None:
        comp += S(' RETURNING {}').format(parse(returning))

    return comp


def composed_create(tbl, columns, schema=None, like=None,
                    inherits=None, constraint=None, parse=composed_parse):
    """
    Create a table as `"schema"."tbl"`

    :param like:
        :[str or [str]]:, parsables passed to `composed_columns`

    :param inherits:
        :[str or [str]]:, parsables passed to `composed_columns`

    :param constraint:
        :[str or NestedIterable]:, table constraints
        parsable by `composed_parse`, passed to `composed_columns`

    :return:
        :Composable:, full create table query
    """

    if isinstance(columns[0], str):
        columns = [columns]
    comp = S('CREATE TABLE {}{} (').format(composed_dot(schema), I(tbl))

    if like is not None:
        comp += composed_columns(like, parse=lambda x:
            S('LIKE {} INCLUDING ALL, ').format(composed_separated(x, '.', )))

    if constraint:
        if isinstance(constraint[0], str): constraint = [constraint]
        comp += composed_columns(constraint, parse=lambda x:
            S('CONSTRAINT ({}), ').format(parse(x)))

    comp += composed_columns(columns, parse=parse) + S(') ')

    if inherits is not None:
        comp += S('INHERITS ({})').format(composed_columns(inherits))

    return comp


def composed_where(index):
    return S('WHERE {} ').format(I(index))


def composed_select_from_table(tbl, columns=None, where_between=None, params=None):
    """
    Select columns from table as `"schema"."tbl"`

    :return:
        :Composable:, full select query which can be further used to compose
    """
    query = S('SELECT {} FROM {} ').format(composed_columns(columns),I_2(tbl))
    if where_between is not None:
        between = where_between[1]
        btw = composed_between(start=between[0], end=between[1])
        query += composed_where(where_between[0]) + btw[0]
        params.extend(btw[1])

    return query


def composed_from_join(join=None, tables=None, columns=None, using=None, parse=composed_parse):
    # def I_2(x): composed_separated(x, '.')
    if columns is not None and not isinstance(columns, (list, tuple)):
        columns = [columns]
    if using is not None and not isinstance(using, (list, tuple)):
        using = [using]
    joinc = []
    for v in ef.multiply_iter(join, max(ef.iter_length(tables[1:], columns, using))):
        vj = '$'+v+'$' if v else v
        joinc.append(parse([vj, '$ JOIN $']))

    if tables:
        tables = list(tables)
        comp = S('FROM {} ').format(I_2(tables[0]))
        if using:
                for t, u, jo in zip(tables[1:], using, joinc):
                    comp += jo + S('{} USING ({}) ').format(I_2(t), composed_columns(u))
        elif columns:
            for t, co, jo in zip(tables[1:], columns, joinc):
                comp += jo + S('{} ').format(I_2(t))
                for j, c in enumerate(co):
                    comp += S('ON {} = {} ').format(I_2(c[0]), I_2(c[1]))
                    if j < len(co):
                        comp += S('AND ')
        else:
            for t in tables[1:]:
                comp += S('NATURAL ') + parse([join, '$ JOIN $']) + S('{} ').format(I_2(t))

    elif columns:
        columns = list(columns)
        comp = S('FROM {} ').format(I_2(columns[0][:-1]))
        for i in range(1, len(columns)):
            toMap = columns[i][:-1], columns[i-1], columns[i-1]
            comp += joinc[i-1] + S('{} ON {} = {} ').format(*map(I_2, toMap))
    else:
        raise ValueError("Either tables or columns need to be given")

    return comp


def composed_set(set_obj, parse=composed_parse):
    """
    Return a composable of the form `SET (...) = (...)`

    :param like:
        :[str or [str]]:, parsables passed to `composed_columns`

    :param inherits:
        :[str or [str]]:, parsables passed to `composed_columns`

    :param set_obj:
        :[str or NestedIterable]:, set table columns
        parsable by `composed_parse`, passed to `composed_columns` and
         `composed_separated`

    :return:
        :Composable:
    """
    if not set_obj:
        return S('')
    col, val = [], []
    for c in set_obj:
        if isinstance(c, (tuple, list)):
            if len(c)>1:
                col.append(c[0])
                val.append(c[1:])
            else:
                col.append(c[0])
                val.append(P)
        else:
                col.append(c)
                val.append(P)
    if len(col)>1:
        formatted = S(' SET ({}) = ({})')
    else:
        formatted = S(' SET {} = {}')
    return formatted.format(composed_columns(col),
                            composed_separated(val, parse=parse))


def composed_between(start=None, end=None):
    """
    Return a composable that compares values to `start` and `end`

    :param start:
        :str or datetime or numeric:

    :param end:
        :str or datetime or numeric:

    :return:
        :(Composable, Array):, composable and values passed to `pg_execute` are returned
    """
    s = psg_operators()[0]
    comp = s('')
    execV = []

    if start is not None and end is not None:
        comp += s('BETWEEN %s AND %s ')
        execV.extend([start, end])
    elif start is None and end is not None:
        comp += s('<= %s ')
        execV.append(end)
    elif start is not None:
        comp += s('>= %s ')
        execV.append(start)

    return comp, execV


def composed_dot(name):
    if name:
        if not isinstance(name, str):
            return [composed_dot(x) for x in name]
        return S('{}.').format(I(name))
    return S('')


def composed_separated(names, sep=', ', enclose=False, AS=False, parse=None):
    if parse is None:
        parse = composed_parse
    if isinstance(names, str):
        names = [names]
    names = list(filter(ef.mbool, names))
    if sep in [',', '.', ', ', ' ', '    ']:
        comp = S(sep).join(map(parse, names))
        if AS:
            comp += S(' ') + N(sep.join(names))
        if enclose:
            return S('({})').format(comp)
        return comp
    else:
        raise ValueError(f'Expression: "{sep}" not found in approved separators')


def upsert_df_to_db_temp_table(engine, ident, df, index=True):
    if isinstance(ident, str):
        ident = (None, ident)
    schema, tbl = ident
    df.head(0).to_sql(tbl, engine, if_exists='append', index=index, schema=schema)
    conn = engine.raw_connection()
    cur = conn.cursor()
    temp = I(tbl + '_' + str(uuid4())[:8])
    ident = I_2(ident)
    cur.execute(S('CREATE TEMP TABLE {} (LIKE {} INCLUDING ALL);').format(temp, ident))
    copy_df2db(cur, df, temp, index=index)
    # cur.copy(s("COPY {} FROM STDIN DELIMITER '\t' CSV HEADER;").format(temp), output)
    cur.execute(S('DELETE FROM {} WHERE ({index}) IN (SELECT {index} FROM {});')
        .format(ident, temp, index=composed_separated(tuple(df.index.names))))
    cur.execute(S('INSERT INTO {} SELECT * FROM {};').format(ident, temp))
    cur.execute(S('DROP TABLE {};').format(temp))
    conn.commit()

# def create_trigger(name, ident, func, **kwargs):
#
#     comp = S("""
#     CREATE OR REPLACE TRIGGER {}
#     AFTER TRUNCATE ON {}
#     FOR EACH STATEMENT
#     EXECUTE FUNCTION {}();
#     """).format(c_(name, I), c_(ident, I), c_(func, I))
#     return comp
#
# def ident2name(x, **kwargs):
#     if isinstance(x, (list, tuple)):
#         return '_'.join(x)
#     elif isinstance(x, str):
#         return x
#     else:
#         return str(uuid4())


def get_tableNames(conn, names, operator='like', not_=False, relkind=('r', 'v'),
                    case=False, schema=None, qualified=None):
    s = psg_operators()[0]
    relkind = (relkind,) if isinstance(relkind, str) else tuple(relkind)
    c, names = composed_regex(operator, names, not_=not_, case=case)
    execV = [relkind, names]
    if schema:
        execV.append((schema,) if isinstance(schema, str) else tuple(schema))
        a = s('AND n.nspname IN %s')
    else:
        a = s('')

    cursor = conn.cursor()
    cursor.execute(s('SELECT {} FROM pg_class c JOIN pg_namespace n ON \
                    n.oid = c.relnamespace WHERE relkind IN %s AND relname {} %s {};') \
        .format(composed_parse({'nspname': qualified, 'relname': True}, safe=True), c, a), execV)
    if qualified:
        return cursor.fetchall()
    else:
        return [x for x, in cursor.fetchall()]


def exmog(cursor, input):
    print(cursor.mogrify(*input))
    cursor.execute(*input)

    return cursor


def composed_regex(operator, names, not_, case):
    s = psg_operators()[0]
    if operator.lower() == 'like':
        c = s('LIKE') if case else s('ILIKE')
        c = s('NOT ')+c+s(' ALL') if not_ else c+s(' ANY')
        if isinstance(names, str):
            names = [names]
        names = (names,)
    elif operator.lower() == 'similar':
        c = s('NOT SIMILAR TO') if not_ else s('SIMILAR TO')
        if not isinstance(names, str):
            names = '|'.join(names)
    elif operator.lower() == 'posix':
        c = s('~')
        if not case:
            c += s('*')
        if not_:
            c = s('!') + c
        if not isinstance(names, str):
            names = '|'.join(names)

    return c, names


def table_exists(conn, name, schema='public'):
    if schema is None:
        schema = 'public'
    query = f"""
    SELECT EXISTS (
       SELECT FROM pg_catalog.pg_class c
       JOIN   pg_catalog.pg_namespace n ON n.oid = c.relnamespace
       WHERE  n.nspname = '{schema}'
       AND    c.relname = '{name}'
       AND    c.relkind = 'r'    -- only tables
       );
    """
    conn = get_conn_if_engine(conn)
    cur = conn.cursor()
    cur.execute(query)
    exists = cur.fetchone()[0]
    cur.close()

    return exists


def get_table_column_names(conn, name):
    conn = get_conn_if_engine(conn)
    column_names = []
    try:
        cur = conn.cursor()
        cur.execute(S('select * from {} LIMIT 0').format(I_2(name)))
        for desc in cur.description:
            column_names.append(desc[0])
        cur.close()
    except psg.Error as e:
        print(e)

    return column_names


def set_comment(conn, tbl, comment, schema=None):
    schema = composed_dot(schema)
    query = S('COMMENT ON TABLE {}{} IS %s').format(schema, I(tbl))
    return pg_execute(conn, query, params=[str(comment)])
