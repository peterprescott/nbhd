'''
Object-oriented database interactivity mediating the full power of SQL.
'''

import os

import pandas as pd
import geopandas as gpd
import networkx as nx

from sqlalchemy import create_engine
from sqlalchemy.exc import ResourceClosedError


class Base():
    'Hide the database code.'

    def __init__(self, user=None, pwd=None, host=None, port=None, db=None):
        if not user:
            user = os.environ.get('DB_USERNAME')
        if not pwd:
            pwd = os.environ.get('DB_PASSWORD')
        if not host:
            host = os.environ.get('DB_HOSTNAME')
        if not port:
            port = os.environ.get('DB_PORT')
        if not db:
            db = os.environ.get('DB_DATABASE')

        __url = f"postgres+psycopg2://{user}:{pwd}@{host}:{port}/{db}"
        print('Initializing database connection...')
        self.engine = create_engine(__url)
        count_tables = len(self.ls())
        print('Database connected!')

    def query(self, sql, spatial=False):
        'Query database.'

        if not spatial:
            return pd.read_sql(sql, self.engine)
        else:
            return gpd.read_postgis(sql, self.engine, geom_col='geometry')

    def ls(self):
        'List database tables'

        sql = '''SELECT table_name
          FROM information_schema.tables
         WHERE table_schema='public'
           AND table_type='BASE TABLE';'''

        return list(pd.read_sql(sql, self.engine).table_name)

    def info(self, table_name):
        sql = f'''
            SELECT
                a.attname as "Column",
                pg_catalog.format_type(a.atttypid, a.atttypmod) as "Datatype"
            FROM
                pg_catalog.pg_attribute a
            WHERE
                a.attnum > 0
                AND NOT a.attisdropped
                AND a.attrelid = (
                    SELECT c.oid
                    FROM pg_catalog.pg_class c
                        LEFT JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace
                    WHERE c.relname = '{table_name}'
                        AND pg_catalog.pg_table_is_visible(c.oid)
                );
            '''
        return self.query(sql)


    def rename(self, table, new_name):

        sql = f'''
        ALTER TABLE {table}
        RENAME TO {new_name}
        '''
        try:
            self.query(sql)
        except ResourceClosedError as e:
            # this is expected
            assert new_name in self.ls()
            print('Table successfully renamed!')


    def select(self, table):
        if table in self.ls():
            sql = f'SELECT * FROM {table}'
            return gpd.read_postgis(sql, self.engine, geom_col='geometry')
        else:
            print(f'`{table}` is not a table in the database.')
            return None
    

    def count(self, table, column=None):
        if column:
            column_values = self.distinct(column, 
                             table)[column].values
            counts = {}
            for value in column_values:
                counts[value] = self.query(f'''
                SELECT COUNT(*) FROM {table} 
                WHERE {column} LIKE '{value}'
                ''')['count'][0]
            return pd.DataFrame(counts, index=['counts']).T
        else:
            sql = f'SELECT (COUNT(*)) FROM {table}'
            return self.query(sql)
    

    def percentile(self, table, numeric, column=None, categorical=None, percentile=0.5):
        if categorical and categorical != 'Total':
            condition = f"WHERE {column} LIKE '{categorical}' "
        else:
            condition = ''
        sql = f'''
        SELECT PERCENTILE_CONT({percentile}) WITHIN GROUP(ORDER BY {numeric}) FROM {table}
        {condition}
        '''
        return self.query(sql)['percentile_cont'][0]


    def distinct(self, column, table):
        sql = f'SELECT DISTINCT({column}) FROM {table}'
        return self.query(sql)


    def spatial_sql(self, table, st_query, polygon, condition=None, crs=27700):
        if condition:
            condition_sql = f'AND {condition}'
        else:
            condition_sql = ''
        sql = f'''
             SELECT * from {table}
                WHERE ST_{st_query.capitalize()}(
                    geometry,
                    ST_GeomFromText('{polygon}', {crs})
                    )
                {condition_sql}
        '''
        return sql


    def knn(self,
            table1,
            table2,
            polygon,
            polygon2=None,
            condition1=None,
            condition2=None,
            t1_columns=None,
            t2_columns=None,
            k=1,
            max_distance=None,
            spatial_query='intersects'):
        '''
        Find k nearest-neighbours for results from table1 and table2 
        as returned by given spatial_query for given polygon.
        '''
        difference_condition = ''
        maximum_condition = ''
        if not polygon2:
            polygon2 = polygon

        sql_table1 = self.spatial_sql(table1, spatial_query, polygon, condition1)

        if table2 == table1:
            sql_table2 = sql_table1
            difference_condition = 'WHERE t1.id != t2.id'
        else:
            sql_table2 = self.spatial_sql(table2, spatial_query, polygon2, condition2)

        if max_distance:
            if difference_condition:
                start_word = 'AND'
            else:
                start_word = 'WHERE'
            maximum_condition = f'''{start_word} ST_Distance(
                        t1.geometry, t2.geometry
                        ) < {max_distance}'''
        t1_col_sql = ''
        if t1_columns:
            for col in t1_columns:
                t1_col_sql += f't1.{col}, '       
        t2_col_sql = ''
        if t2_columns:
            for col in t2_columns:
                t2_col_sql += f't2.{col}, '
            

        sql = f'''
        SELECT t1.id AS {table1}_id, {t1_col_sql}
               t1.geometry AS {table1}_geometry,
               t2.id AS {table2}_id, {t2_col_sql}
               t2.geometry AS {table2}_geometry,
               ST_Distance(t1.geometry, t2.geometry) AS dist
        FROM ({sql_table1}) AS t1
        CROSS JOIN LATERAL (
          SELECT t2.* 
          FROM ({sql_table2}) AS t2
          {difference_condition}
          {maximum_condition}
          ORDER BY t1.geometry <-> t2.geometry
          LIMIT {k}
        ) AS t2 ;
        '''

        return self.query(sql)


    def contains(self, table, polygon):
        sql = self.spatial_sql(table, 'contains', polygon)
        return self.query(sql, spatial=True)


    def crosses(self, table, polygon):
        sql = self.spatial_sql(table, 'crosses', polygon)
        return self.query(sql, spatial=True)


    def disjoint(self, table, polygon):
        sql = self.spatial_sql(table, 'disjoint', polygon)
        return self.query(sql, spatial=True)


    def equals(self, table, polygon):
        sql = self.spatial_sql(table, 'equals', polygon)
        return self.query(sql, spatial=True)


    def intersects(self, table, polygon):
        sql = self.spatial_sql(table, 'intersects', polygon)
        return self.query(sql, spatial=True)


    def overlaps(self, table, polygon):
        sql = self.spatial_sql(table, 'overlaps', polygon)
        return self.query(sql, spatial=True)


    def touches(self, table, polygon):
        sql = self.spatial_sql(table, 'touches', polygon)
        return self.query(sql, spatial=True)


    def within(self, table, polygon):
        sql = self.spatial_sql(table, 'within', polygon)
        return self.query(sql, spatial=True)


    def covers(self, table, polygon):
        sql = self.spatial_sql(table, 'covers', polygon)
        return self.query(sql, spatial=True)


    def coveredBy(self, table, polygon):
        sql = self.spatial_sql(table, 'coveredBy', polygon)
        return self.query(sql, spatial=True)


    def containsProperly(self, table, polygon):
        sql = self.spatial_sql(table, 'containsProperly', polygon)
        return self.query(sql, spatial=True)

