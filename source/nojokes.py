import pandas as pd
import mysql.connector as sql

def mysql_import(db_name, table_name):
    db_connection = sql.connect(host='localhost', database=db_name, user='root', password='cts')
    db_cursor = db_connection.cursor()
    query = 'SELECT * FROM ' + table_name
    db_cursor.execute(query)
    table_rows = db_cursor.fetchall()
    df = pd.DataFrame(table_rows)
    return df


