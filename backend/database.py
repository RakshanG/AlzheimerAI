import pymysql
from pymysql.cursors import DictCursor

DB_CONFIG = {
    "host":     "localhost",
    "user":     "root",
    "password": "",
    "database": "neuroscan",
    "cursorclass": DictCursor
}

def get_connection():
    return pymysql.connect(**DB_CONFIG)