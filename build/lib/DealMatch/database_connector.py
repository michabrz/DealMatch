from dotenv import load_dotenv
import os
from sqlalchemy import create_engine

load_dotenv()

SQL_KEY = os.getenv('SQL_url')

def db_connector():
    sqlEngine = create_engine(SQL_KEY, pool_recycle=3306)
    dbConnection = sqlEngine.connect()
    return dbConnection
