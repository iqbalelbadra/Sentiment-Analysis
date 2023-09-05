import sqlite3
import numpy as np

conn = sqlite3.connect('Data/temp.db', check_same_thread=False)

def create_table():
    conn.execute("""CREATE TABLE IF NOT EXISTS data (id INTEGER PRIMARY KEY AUTOINCREMENT, data_text char(1000), sentiment_pred char(1000))""")
    conn.commit()

def insert_to_table(value_1,value_2):
    #value_1 = value_1.encode('utf-8')
    #value_2_str = np.array2string(value_2, separator=',')
    value_2_str = value_2[0]
    query = f"INSERT INTO data (data_text, sentiment_pred) VALUES (?, ?);"
    conn.execute(query, (value_1, value_2_str))
    conn.commit()
   
