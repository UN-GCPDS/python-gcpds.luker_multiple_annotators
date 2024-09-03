import sqlite3
import json
from datetime import datetime

class SQLiteDB:
    def __init__(self, db_name):
        self.db_name = db_name
        self.conn = sqlite3.connect(self.db_name)
        self.cursor = self.conn.cursor()

    def create_table(self, table_name, columns):
        columns_with_types = ", ".join([f"{col} {dtype}" for col, dtype in columns.items()])
        create_table_query = f"CREATE TABLE IF NOT EXISTS {table_name} ({columns_with_types})"
        self.cursor.execute(create_table_query)
        self.conn.commit()

    def insert_data(self, table_name, data):
        columns = ", ".join(data.keys())
        placeholders = ", ".join(["?" for _ in data])
        insert_query = f"INSERT INTO {table_name} ({columns}) VALUES ({placeholders})"
        self.cursor.execute(insert_query, tuple(data.values()))
        self.conn.commit()

    def read_data(self, table_name, conditions=None):
        query = f"SELECT * FROM {table_name}"
        if conditions:
            where_clause = " AND ".join([f"{col} = ?" for col in conditions])
            query += f" WHERE {where_clause}"
            self.cursor.execute(query, tuple(conditions.values()))
        else:
            self.cursor.execute(query)
        
        return self.cursor.fetchall()

    def update_data(self, table_name, data, conditions):
        set_clause = ", ".join([f"{col} = ?" for col in data])
        where_clause = " AND ".join([f"{col} = ?" for col in conditions])
        update_query = f"UPDATE {table_name} SET {set_clause} WHERE {where_clause}"
        self.cursor.execute(update_query, tuple(data.values()) + tuple(conditions.values()))
        self.conn.commit()

    def delete_data(self, table_name, conditions):
        where_clause = " AND ".join([f"{col} = ?" for col in conditions])
        delete_query = f"DELETE FROM {table_name} WHERE {where_clause}"
        self.cursor.execute(delete_query, tuple(conditions.values()))
        self.conn.commit()

    def close(self):
        self.conn.close()


