import pandas as pd
import sqlite3
import os

# Path to the folder containing CSV files
folder = 'data/chocolate'

# Connect to SQLite database (or create it)
conn = sqlite3.connect('data/chocolate/databases/chocolate_new.db')
cursor = conn.cursor()

# Get list of CSV files
csv_files = [f for f in os.listdir(folder) if f.endswith('.csv')]

for csv_file in csv_files:
    # Read CSV file to get schema
    df = pd.read_csv(os.path.join(folder, csv_file))
    df.columns.values[0] = 'cod_sampler'  # Set the first column name
    
    # Define table name from file name without extension
    table_name = csv_file[:-4].replace(' ', '_')  # Replace spaces with underscores
    
    # Generate SQL command to create table
    columns = ', '.join(f'"{col}" TEXT' for col in df.columns)
    create_table_sql = f'CREATE TABLE IF NOT EXISTS {table_name} ({columns});'
    
    # Execute the SQL command
    cursor.execute(create_table_sql)

# Commit changes and close the connection
conn.commit()
conn.close()