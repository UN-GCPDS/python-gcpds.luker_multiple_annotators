import pandas as pd
import sqlite3
import os

folder = 'data/chocolate'
csv_files = [f for f in os.listdir(folder) if f.endswith('.csv')]

conn = sqlite3.connect('data/chocolate/databases/chocolate.db')

for csv_file in csv_files:
    df = pd.read_csv(os.path.join(folder, csv_file))
    df.columns.values[0] = 'cod_sampler'
    # name table after filename without extension
    table_name = csv_file[:-4].replace(' ', '_')
    df.to_sql(table_name, conn, if_exists='replace', index=False)

conn.close()
