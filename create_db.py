import pandas as pd
import sqlite3

csv_files = []

conn = sqlite3.connect('luker.db')

for csv_file in csv_files:
    df = pd.read_csv(csv_file)
    # name table after filename without extension
    table_name = csv_file[:-4]
    df.to_sql(table_name, conn, if_exists='replace', index=False)

conn.close()
