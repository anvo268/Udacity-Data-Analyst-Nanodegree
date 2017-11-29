import sqlite3
import pandas as pd
import re

def clean_cell(s):
    """Strips the weird unicode indicators from a cell value"""
    try:
        m = re.match(r"b['\"](.*)['\"]", s)
        try:
            return m.group(1)
        except AttributeError:
            print(s)
            return s
    # Means it's not a string    
    except TypeError:
        return s
    
    
def clean_dataframe(df):
    """Applies clean_cell across a whole dataframe"""
    df.columns = pd.Series(df.columns).apply(clean_cell)
    for col in df.columns:
        df[col] = df[col].apply(clean_cell)
        
    return df


def create_tables(c):
    """Creates the tables in the db. Takes a cursor."""
    create_nodes = """CREATE TABLE nodes (
        id INTEGER PRIMARY KEY NOT NULL,
        lat REAL,
        lon REAL,
        user TEXT,
        uid INTEGER,
        version INTEGER,
        changeset INTEGER,
        timestamp TEXT
    );"""

    create_nodes_tags = """CREATE TABLE nodes_tags (
        id INTEGER,
        key TEXT,
        value TEXT,
        type TEXT,
        FOREIGN KEY (id) REFERENCES nodes(id)
    );"""

    create_ways = """CREATE TABLE ways (
        id INTEGER PRIMARY KEY NOT NULL,
        user TEXT,
        uid INTEGER,
        version TEXT,
        changeset INTEGER,
        timestamp TEXT
    );"""

    create_ways_tags = """CREATE TABLE ways_tags (
        id INTEGER NOT NULL,
        key TEXT NOT NULL,
        value TEXT NOT NULL,
        type TEXT,
        FOREIGN KEY (id) REFERENCES ways(id)
    );
    """

    create_ways_nodes = """CREATE TABLE ways_nodes (
        id INTEGER NOT NULL,
        node_id INTEGER NOT NULL,
        position INTEGER NOT NULL,
        FOREIGN KEY (id) REFERENCES ways(id),
        FOREIGN KEY (node_id) REFERENCES nodes(id)
    );"""

    c.execute(create_nodes)
    c.execute(create_nodes_tags)
    c.execute(create_ways)
    c.execute(create_ways_tags)
    c.execute(create_ways_nodes)


def std_phone_number(num):
    """Uses a regex to standardize phone numbers. Meant to be applied to a series"""
    rgx = r'(\+1)?[-\s\(]*(\d{3})[-\s\)]*(\d{3})[-\s]*(\d{4})'
    try:
        m = re.match(rgx, num)
        return '({}) {}-{}'.format(m.group(2), m.group(3), m.group(4))
    except:
        return num


def load_data(name, conn):
    """Cleans the data and loads it into the database. Takes a database connection and the name of the csv"""
    path = 'CSVS/' + name + '.csv'
    df = pd.read_csv(path)
    df = clean_dataframe(df)

    # quick fix for bad names
    if name == 'ways_tags':
        df.loc[df.value == 'bleacher', 'value']= 'bleachers'

    # fix for phone numbers and zip codes
    if name in ('ways_tags', 'nodes_tags'):
        df.loc[df.key == 'phone', 'value'] = df.loc[df.key == 'phone'].value.apply(std_phone_number)

        idx = (df.key == 'postal_code') | (df.key == 'postcode')
        df.loc[idx, 'key'] = 'postcode'

    df.to_sql(name, conn, 'sqlite', if_exists='append', index=False)


def main():
    conn = sqlite3.connect('SloMap.db')
    c = conn.cursor()

    create_tables(c)

    load_data('nodes', conn)
    load_data('nodes_tags', conn)
    load_data('ways', conn)
    load_data('ways_tags', conn)
    load_data('ways_nodes', conn)

    conn.close()


if __name__ == '__main__':
    main()
