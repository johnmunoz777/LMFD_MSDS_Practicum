import sqlite3
import os

# Get current working directory
os.getcwd()

# Connect to SQLite (creates 'capstone.db' if it doesn't exist)
conn = sqlite3.connect("capstone.db")  
cursor = conn.cursor()

# Drop the members table if it exists
cursor.execute("DROP TABLE IF EXISTS members")

# Recreate the members table with additional columns
cursor.execute("""
CREATE TABLE members(
    id INTEGER PRIMARY KEY,
    name TEXT,
    age INTEGER,
    date_of_birth TEXT,
    address TEXT,
    loyalty INTEGER,
    member_since TEXT,
    gender TEXT,
    email TEXT,
    phone_number TEXT,
    membership_type TEXT,
    status TEXT,
    occupation TEXT,
    interests TEXT,
    marital_status TEXT
)
""")

# Commit changes and close connection
conn.commit()
conn.close()

print("Database and table recreated successfully.")