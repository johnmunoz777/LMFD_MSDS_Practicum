import sqlite3
import os
import pandas as pd
def create_connection():
    return sqlite3.connect("capstone.db")
# Function to insert a new member or update if ID exists
def add_new_member(data):
    conn = create_connection()
    cursor = conn.cursor()
    # Check if the member already exists by ID
    cursor.execute("SELECT id FROM members WHERE id = ?", (data[0],))
    existing_member = cursor.fetchone()
    if existing_member:
        print(f"Member with ID {data[0]} already exists. Updating record...")
        cursor.execute("""
        UPDATE members SET 
            name = ?, age = ?, date_of_birth = ?, address = ?, loyalty = ?, member_since = ?, 
            gender = ?, email = ?, phone_number = ?, membership_type = ?, status = ?, 
            occupation = ?, interests = ?, marital_status = ? 
        WHERE id = ?
        """, data[1:] + (data[0],))
    else:
        print(f"Inserting new member: {data[1]}")
        cursor.execute("""
        INSERT INTO members (
            id, name, age, date_of_birth, address, loyalty, member_since, gender, 
            email, phone_number, membership_type, status,occupation, interests, marital_status
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, data)
    conn.commit()
    conn.close()
# Example adding new member
new_member =(13, "kevin", 33, "1993-11-15", "456 rex ave, TX", 90, "2018-08-14", "Male", "kevin@mail.com", "777-777-9012", "Gold", "Active", "Data Scientist", "Analytics", "Biking", "Single")
add_new_member(new_member)
