import sqlite3
import os
import pandas as pd
# Get current working directory
os.getcwd()
# Connect to SQLite (creates 'capstone.db' if it doesn't exist)
conn = sqlite3.connect("capstone.db")  
cursor = conn.cursor()
# Sample data for 12 Volunteers
data = [
    (1, "angela", 32, "1992-05-14", "123 Maple St, NY", 85, "2018-06-12", "Female", "angela@mail.com", "123-456-7890", "Premium", "Active", "Software Engineer", "Tech, Hiking", "Single"),
    (2, "classmate", 40, "1984-09-23", "456 Oak Ave, CA", 120, "2015-03-25", "Male", "classmate@mail.com", "234-567-8901", "Gold", "Active", "Marketing Manager", "Reading, Tennis", "Married"),
    (3, "giuliana", 24, "2000-11-08", "789 Pine Rd, TX", 60, "2020-10-10", "Female", "giuliana@mail.com", "345-678-9012", "Silver", "Active", "Graphic Designer", "Art, Photography", "Single"),
    (4, "javier", 40, "1987-12-01", "147 Elm St, FL", 95, "2017-05-22", "Male", "javier@mail.com", "456-789-0123", "Platinum", "Inactive", "Accountant", "Finance, Running", "Married"),
    (5, "john", 37, "1979-03-08", "369 Cedar Blvd, WA", 200, "2012-11-15", "Male", "john@mail.com", "567-890-1234", "Platinum", "Active", "Entrepreneur", "Business, Travel", "Married"),
    (6, "maite", 30, "1994-08-25", "258 Birch Ln, CO", 75, "2019-09-30", "Female", "maite@mail.com", "678-901-2345", "Gold", "Active", "Nurse", "Health, Volunteering", "Single"),
    (7, "mike", 50, "1974-04-12", "369 Maple Dr, AZ", 150, "2010-07-08", "Male", "mike@mail.com", "789-012-3456", "Premium", "Inactive", "Teacher", "Education, Gardening", "Divorced"),
    (8, "ron", 38, "1986-11-30", "741 Spruce Way, NV", 80, "2016-02-14", "Male", "ron@mail.com", "890-123-4567", "Silver", "Active", "Engineer", "DIY, Cars", "Married"),
    (9, "shanti", 29, "1995-06-22", "852 Walnut St, IL", 50, "2021-01-10", "Female", "shanti@mail.com", "901-234-5678", "Silver", "Active", "Consultant", "Travel, Yoga", "Single"),
    (10, "tom", 36, "1988-01-17", "123 Redwood Ct, MI", 90, "2015-12-05", "Male", "tom@mail.com", "123-555-7890", "Gold", "Active", "Project Manager", "Leadership, Sports", "Married"),
    (11, "vilma", 27, "1997-09-29", "234 Chestnut Rd, GA", 40, "2022-07-21", "Female", "vilma@mail.com", "234-666-8901", "Silver", "Active", "Student", "Technology, Music", "Single"),
    (12, "will", 31, "1993-11-15", "456 Poplar St, TX", 70, "2018-08-14", "Male", "will@mail.com", "345-777-9012", "Gold", "Active", "Data Analyst", "Analytics, Chess", "Single")
]
# Convert to DataFrame
columns = ["id", "name", "age", "date_of_birth", "address", "loyalty", "member_since", "gender", "email", "phone_number", "membership_type", "status", "occupation", "interests", "marital_status"]
df = pd.DataFrame(data, columns=columns)
# Insert data into the members table
df.to_sql("members", conn, if_exists="append", index=False)
# Commit changes and close connection
conn.commit()
conn.close()
print("Database and table recreated successfully. 12 members inserted successfully.")
