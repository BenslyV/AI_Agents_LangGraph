import psycopg
import os
from dotenv import load_dotenv

# ============================================================
# Load environment variables
# ============================================================

load_dotenv()

# Get PostgreSQL password from .env
postgres_password = os.getenv("POSTGRES_PASSWORD")


# ============================================================
# Connect to PostgreSQL
# ============================================================

connection = psycopg.connect(
    host="localhost",
    port=5432,
    dbname="engineering_students",
    user="postgres",
    password=postgres_password
)

# ============================================================
# Create a cursor object
# ============================================================

cursor = connection.cursor()


# ============================================================
# Create the table
# ============================================================

table_info = """
CREATE TABLE egg_students (
    NAME VARCHAR(25),
    BRANCH VARCHAR(25),
    SECTION VARCHAR(10),
    MARKS INT
)
"""

cursor.execute(table_info)


# ============================================================
# Insert records
# ============================================================

cursor.execute("""
    INSERT INTO egg_students
    VALUES ('Rahul', 'Computer Science', 'A', 92)
""")

cursor.execute("""
    INSERT INTO egg_students
    VALUES ('Anjali', 'Electronics', 'B', 88)
""")

cursor.execute("""
    INSERT INTO egg_students
    VALUES ('Arun', 'Mechanical', 'A', 76)
""")

cursor.execute("""
    INSERT INTO egg_students
    VALUES ('Priya', 'Computer Science', 'B', 95)
""")

cursor.execute("""
    INSERT INTO egg_students
    VALUES ('Vikram', 'Civil', 'A', 81)
""")

cursor.execute("""
    INSERT INTO egg_students
    VALUES ('Bensly', 'ECe', 'A', 81)
""")


# ============================================================
# Display records
# ============================================================

print("The inserted engineering student records are:")

cursor.execute("""
    SELECT * FROM egg_students
""")

data = cursor.fetchall()

for row in data:
    print(row)


# ============================================================
# Commit changes
# ============================================================

connection.commit()


# ============================================================
# Close connection
# ============================================================

cursor.close()
connection.close()