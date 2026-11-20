import os
import sqlite3

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "sqlite:////app/data/database.db"
)

DB_PATH = DATABASE_URL.replace("sqlite:///", "", 1)


def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn