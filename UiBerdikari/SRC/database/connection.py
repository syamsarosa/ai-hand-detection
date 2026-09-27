import os
from pathlib import Path

import mysql.connector
from mysql.connector import Error
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[3]

load_dotenv(PROJECT_ROOT / ".env")

APP_ENV = os.getenv("APP_ENV", "development")

DB_HOST = os.getenv("DB_HOST")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")

DB_PASSWORD = os.getenv("DB_PASSWORD", "")

if not all([DB_HOST, DB_NAME, DB_USER]):
    raise RuntimeError("Konfigurasi DB tidak lengkap.")

if APP_ENV == "production" and not DB_PASSWORD:
    raise RuntimeError("DB_PASSWORD HARUS ADA di production.")

def connection():
    """Membuat koneksi ke database."""
    try:
        conn = mysql.connector.connect(
            host=DB_HOST,
            database=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD
        )

        if conn.is_connected():
            return conn

    except Error as e:
        print(f"Error saat menghubungkan ke MariaDB: {e}")
        return None

if __name__ == "__main__":
    conn = connection()

    if conn:
        print("Database connection: OK")
        conn.close()
    else:
        print("Database connection: FAILED")
