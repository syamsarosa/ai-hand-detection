"""Database repository for incorrect detection logging.

This module owns the database I/O for the logdeteksi table.
The application remains responsible for collecting values from GUI state
and managing the background worker/queue.
"""

from mysql.connector import Error
from database.connection import connection as get_db_connection

def insert_error_detection(data_to_insert):
    """Insert one incorrect-detection record into ``logdeteksi``.

    Parameters
    ----------
    data_to_insert:
        Tuple containing:
        id_proses, username, id_barcode, data_current,
        posisi_terdeteksi, kondisi, time_actual, timer

    Returns
    -------
    bool
        True when the insert succeeds, otherwise False.
    """
    conn = get_db_connection()
    if not conn:
        return False

    cursor = None
    try:
        cursor = conn.cursor()
        query = (
            "INSERT INTO logdeteksi "
            "(id_proses, username, id_barcode, data_current, "
            "posisi_terdeteksi, kondisi, time_actual, timer) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s)"
        )
        cursor.execute(query, data_to_insert)
        conn.commit()
        return True
    except Error as e:
        print(f"DB Log Detection Error: {e}")
        return False
    finally:
        if cursor is not None:
            cursor.close()
        if conn.is_connected():
            conn.close()
