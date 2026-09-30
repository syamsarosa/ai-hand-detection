from database.connection import connection

def create_process(
    user, barcode, data_current, remaining, waktu_mulai, total_work_order
):
    """Menyimpan data awal proses ke tabel logproses."""

    conn = connection()

    if not conn:
        return None

    try:
        cursor = conn.cursor()

        query = """
            INSERT INTO logproses
            (
                user,
                id_barcode,
                data_current,
                remaining,
                waktu_mulai,
                total_work_order
            )
            VALUES (%s, %s, %s, %s, %s, %s)
        """

        data = (user, barcode, data_current, remaining, waktu_mulai, total_work_order)

        cursor.execute(query, data)
        conn.commit()

        return cursor.lastrowid

    except Exception as e:
        print(f"DB Error: {e}")
        return None

    finally:
        if conn.is_connected():
            cursor.close()
            conn.close()


def finish_process(
    process_id, waktu_selesai, timer, durasi_detik, sisa_bundle, work_order
):
    """Memperbarui data akhir proses pada tabel logproses."""

    conn = connection()

    if not conn:
        return False

    try:
        cursor = conn.cursor()

        query = """
            UPDATE logproses
            SET
                waktu_selesai = %s,
                timer = %s,
                durasi_detik = %s,
                sisa_bundle = %s,
                work_order = %s
            WHERE id = %s
        """

        data = (waktu_selesai, timer, durasi_detik, sisa_bundle, work_order, process_id)

        cursor.execute(query, data)
        conn.commit()

        return True

    except Exception as e:
        print(f"DB Update Error: {e}")
        return False

    finally:
        if conn.is_connected():
            cursor.close()
            conn.close()
