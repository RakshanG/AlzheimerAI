from database import get_connection

def migrate():
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            try:
                cur.execute("ALTER TABLE scans ADD COLUMN model_type VARCHAR(10) DEFAULT '2D'")
                print("Added column: model_type")
            except Exception as e:
                print(f"model_type already exists: {e}")
            try:
                cur.execute("ALTER TABLE scans ADD COLUMN prob_moderate FLOAT DEFAULT 0.0")
                print("Added column: prob_moderate")
            except Exception as e:
                print(f"prob_moderate already exists: {e}")
            cur.execute("UPDATE scans SET model_type = '2D' WHERE model_type IS NULL OR model_type = ''")
            print("Backfilled existing scans with model_type = 2D")
            conn.commit()
            print("Migration complete.")
    finally:
        conn.close()

if __name__ == "__main__":
    migrate()