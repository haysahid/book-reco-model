import os
from datetime import datetime
import sqlite3


DB_PATH = "model_history.db"


def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    # Tambahkan kolom reference dan created_by jika belum ada
    c.execute('''
        CREATE TABLE IF NOT EXISTS model_histories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT,
            algorithm TEXT,
            n_factors INTEGER,
            n_epochs INTEGER,
            lr_all REAL,
            reg_all REAL,
            rmse REAL,
            mae REAL,
            reference TEXT,
            created_by TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    # Alter table jika kolom reference belum ada (untuk migrasi)
    try:
        c.execute('ALTER TABLE model_histories ADD COLUMN reference TEXT')
    except sqlite3.OperationalError:
        pass  # Kolom sudah ada
    # Alter table jika kolom created_by belum ada (untuk migrasi)
    try:
        c.execute('ALTER TABLE model_histories ADD COLUMN created_by TEXT')
    except sqlite3.OperationalError:
        pass  # Kolom sudah ada
    c.execute('''
        CREATE TABLE IF NOT EXISTS active_model (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_history_id INTEGER,
            set_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(model_history_id) REFERENCES model_histories(id)
        )
    ''')
    conn.commit()
    conn.close()


def save_model_history(filename, algorithm, n_factors, n_epochs, lr_all, reg_all, rmse, mae, reference, created_by):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''INSERT INTO model_histories (filename, algorithm, n_factors, n_epochs, lr_all, reg_all, rmse, mae, reference, created_by)
                 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
              (filename, algorithm, n_factors, n_epochs, lr_all, reg_all, float(rmse), float(mae), reference, created_by))
    conn.commit()

    # Retrieve the last inserted record
    last_id = c.lastrowid
    c.execute('''SELECT id, filename, algorithm, n_factors, n_epochs, lr_all, reg_all, rmse, mae, reference, created_by, created_at
                 FROM model_histories WHERE id = ?''', (last_id,))
    row = c.fetchone()
    conn.close()
    if row:
        return {
            "id": row[0],
            "filename": row[1],
            "algorithm": row[2],
            "n_factors": row[3],
            "n_epochs": row[4],
            "lr_all": row[5],
            "reg_all": row[6],
            "rmse": row[7],
            "mae": row[8],
            "reference": row[9],
            "created_by": row[10],
            "created_at": row[11],
        }
    return None


def get_model_history(limit=20):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''SELECT id, filename, algorithm, n_factors, n_epochs, lr_all, reg_all, rmse, mae, reference, created_by, created_at
                 FROM model_histories ORDER BY id DESC LIMIT ?''', (limit,))
    rows = c.fetchall()
    conn.close()
    history = []
    for row in rows:
        history.append({
            "id": row[0],
            "filename": row[1],
            "algorithm": row[2],
            "n_factors": row[3],
            "n_epochs": row[4],
            "lr_all": row[5],
            "reg_all": row[6],
            "rmse": row[7],
            "mae": row[8],
            "reference": row[9],
            "created_by": row[10],
            "created_at": row[11],
        })
    return history


def get_model_by_id(model_id):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''SELECT id, filename, algorithm, n_factors, n_epochs, lr_all, reg_all, rmse, mae, reference, created_by, created_at
                 FROM model_histories WHERE id = ?''', (model_id,))
    row = c.fetchone()
    conn.close()
    if row:
        return {
            "id": row[0],
            "filename": row[1],
            "algorithm": row[2],
            "n_factors": row[3],
            "n_epochs": row[4],
            "lr_all": row[5],
            "reg_all": row[6],
            "rmse": row[7],
            "mae": row[8],
            "reference": row[9],
            "created_by": row[10],
            "created_at": row[11],
        }
    return None


def set_active_model(model_history_id):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    # Hapus entry lama agar hanya satu model aktif
    c.execute('DELETE FROM active_model')
    c.execute('INSERT INTO active_model (model_history_id) VALUES (?)',
              (model_history_id,))
    conn.commit()
    conn.close()


def get_active_model():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''SELECT am.model_history_id, am.set_at,
                        mh.id, mh.filename, mh.algorithm, mh.n_factors, mh.n_epochs, mh.lr_all, mh.reg_all, mh.rmse, mh.mae, mh.reference, mh.created_by, mh.created_at
                 FROM active_model am
                 JOIN model_histories mh ON am.model_history_id = mh.id
                 ORDER BY am.set_at DESC LIMIT 1''')
    row = c.fetchone()
    conn.close()
    if row:
        return {
            "model_history_id": row[0],
            "set_at": row[1],
            "model": {
                "id": row[2],
                "filename": row[3],
                "algorithm": row[4],
                "n_factors": row[5],
                "n_epochs": row[6],
                "lr_all": row[7],
                "reg_all": row[8],
                "rmse": row[9],
                "mae": row[10],
                "reference": row[11],
                "created_by": row[12],
                "created_at": row[13],
            }
        }
    return None


def delete_model_history(model_id):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    active_model = get_active_model()
    if active_model and active_model['model_history_id'] == model_id:
        c.execute('''SELECT id FROM model_histories
                     WHERE id != ?
                     ORDER BY id DESC LIMIT 1''', (model_id,))
        row = c.fetchone()
        if row:
            new_active_id = row[0]
            c.execute('DELETE FROM active_model')
            c.execute('INSERT INTO active_model (model_history_id) VALUES (?)',
                      (new_active_id,))
        else:
            c.execute('DELETE FROM active_model')

    c.execute('DELETE FROM model_histories WHERE id = ?', (model_id,))

    conn.commit()
    affected_rows = c.rowcount
    conn.close()
    return affected_rows > 0


def delete_db():
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
        return True
    return False
