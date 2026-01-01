# Fungsi untuk set model aktif
import os
from datetime import datetime
import sqlite3


DB_PATH = "model_history.db"


def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
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
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
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


def save_model_history(filename, algorithm, n_factors, n_epochs, lr_all, reg_all, rmse, mae):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''INSERT INTO model_histories (filename, algorithm, n_factors, n_epochs, lr_all, reg_all, rmse, mae)
                 VALUES (?, ?, ?, ?, ?, ?, ?, ?)''',
              (filename, algorithm, n_factors, n_epochs, lr_all, reg_all, float(rmse), float(mae)))
    conn.commit()

    # Retrieve the last inserted record
    last_id = c.lastrowid
    c.execute('''SELECT id, filename, algorithm, n_factors, n_epochs, lr_all, reg_all, rmse, mae, created_at
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
            "created_at": row[9],
        }
    return None


def get_model_history(limit=20):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''SELECT id, filename, algorithm, n_factors, n_epochs, lr_all, reg_all, rmse, mae, created_at
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
            "created_at": row[9],
        })
    return history


def get_model_by_id(model_id):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''SELECT id, filename, algorithm, n_factors, n_epochs, lr_all, reg_all, rmse, mae, created_at
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
            "created_at": row[9],
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
                        mh.id, mh.filename, mh.algorithm, mh.n_factors, mh.n_epochs, mh.lr_all, mh.reg_all, mh.rmse, mh.mae, mh.created_at
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
                "created_at": row[11],
            }
        }
    return None


def delete_db():
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
        return True
    return False
