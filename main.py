from model_db import (init_db, save_model_history, get_model_history,
                      set_active_model, get_active_model, get_model_by_id, delete_model_history)
import os
import logging
import json
import shutil
import pickle
import pandas as pd
from typing import Optional
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from surprise import Reader, SVD, Dataset, accuracy
from surprise.model_selection import train_test_split
from datetime import datetime
from surprise.model_selection import GridSearchCV


init_db()

app = FastAPI(
    title="Sistem Rekomendasi Buku API",
    description="Backend untuk training SVD dan prediksi rekomendasi buku dengan fitur Cold Start."
)

MODEL_DIR = "models"
DATASET_DIR = "datasets"

# Buat folder dataset jika belum ada
os.makedirs(DATASET_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


def load_model(model_path=None):
    if model_path is None:
        # Get active model
        active_model = get_active_model()
        if active_model:
            model_path = os.path.join(
                MODEL_DIR, active_model["model"]["filename"])
        else:
            # Get latest model
            model = get_model_history(limit=1)
            if not model:
                return None
            model_path = os.path.join(MODEL_DIR, model[0]["filename"])

    if model_path and os.path.exists(model_path):
        with open(model_path, 'rb') as f:
            return pickle.load(f)

    return None


# Global variable untuk menyimpan model yang sedang aktif
model = load_model()


@app.post("/train", tags=["Model Training"])
async def train_model(
    books_file: UploadFile = File(
        ..., description="File Excel berisi daftar buku (kolom: id, title, author)"),
    transactions_file: UploadFile = File(
        ..., description="File Excel transaksi (kolom: user_id, book_id, quantity, rating)"),
    n_factors: int = Form(100, description="Jumlah faktor laten"),
    n_epochs: int = Form(20, description="Jumlah iterasi training"),
    lr_all: float = Form(0.005, description="Learning rate"),
    reg_all: float = Form(0.02, description="Regularization term"),
    reference: str = Form(
        "rating", description="Referensi rekomendasi: 'rating' atau 'transaction'"),
    created_by: str = Form(
        "manual", description="Penanda model dibuat oleh siapa: 'manual' atau 'auto'")
):
    logging.info(f"Menerima permintaan training model baru: {reference}")

    """
    Endpoint untuk mengunggah dataset baru, melakukan training, dan evaluasi model.
    """
    try:
        logging.info("Memulai proses training model...")

        # 1. Simpan File ke Lokal
        books_path = os.path.join(DATASET_DIR, "books.xlsx")
        trans_path = os.path.join(DATASET_DIR, "transaction_items.xlsx")

        with open(books_path, "wb") as buffer:
            shutil.copyfileobj(books_file.file, buffer)
        with open(trans_path, "wb") as buffer:
            shutil.copyfileobj(transactions_file.file, buffer)

        logging.info("File dataset berhasil disimpan.")

        # 2. Persiapan Data
        df_books = pd.read_excel(books_path)
        df_trans = pd.read_excel(trans_path)

        df_books.drop(columns=['no'])
        df_trans.drop(columns=['no'])

        # Ganti nama kolom id untuk menghindari konflik
        df_books.rename(columns={"id": "book_id"}, inplace=True)
        df_trans.rename(columns={"id": "transaction_item_id"}, inplace=True)

        logging.info("Dataset berhasil dimuat ke DataFrame.")
        # Tampilkan struktur data untuk debugging
        logging.info(f"Books DataFrame columns: {df_books.columns.tolist()}")
        logging.info(
            f"Transactions DataFrame columns: {df_trans.columns.tolist()}")

        # Gabungkan data untuk validasi awal
        df = df_books.merge(df_trans, how="left",
                            left_on="book_id", right_on="book_id", validate="one_to_many")
        # Pilih kolom target sesuai reference
        if reference == "transaction":
            target_col = "quantity"
        else:
            target_col = "rating"
        if target_col not in df.columns:
            raise ValueError(
                f"Kolom {target_col} tidak ditemukan pada data transaksi.")
        data_reco = df[['user_id', 'book_id', target_col]].dropna()

        logging.info(
            f"Data gabungan untuk training disiapkan dengan tipe: {reference}.")

        if data_reco.empty:
            logging.error("Dataset kosong setelah digabungkan.")
            raise ValueError(
                "Dataset kosong setelah digabungkan. Pastikan ID buku di kedua file cocok.")

        # 3. Setup Dataset Surprise
        # Untuk transaksi, rating_scale bisa disesuaikan jika perlu
        min_value = None
        max_value = None

        if reference == "transaction":
            min_value = int(data_reco[target_col].min())
            max_value = int(data_reco[target_col].max())
            reader = Reader(rating_scale=(min_value, max_value))
        else:
            min_value = 1
            max_value = 5
            reader = Reader(rating_scale=(min_value, max_value))

        data = Dataset.load_from_df(
            data_reco[['user_id', 'book_id', target_col]], reader)

        logging.info("Memulai training model SVD...")

        # 4. Evaluasi Model (Hold-out Validation)
        trainset_eval, testset_eval = train_test_split(
            data, test_size=0.2, random_state=42)
        eval_algo = SVD(n_factors=n_factors, n_epochs=n_epochs,
                        lr_all=lr_all, reg_all=reg_all, random_state=42)
        eval_algo.fit(trainset_eval)
        predictions = eval_algo.test(testset_eval)

        rmse_score = accuracy.rmse(predictions, verbose=False)
        mae_score = accuracy.mae(predictions, verbose=False)

        logging.info(f"Evaluasi Model - RMSE: {rmse_score}, MAE: {mae_score}")

        # 5. Training Final pada Seluruh Data
        full_trainset = data.build_full_trainset()
        global model
        model = SVD(n_factors=n_factors, n_epochs=n_epochs,
                    lr_all=lr_all, reg_all=reg_all, random_state=42)
        model.fit(full_trainset)

        # Simpan model ke file dengan suffix tanggal
        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_filename = f"svd_model_{now}.pkl"
        model_path = os.path.join(MODEL_DIR, model_filename)
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)

        # Simpan metadata ke database (modul terpisah)
        algorithm = "SVD"
        model_metadata = save_model_history(model_filename, algorithm, n_factors, n_epochs,
                                            lr_all, reg_all, float(rmse_score), float(mae_score), min_value, max_value, reference, created_by)

        # Perbarui model aktif
        set_active_model(model_metadata["id"])

        return {
            "status": "Success",
            "model": model_metadata
        }

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Gagal melatih model: {str(e)}")


def tuning_model(data_reco, param_grid: dict[str, list] = None, cv=3, n_jobs=-1, reference: str = "rating"):
    """
    Melakukan grid search hyperparameter SVD menggunakan Surprise GridSearchCV.
    data_reco: DataFrame dengan kolom ['user_id', 'book_id', 'quantity', 'rating']
    param_grid: dict parameter grid
    cv: jumlah cross-validation folds
    n_jobs: paralel jobs
    reference: referensi rekomendasi ('rating' atau 'transaksi')
    """
    if param_grid is None:
        param_grid = {
            'n_epochs': [5, 10, 20],
            'lr_all': [0.002, 0.005, 0.007],
        }

    logging.info(
        f"Parameter grid untuk tuning: {param_grid}, CV: {cv}, n_jobs: {n_jobs}, reference: {reference}")

    min_value = None
    max_value = None

    if reference == "transaction":
        target_col = "quantity"
        min_value = int(data_reco[target_col].min())
        max_value = int(data_reco[target_col].max())
        reader = Reader(rating_scale=(min_value, max_value))
    else:
        target_col = "rating"
        min_value = 1
        max_value = 5
        reader = Reader(rating_scale=(min_value, max_value))

    data = Dataset.load_from_df(
        data_reco[['user_id', 'book_id', target_col]], reader)

    gs = GridSearchCV(SVD, param_grid, measures=[
                      'rmse', 'mae'], cv=cv, n_jobs=n_jobs, joblib_verbose=1)
    gs.fit(data)

    return {
        'best_params': gs.best_params['rmse'],
        'best_score_rmse': gs.best_score['rmse'],
        'best_score_mae': gs.best_score['mae'],
        'min_value': min_value,
        'max_value': max_value,
        'config': {
            'param_grid': param_grid,
            'cv': cv,
            'n_jobs': n_jobs,
            'reference': reference
        },
        'cv_results': {k: v.tolist() if hasattr(v, "tolist") else v for k, v in gs.cv_results.items()},
    }


@app.post("/tune", tags=["Model Tuning"])
async def tune_model(
    books_file: UploadFile = File(
        ..., description="File Excel berisi daftar buku (kolom: id, title, author)"),
    transactions_file: UploadFile = File(
        ..., description="File Excel transaksi (kolom: user_id, book_id, 'quantity', rating)"),
    param_grid: Optional[str] = Form(
        None, description="Parameter grid dalam format JSON"),
    cv: int = Form(3, description="Jumlah cross-validation folds"),
    n_jobs: int = Form(-1, description="Jumlah pekerjaan paralel"),
    reference: str = Form(
        "rating", description="Referensi rekomendasi: 'rating' atau 'transaksi'")
):
    """
    Endpoint untuk tuning hyperparameter SVD dengan grid search.
    """
    try:
        if param_grid:
            param_grid = json.loads(param_grid)

        # Simpan file sementara
        books_path = os.path.join(DATASET_DIR, "books.xlsx")
        trans_path = os.path.join(DATASET_DIR, "transaction_items.xlsx")
        with open(books_path, "wb") as buffer:
            shutil.copyfileobj(books_file.file, buffer)
        with open(trans_path, "wb") as buffer:
            shutil.copyfileobj(transactions_file.file, buffer)
        df_books = pd.read_excel(books_path)
        df_trans = pd.read_excel(trans_path)
        df_books.rename(columns={"id": "book_id"}, inplace=True)
        df_trans.rename(columns={"id": "transaction_item_id"}, inplace=True)
        df = df_books.merge(df_trans, how="left", left_on="book_id",
                            right_on="book_id", validate="one_to_many")
        # Pilih kolom target sesuai reference
        if reference == "transaction":
            target_col = "quantity"
        else:
            target_col = "rating"
        if target_col not in df.columns:
            raise ValueError(
                f"Kolom {target_col} tidak ditemukan pada data transaksi.")
        data_reco = df[['user_id', 'book_id', target_col]].dropna()
        if data_reco.empty:
            raise ValueError(
                "Dataset kosong setelah digabungkan. Pastikan ID buku di kedua file cocok.")
        result = tuning_model(data_reco, param_grid, cv, n_jobs, reference)
        return {"status": "Success", "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Tuning gagal: {str(e)}")


@app.get("/recommend/{user_id}", tags=["Recommendation"])
def get_recommendations(user_id: int, limit: int = 10):
    """
    Memberikan rekomendasi buku. Menggunakan SVD untuk user lama 
    dan Popularitas (Cold Start) untuk user baru.
    """
    if model is None:
        raise HTTPException(
            status_code=404, detail="Model belum tersedia. Sila gunakan endpoint /train terlebih dahulu.")

    try:
        # Load dataset dari penyimpanan lokal
        df_books = pd.read_excel(os.path.join(DATASET_DIR, "books.xlsx"))
        df_trans = pd.read_excel(os.path.join(
            DATASET_DIR, "transaction_items.xlsx"))

        user_history = df_trans[df_trans['user_id'] == user_id]

        # LOGIKA COLD START (User Baru)
        if user_history.empty:
            popular_books = df_trans.groupby(
                'book_id')['rating'].sum().reset_index()
            popular_ids = popular_books.sort_values(
                by='rating', ascending=False).head(limit)['book_id'].tolist()

            reco_list = []
            for b_id in popular_ids:
                b_info = df_books[df_books['id'] == b_id].iloc[0]
                reco_list.append({
                    "id": int(b_id), "title": b_info['title'], "author": b_info['author'],
                    "reason": "Populer (Cold Start)"
                })
            return {"user_id": user_id, "strategy": "Cold Start", "results": reco_list}

        # LOGIKA SVD (User Lama)
        all_ids = df_books['id'].unique()
        interacted_ids = user_history['book_id'].unique()
        to_predict = [i for i in all_ids if i not in interacted_ids]

        preds = []
        for b_id in to_predict:
            preds.append((b_id, model.predict(user_id, b_id).est))

        preds.sort(key=lambda x: x[1], reverse=True)
        top_n = preds[:limit]

        reco_list = []
        for b_id, score in top_n:
            b_info = df_books[df_books['id'] == b_id].iloc[0]
            reco_list.append({
                "id": int(b_id), "title": b_info['title'], "author": b_info['author'],
                "score": round(score, 2), "reason": "Berdasarkan minat Anda"
            })

        return {"user_id": user_id, "strategy": "SVD Matrix Factorization", "results": reco_list}

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error saat memproses: {str(e)}")


@app.get("/model-history", tags=["Model History"])
def model_history(limit: int = 20):
    """
    Mengembalikan riwayat training model (maksimal 20 terakhir secara default).
    """
    models = get_model_history(limit)
    return {"models": models}


@app.post("/set-active-model/{model_history_id}", tags=["Model History"])
def set_active(model_history_id: int):
    """
    Menetapkan model tertentu sebagai model aktif berdasarkan ID riwayat model.
    """
    try:
        set_active_model(model_history_id)

        # Reload model yang baru ditetapkan sebagai aktif
        global model
        model_data = get_model_by_id(model_history_id)
        if model_data is None:
            raise HTTPException(
                status_code=404, detail="Model dengan ID tersebut tidak ditemukan.")

        model_path = os.path.join(MODEL_DIR, model_data["filename"])
        if not os.path.exists(model_path):
            raise HTTPException(
                status_code=404, detail="File model tidak ditemukan di server.")

        # Muat ulang model dari file yang baru ditetapkan sebagai aktif
        model = load_model(model_path)

        return {"status": "Success", "message": f"Model dengan ID {model_history_id} telah ditetapkan sebagai aktif."}
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Gagal menetapkan model aktif: {str(e)}")


@app.get("/active-model", tags=["Model History"])
def active_model():
    """
    Mengembalikan informasi model yang sedang aktif digunakan.
    """
    active = get_active_model()
    if active is None:
        raise HTTPException(
            status_code=404, detail="Tidak ada model aktif saat ini.")
    return {"active_model": active}

# Delete model


@app.delete("/model-history/{model_history_id}", tags=["Model History"])
def delete_model(model_history_id: int):
    """
    Menghapus model dari riwayat berdasarkan ID riwayat model.
    """
    try:
        model_data = get_model_by_id(model_history_id)
        if model_data is None:
            raise HTTPException(
                status_code=404, detail="Model dengan ID tersebut tidak ditemukan.")

        model_path = os.path.join(MODEL_DIR, model_data["filename"])
        if os.path.exists(model_path):
            os.remove(model_path)

        # Hapus dari database
        delete_model_history(model_history_id)

        return {"status": "Success", "message": f"Model dengan ID {model_history_id} telah dihapus."}
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Gagal menghapus model: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    # Reconfigure logging to ensure output appears in terminal
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )
    logging.info("Starting FastAPI app with Uvicorn...")
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
