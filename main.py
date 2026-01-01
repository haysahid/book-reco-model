import os
import logging
import shutil
import pickle
import pandas as pd
from typing import Optional
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from surprise import Reader, SVD, Dataset, accuracy
from surprise.model_selection import train_test_split

app = FastAPI(
    title="Sistem Rekomendasi Buku API",
    description="Backend untuk training SVD dan prediksi rekomendasi buku dengan fitur Cold Start."
)

MODEL_DIR = "models"
DATASET_DIR = "datasets"

# Buat folder dataset jika belum ada
os.makedirs(DATASET_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


def load_trained_model():
    model_path = os.path.join(MODEL_DIR, "svd_model.pkl")
    if os.path.exists(model_path):
        with open(model_path, 'rb') as f:
            return pickle.load(f)
    return None


# Global variable untuk menyimpan model yang sedang aktif
model = load_trained_model()


@app.post("/train", tags=["Model Training"])
async def train_model(
    books_file: UploadFile = File(
        ..., description="File Excel berisi daftar buku (kolom: id, title, author)"),
    transactions_file: UploadFile = File(
        ..., description="File Excel transaksi (kolom: user_id, book_id, quantity)"),
    n_factors: int = Form(100, description="Jumlah faktor laten"),
    n_epochs: int = Form(20, description="Jumlah iterasi training"),
    lr_all: float = Form(0.005, description="Learning rate"),
    reg_all: float = Form(0.02, description="Regularization term")
):
    logging.info("Menerima permintaan training model baru.")

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
        data_reco = df[['user_id', 'book_id', 'quantity']].dropna()

        logging.info("Data gabungan untuk training disiapkan.")

        if data_reco.empty:
            logging.error("Dataset kosong setelah digabungkan.")
            raise ValueError(
                "Dataset kosong setelah digabungkan. Pastikan ID buku di kedua file cocok.")

        # 3. Setup Dataset Surprise
        reader = Reader(rating_scale=(1, 5))
        data = Dataset.load_from_df(
            data_reco[['user_id', 'book_id', 'quantity']], reader)

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

        # Simpan ke file pickle
        model_path = os.path.join(MODEL_DIR, "svd_model.pkl")
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)

        return {
            "status": "Success",
            "evaluation": {"rmse": round(rmse_score, 4), "mae": round(mae_score, 4)},
            "config": {"n_factors": n_factors, "n_epochs": n_epochs}
        }

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Gagal melatih model: {str(e)}")


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
                'book_id')['quantity'].sum().reset_index()
            popular_ids = popular_books.sort_values(
                by='quantity', ascending=False).head(limit)['book_id'].tolist()

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
