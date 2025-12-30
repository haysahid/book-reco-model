import os
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

MODEL_PATH = "svd_model.pkl"
UPLOAD_DIR = "datasets"

# Buat folder dataset jika belum ada
os.makedirs(UPLOAD_DIR, exist_ok=True)


def load_trained_model():
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, 'rb') as f:
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
    """
    Endpoint untuk mengunggah dataset baru, melakukan training, dan evaluasi model.
    """
    try:
        # 1. Simpan File ke Lokal
        books_path = os.path.join(UPLOAD_DIR, "books.xlsx")
        trans_path = os.path.join(UPLOAD_DIR, "transaction_items.xlsx")

        with open(books_path, "wb") as buffer:
            shutil.copyfileobj(books_file.file, buffer)
        with open(trans_path, "wb") as buffer:
            shutil.copyfileobj(transactions_file.file, buffer)

        # 2. Persiapan Data
        df_books = pd.read_excel(books_path)
        df_trans = pd.read_excel(trans_path)

        # Gabungkan data untuk validasi awal
        df = df_books.merge(df_trans, how="left",
                            left_on="id", right_on="book_id")
        data_reco = df[['user_id', 'book_id', 'quantity']].dropna()

        if data_reco.empty:
            raise ValueError(
                "Dataset kosong setelah digabungkan. Pastikan ID buku di kedua file cocok.")

        # 3. Setup Dataset Surprise
        reader = Reader(rating_scale=(1, 5))
        data = Dataset.load_from_df(
            data_reco[['user_id', 'book_id', 'quantity']], reader)

        # 4. Evaluasi Model (Hold-out Validation)
        trainset_eval, testset_eval = train_test_split(
            data, test_size=0.2, random_state=42)
        eval_algo = SVD(n_factors=n_factors, n_epochs=n_epochs,
                        lr_all=lr_all, reg_all=reg_all, random_state=42)
        eval_algo.fit(trainset_eval)
        predictions = eval_algo.test(testset_eval)

        rmse_score = accuracy.rmse(predictions, verbose=False)
        mae_score = accuracy.mae(predictions, verbose=False)

        # 5. Training Final pada Seluruh Data
        full_trainset = data.build_full_trainset()
        global model
        model = SVD(n_factors=n_factors, n_epochs=n_epochs,
                    lr_all=lr_all, reg_all=reg_all, random_state=42)
        model.fit(full_trainset)

        # Simpan ke file pickle
        with open(MODEL_PATH, 'wb') as f:
            pickle.dump(model, f)

        return {
            "status": "Success",
            "evaluation": {"rmse": round(rmse_score, 4), "mae": round(mae_score, 4)},
            "config": {"n_factors": n_factors, "n_epochs": n_epochs}
        }

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Gagal melatih model: {str(e)}")


@app.get("/recommendations/{user_id}", tags=["Recommendation"])
def get_recommendations(user_id: int, n: int = 10):
    """
    Memberikan rekomendasi buku. Menggunakan SVD untuk user lama 
    dan Popularitas (Cold Start) untuk user baru.
    """
    if model is None:
        raise HTTPException(
            status_code=404, detail="Model belum tersedia. Sila gunakan endpoint /train terlebih dahulu.")

    try:
        # Load dataset dari penyimpanan lokal
        df_books = pd.read_excel(os.path.join(UPLOAD_DIR, "books.xlsx"))
        df_trans = pd.read_excel(os.path.join(
            UPLOAD_DIR, "transaction_items.xlsx"))

        user_history = df_trans[df_trans['user_id'] == user_id]

        # LOGIKA COLD START (User Baru)
        if user_history.empty:
            popular_books = df_trans.groupby(
                'book_id')['quantity'].sum().reset_index()
            popular_ids = popular_books.sort_values(
                by='quantity', ascending=False).head(n)['book_id'].tolist()

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
        top_n = preds[:n]

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
    uvicorn.run(app, host="0.0.0.0", port=8000)
