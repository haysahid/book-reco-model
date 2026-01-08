# Book Recommendation Model API

A small FastAPI backend for training and serving an SVD-based book recommendation model (using the `surprise` library). The project supports both rating-based and transaction-based training, stores model metadata (including min/max value ranges), and keeps a model history in a lightweight SQLite database.

## Features

- Train SVD models using user-item ratings or transaction quantities.
- Grid search tuning for SVD hyperparameters.
- Store model files under `models/` and datasets under `datasets/`.
- Model metadata persisted in `model_history.db` (SQLite): filename, algorithm, hyperparameters, RMSE/MAE, reference type (rating/transaction), created_by (manual/auto), and min/max value range.
- Endpoints for recommendations with Cold Start fallback (popularity).

## Requirements

- Python 3.8+
- A virtual environment is recommended.

Recommended Python packages (install with pip):

```
fastapi
uvicorn
pandas
scikit-surprise
openpyxl
```

You can install with:

```bash
python -m venv venv
source venv/bin/activate
pip install fastapi uvicorn pandas scikit-surprise openpyxl
```

Note: `scikit-surprise` may require a C compiler on some platforms. See Surprise installation docs if you encounter build issues.

## Run the API

Start the app with Uvicorn:

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Open the interactive API docs at: `http://localhost:8000/docs`

## File layout

- `main.py` - FastAPI application, training/tuning/recommendation endpoints.
- `model_db.py` - Simple SQLite helpers to initialize DB and save/get model history.
- `datasets/` - Uploaded dataset files (books.xlsx, transaction_items.xlsx).
- `models/` - Saved model pickle files.
- `model_history.db` - SQLite database file created by the app.

## Important data/model fields

When a model is trained and saved, the application stores metadata in `model_histories` table. Notable fields:

- `filename` — model file name under `models/`.
- `algorithm` — currently `SVD`.
- `n_factors`, `n_epochs`, `lr_all`, `reg_all` — hyperparameters.
- `rmse`, `mae` — evaluation metrics from holdout evaluation.
- `min_value`, `max_value` — the min/max of the target column used during training (helpful for UI display and interpretation).
- `reference` — which column was used as the target: `rating` or `transaction` (transaction uses `quantity`).
- `created_by` — indicates whether the model is `manual` or `auto` (useful for auto-training jobs).

## API Endpoints

All endpoints are described in the interactive docs at `/docs`, but here are the most important ones with quick usage examples.

### POST /train
Train a new model and save metadata.

Form-data fields:
- `books_file` (file, required): Excel file with book list (expected columns: `id`, `title`, `author`, etc.).
- `transactions_file` (file, required): Excel transactions (expected columns: `user_id`, `book_id`, `quantity`, `rating`, etc.).
- `n_factors` (int, default 100)
- `n_epochs` (int, default 20)
- `lr_all` (float, default 0.005)
- `reg_all` (float, default 0.02)
- `reference` (str, default `rating`): "rating" or "transaction" (if `transaction`, `quantity` is used as the target)
- `created_by` (str, default `manual`): "manual" or "auto"

Example curl (multipart/form-data):

```bash
curl -X POST "http://localhost:8000/train" \
  -F "books_file=@./datasets/books.xlsx" \
  -F "transactions_file=@./datasets/transaction_items.xlsx" \
  -F "n_factors=50" \
  -F "n_epochs=20" \
  -F "reference=rating" \
  -F "created_by=manual"
```

Notes:
- The endpoint will evaluate on a hold-out 20% split and compute RMSE and MAE before training on the full dataset.
- The `min_value` and `max_value` stored correspond to the target column range (either rating range or transaction quantity range).

### POST /tune
Tuning SVD hyperparameters with GridSearchCV.

Form-data fields:
- `books_file`, `transactions_file` (files)
- `param_grid` (optional JSON string)
- `cv` (int, default 3)
- `n_jobs` (int, default -1)
- `reference` (str, default `rating`)

Example:

```bash
curl -X POST "http://localhost:8000/tune" \
  -F "books_file=@./datasets/books.xlsx" \
  -F "transactions_file=@./datasets/transaction_items.xlsx" \
  -F "param_grid={\"n_epochs\": [5,10], \"lr_all\": [0.002,0.005]}" \
  -F "reference=transaction"
```

### GET /recommend/{user_id}
Get recommendations for a user. Uses currently active SVD model; for cold-start users the endpoint falls back to popularity by summing `rating`.

Query params:
- `limit` (int, default 10)

Example:
```
GET /recommend/123?limit=5
```

### GET /model-history
Return recent model training history (including `min_value`/`max_value`).

Example:
```
GET /model-history
```

### POST /set-active-model/{model_history_id}
Mark a saved model as active (the app will load it for predictions).

### GET /active-model
Return current active model metadata.

### DELETE /model-history/{model_history_id}
Remove a model record and delete its file.

## Notes & tips

- The app expects uploaded Excel files to have consistent ID fields between `books.xlsx` and `transaction_items.xlsx` (book IDs must match).
- The project saves datasets into `datasets/` with fixed filenames (`books.xlsx` and `transaction_items.xlsx`). Overwriting is expected when re-training.
- When using `reference=transaction`, the Surprise `Reader` rating_scale is set to the observed min/max of the `quantity` column so metrics and model behavior are consistent.
- If you change database schema and have an existing `model_history.db`, a migration or recreation of the DB may be needed. The code attempts to `ALTER TABLE` to add columns if missing, but backing up DB is recommended.
