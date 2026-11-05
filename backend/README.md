# Spot the Bot: starter inference backend

This starter is based on the supplied TF-IDF + Keras notebooks and trained artifacts. **It does not retrain a model.**

## Files
- `backend/app/inference.py`: loads fitted TF-IDF vectorizer and trained Keras network; scores raw text.
- `backend/app/main.py`: FastAPI `/predict` and `/health` endpoints.
- `backend/models/`: place your own two exported artifacts here (not included in starter ZIP).

## Prepare
1. Copy your `tfidf_vectorizer.joblib` and `feedforward_neural_network.keras` into `backend/models/`.
2. Open the existing PyCharm project and add `backend/` to its repository root. Keep your existing notebooks and docs unchanged.
3. Create a **Python 3.11** virtual environment for the backend (separate from notebook environment). Install packages: `pip install -r backend/requirements.txt` from repository root, or `pip install -r requirements.txt` within `backend/`.
4. From `backend/`, run: `uvicorn app.main:app --reload --port 8000`.
5. Open `http://127.0.0.1:8000/docs`; send `POST /predict` with `{"text":"A sample paragraph for classification."}`.

## Railway later
- Repo service root: `backend`.
- Start command: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`.
- Environment variable: `FRONTEND_ORIGIN=https://your-app.vercel.app`.
- Turn off Serverless sleeping; monitor memory and cost. A single worker is appropriate initially.
- Ensure your trusted model artifacts are available during build/deployment (for example: securely provisioned artifacts or Git LFS with proper deployment support).

## Before public release
- Verify that notebook and deployed scores match on a small set of identical texts.
- Record exact Python, scikit-learn, Keras/TensorFlow, NumPy versions used to save artifacts, then pin tested versions.
- Match label mapping (the notebook evaluation uses Human=0, ChatGPT=1) and threshold 0.5.
- The output is a raw sigmoid *model score*, not a calibrated probability or definitive detection claim.
- Add rate limits or access controls if exposing the expensive public endpoint.
- Fix notebook metadata: actual trained layers are 32 → 16 → 1, despite the metadata cell saying 64 → 32 → 1.
