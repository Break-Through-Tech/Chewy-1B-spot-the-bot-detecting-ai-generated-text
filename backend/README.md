# Spot the Bot inference backend

The FastAPI service loads the checked-in TF-IDF vectorizer and Keras classifier from `backend/models/`. It performs inference only; it does not train or fit the model.

## Run locally

From the repository root:

```bash
cd backend
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

The model files are already in `backend/models/`; no manual copy is needed. Check `GET /health`, open `/docs`, and try `POST /predict` with:

```json
{"text":"A sample paragraph for classification."}
```

The Next.js frontend sends requests through its own `/api/predict` route. Set `FASTAPI_URL=http://127.0.0.1:8000` in `frontend/.env.local` when running the frontend locally.

## Deploy the backend

- Set the service root to `backend` and use the start command `uvicorn app.main:app --host 0.0.0.0 --port $PORT`.
- The model files must be present in the deployment image. They are tracked in this repository.
- Set `FRONTEND_ORIGIN` to the deployed frontend origin if the browser will call FastAPI directly. The included Next.js route proxies requests server-side.
- Start with one worker; each worker loads its own TensorFlow model copy.
- Add rate limits or access controls before exposing the prediction service publicly.

The API's `ai_score` is a raw sigmoid score and is not a calibrated probability. Predictions are experimental and should not be treated as proof of authorship.
