"""Load the trained TF-IDF + Keras classifier and score incoming text.

No training or vectorizer fitting happens in the API.
Only load trusted joblib model artifacts; joblib uses Python pickle internally.
"""
from pathlib import Path

import joblib
import numpy as np
from tensorflow import keras

MODEL_DIR = Path(__file__).resolve().parents[1] / "models"
VECTORIZER_PATH = MODEL_DIR / "tfidf_vectorizer.joblib"
MODEL_PATH = MODEL_DIR / "feedforward_neural_network.keras"
THRESHOLD = 0.5


class SpotTheBotPredictor:
    def __init__(self, vectorizer_path: Path = VECTORIZER_PATH,
                 model_path: Path = MODEL_PATH):
        if not vectorizer_path.is_file() or not model_path.is_file():
            raise FileNotFoundError(
                "Missing model artifacts. Place tfidf_vectorizer.joblib and "
                "feedforward_neural_network.keras in backend/models/."
            )
        self.vectorizer = joblib.load(vectorizer_path)
        self.model = keras.models.load_model(model_path, compile=False)
        expected_features = self.model.input_shape[-1]
        actual_features = len(self.vectorizer.vocabulary_)
        if expected_features != actual_features:
            raise ValueError(
                f"Model expects {expected_features} features but vectorizer has "
                f"{actual_features}. Check that artifacts were trained together."
            )

    def predict(self, text: str) -> dict:
        if not text.strip():
            raise ValueError("Text must not be empty")
        # Matches notebook: vectorizer.transform(...); sparse row -> dense float32.
        features = self.vectorizer.transform([text]).toarray().astype(np.float32)
        # Output is P(class 1) under the notebook's Human=0 / ChatGPT=1 mapping.
        ai_score = float(self.model.predict(features, verbose=0).reshape(-1)[0])
        predicted_class = int(ai_score >= THRESHOLD)
        return {
            "prediction": "AI" if predicted_class == 1 else "Human",
            "ai_score": ai_score,
            "threshold": THRESHOLD,
            "model_version": "tfidf-ffnn-32-16",
        }
