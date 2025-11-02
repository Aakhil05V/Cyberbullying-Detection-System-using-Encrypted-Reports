import os
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer

class CyberbullyingDetector:
    def __init__(self):
        self.model_path = 'model/cyberbullying_model.pkl'
        self.vectorizer_path = 'model/vectorizer.pkl'

        # Create model folder if missing
        os.makedirs('model', exist_ok=True)

        if os.path.exists(self.model_path) and os.path.exists(self.vectorizer_path):
            self.model = joblib.load(self.model_path)
            self.vectorizer = joblib.load(self.vectorizer_path)
        else:
            print("⚠️ Model not found — training a simple baseline model...")
            self._train_dummy_model()

    def _train_dummy_model(self):
        """Train a simple placeholder model"""
        texts = [
            "I hate you", "You're ugly", "You're dumb", 
            "Have a nice day", "You're awesome", "I like you"
        ]
        labels = [1, 1, 1, 0, 0, 0]  # 1 = Cyberbullying, 0 = Normal

        self.vectorizer = CountVectorizer()
        X = self.vectorizer.fit_transform(texts)
        y = np.array(labels)

        self.model = RandomForestClassifier(n_estimators=10, random_state=42)
        self.model.fit(X, y)

        joblib.dump(self.model, self.model_path)
        joblib.dump(self.vectorizer, self.vectorizer_path)
        print("✅ Dummy model trained and saved.")

    def detect_cyberbullying(self, text):
        """Predict cyberbullying status"""
        X = self.vectorizer.transform([text])
        prediction = self.model.predict(X)[0]
        prob = self.model.predict_proba(X)[0][prediction]

        label = "Cyberbullying" if prediction == 1 else "Not Cyberbullying"
        severity = "high" if prediction == 1 else "low"

        return {
            "label": label,
            "severity": severity,
            "confidence": float(prob)
        }

    def batch_detect(self, texts):
        """Batch process"""
        X = self.vectorizer.transform(texts)
        preds = self.model.predict(X)
        probs = self.model.predict_proba(X)

        results = []
        for i, text in enumerate(texts):
            pred = preds[i]
            prob = probs[i][pred]
            results.append({
                "text": text,
                "label": "Cyberbullying" if pred == 1 else "Not Cyberbullying",
                "confidence": float(prob),
                "severity": "high" if pred == 1 else "low"
            })
        return results
