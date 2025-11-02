"""
Cyberbullying Detection System with Encrypted Reports
Main detection module for identifying harmful content
"""

import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import pickle
import os
from datetime import datetime


class CyberbullyingDetector:
    """Main class for detecting cyberbullying content"""

    def __init__(self, model_path=None):
        self.model = None
        self.vectorizer = None
        self.model_path = model_path or 'models/detector_model.pkl'
        self.toxic_keywords = [
            'hate', 'stupid', 'idiot', 'kill', 'die', 'threat',
            'abuse', 'insult', 'discriminate', 'racist', 'sexist'
        ]
        self.load_model()

    def preprocess_text(self, text):
        """Clean and preprocess text data"""
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def extract_features(self, text):
        """Extract features from text"""
        text = self.preprocess_text(text)
        if self.vectorizer is None:
            self.vectorizer = TfidfVectorizer(max_features=1000)
        return self.vectorizer.fit_transform([text])

    def detect_cyberbullying(self, text, threshold=0.5):
        """Detect if text contains cyberbullying content"""
        if not text or len(text.strip()) == 0:
            return {'is_bullying': False, 'confidence': 0.0, 'severity': 'none'}

        # Keyword-based detection
        text_lower = text.lower()
        keyword_score = sum(1 for keyword in self.toxic_keywords if keyword in text_lower) / len(self.toxic_keywords)

        # Pattern-based detection (ALL CAPS, repeated characters, etc.)
        caps_ratio = sum(1 for c in text if c.isupper()) / max(len(text), 1)
        pattern_score = 0.3 if caps_ratio > 0.5 else 0

        # ML-based detection
        features = self.extract_features(text)
        ml_score = self.model.predict_proba(features)[0][1] if self.model else 0.0

        # Combined score
        combined_score = (keyword_score * 0.3) + (pattern_score * 0.2) + (ml_score * 0.5)

        return {
            'is_bullying': combined_score > threshold,
            'confidence': float(combined_score),
            'severity': self.get_severity(combined_score),
            'timestamp': datetime.now().isoformat()
        }

    def get_severity(self, score):
        """Classify severity based on score"""
        if score < 0.3:
            return 'low'
        elif score < 0.6:
            return 'medium'
        else:
            return 'high'

    def load_model(self):
        """Load pre-trained model if available"""
        try:
            if os.path.exists(self.model_path):
                with open(self.model_path, 'rb') as f:
                    self.model = pickle.load(f)
            else:
                # Create a default model placeholder
                self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        except Exception as e:
            print(f'Error loading model: {e}')
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)

    def save_model(self):
        """Save trained model"""
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        with open(self.model_path, 'wb') as f:
            pickle.dump(self.model, f)

    def batch_detect(self, texts):
        """Detect cyberbullying in multiple texts"""
        results = []
        for text in texts:
            results.append(self.detect_cyberbullying(text))
        return results


if __name__ == '__main__':
    detector = CyberbullyingDetector()
    test_text = "This is a test message"
    result = detector.detect_cyberbullying(test_text)
    print(f'Detection Result: {result}')
