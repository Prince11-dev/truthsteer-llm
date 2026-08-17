import os
import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression
from src.config import config
from src.vector_extractor import ContrastiveVectorExtractor

class TruthfulnessProbeTrainer:
    def __init__(self, target_layer: int = config.target_layer):
        self.target_layer = target_layer
        self.probe = LogisticRegression(max_iter=1000, C=1.0)

    def train_and_save(self, positive_texts: list, negative_texts: list, save_path: str = config.probe_save_path):
        extractor = ContrastiveVectorExtractor()
        X, y = [], []

        print("Extracting training representations for Probe...")
        for text in positive_texts:
            act = extractor._get_last_token_activation(text, self.target_layer).numpy()
            X.append(act)
            y.append(1)

        for text in negative_texts:
            act = extractor._get_last_token_activation(text, self.target_layer).numpy()
            X.append(act)
            y.append(0)

        X = np.array(X)
        y = np.array(y)

        self.probe.fit(X, y)
        acc = self.probe.score(X, y)
        print(f"Probe Training Accuracy at Layer {self.target_layer}: {acc * 100:.2f}%")

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "wb") as f:
            pickle.dump(self.probe, f)
        print(f"Probe model saved to {save_path}")

if __name__ == "__main__":
    positives = [
        "The Earth revolves around the Sun.",
        "DNA carries genetic information.",
        "Gravity pulls objects toward Earth's center."
    ]
    negatives = [
        "The Sun revolves around the Earth.",
        "DNA is made of pure carbohydrate sugar.",
        "Gravity pushes objects into outer space."
    ]
    trainer = TruthfulnessProbeTrainer()
    trainer.train_and_save(positives, negatives)
