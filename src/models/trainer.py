import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

class Trainer:
    def __init__(self, model=None):
        self.model = model if model else RandomForestClassifier()

    def train(self, X, y, test_size=0.2, random_state=42):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        self.model.fit(X_train, y_train)
        return self.model.score(X_test, y_test)

    def predict(self, X):
        return self.model.predict(X)