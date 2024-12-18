import pickle
from abc import ABC, abstractmethod

import pandas as pd


class ArtifactLoader(ABC):
    def __init__(self, path_model: str, path_encoder: str = None):
        self.path_model = path_model
        self.path_encoder = path_encoder
        self.model = self.load_artifact(self.path_model)

        if self.path_encoder:
            self.le = self.load_artifact(self.path_encoder)

    def load_artifact(self, path_to_artifact):
        """Load from a pickle file."""
        with open(path_to_artifact, "rb") as f:
            artifact = pickle.load(f)
        return artifact

    @abstractmethod
    def preprocess_input(self, request: dict) -> pd.DataFrame:
        pass

    @abstractmethod
    def predict_price(self, request: dict) -> dict:
        pass
