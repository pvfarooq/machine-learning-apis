import pandas as pd
from sklearn.impute import SimpleImputer

from schemas.breast_cancer import BreastCancerRequest, BreastCancerResponse

from .artifact_loader import ArtifactLoader


class BreastCancerPredictionService(ArtifactLoader):
    def __init__(self):
        self.path_model = (
            "ml_models/logistic_regression/artifacts/breast_cancer_prediction.pkl"
        )
        self.imputer = SimpleImputer(strategy="mean")  # Replace NaN with column mean
        super().__init__(self.path_model)

    def preprocess_input(self, request: BreastCancerRequest) -> pd.DataFrame:
        model_dict = request.model_dump()

        model_dict["concave points_mean"] = model_dict.pop("concave_points_mean")
        model_dict["concave points_worst"] = model_dict.pop("concave_points_worst")
        model_dict["concave points_se"] = model_dict.pop("concave_points_se")

        df = pd.DataFrame.from_dict([model_dict])

        df_imputed = pd.DataFrame(self.imputer.fit_transform(df), columns=df.columns)
        return df_imputed

    def predict(self, request: BreastCancerRequest) -> BreastCancerResponse:
        df = self.preprocess_input(request)
        malignent = self.model.predict(df)[0]
        return BreastCancerResponse(
            diagnosis="Malignant" if malignent == 1 else "Benign"
        )
