import pandas as pd

from schemas.salary import SalaryRequest, SalaryResponse

from .artifact_loader import ArtifactLoader


class SalaryService(ArtifactLoader):
    def __init__(self):
        self.path_model = (
            "ml_models/linear_regression/artifacts/salary_prediction_model.pkl"
        )
        super().__init__(self.path_model)

    def preprocess_input(self, request: SalaryRequest) -> pd.DataFrame:
        data_dict = {
            "YearsExperience": request.yoe,
        }
        data_df = pd.DataFrame.from_dict([data_dict])
        return data_df

    def predict_price(self, request: SalaryRequest) -> SalaryResponse:
        input_df = self.preprocess_input(request)
        salary = round(self.model.predict(input_df)[0], 2)
        response = SalaryResponse
        response.salary = salary
        return response
