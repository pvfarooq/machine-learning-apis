import pandas as pd

from schemas.apartment import ApartmentRequest, ApartmentResponse

from .artifact_loader import ArtifactLoader


class ApartmentService(ArtifactLoader):
    def __init__(self):
        self.path_model = "ml_models/random_forest_regression/artifacts/randomForestForApartmentPrice.pkl"
        self.path_encoder = (
            "ml_models/random_forest_regression/artifacts/neighbourhood_encoder.pkl"
        )
        super().__init__(self.path_model, self.path_encoder)

    def preprocess_input(self, request: ApartmentRequest) -> pd.DataFrame:
        data_dict = {
            "rooms": request.rooms,
            "size": request.size,
            "bathrooms": request.bathrooms,
            "neighbourhood": request.neighbourhood,
            "year_built": request.year_built,
        }
        data_df = pd.DataFrame.from_dict([data_dict])

        data_df.neighbourhood = data_df.neighbourhood.str.lower()
        data_df.neighbourhood = self.le.transform(data_df.neighbourhood)
        data_df.neighbourhood = data_df.neighbourhood.astype("category")
        return data_df

    def predict(self, request: ApartmentRequest) -> ApartmentResponse:
        input_df = self.preprocess_input(request)
        apartment_price = self.model.predict(input_df)[0]
        apartment_price = int(apartment_price)
        response = ApartmentResponse
        response.price = apartment_price
        return response
