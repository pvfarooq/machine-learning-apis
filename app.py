from fastapi import FastAPI, HTTPException

from schemas import (
    ApartmentRequest,
    ApartmentResponse,
    BreastCancerRequest,
    BreastCancerResponse,
    SalaryRequest,
    SalaryResponse,
)
from services import ApartmentService, BreastCancerPredictionService, SalaryService

app = FastAPI()


@app.post("/lr/predict-house-price/")
def predict_house_price(request: ApartmentRequest) -> ApartmentResponse:
    apartment_service = ApartmentService()
    try:
        response = apartment_service.predict(request)
        return response
    except Exception as e:
        error = f"Failed to predict house price. (error: {str(e)})"
        raise HTTPException(status_code=400, detail=error)


@app.post("/lr/predict-salary")
def predict_salary(request: SalaryRequest) -> SalaryResponse:
    salary_service = SalaryService()
    try:
        response = salary_service.predict(request)
        return response
    except Exception as e:
        error = f"Failed to predict salary. (error: {str(e)})"
        raise HTTPException(status_code=400, detail=error)


@app.post("/lr/predict-breast-cancer")
def predict_breast_cancer(request: BreastCancerRequest) -> BreastCancerResponse:
    breast_cancer_service = BreastCancerPredictionService()
    try:
        response = breast_cancer_service.predict(request)
        return response
    except Exception as e:
        error = f"Failed to predict breast cancer. (error: {str(e)})"
        raise HTTPException(status_code=400, detail=error)
