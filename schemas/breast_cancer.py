from pydantic import BaseModel


class BreastCancerRequest(BaseModel):
    radius_mean: float
    texture_mean: float
    perimeter_mean: float
    area_mean: float
    smoothness_mean: float
    compactness_mean: float
    concavity_mean: float = 0.0
    concave_points_mean: float = 0.0
    symmetry_mean: float = 0.0
    fractal_dimension_mean: float = 0.0

    radius_se: float = 0.0
    texture_se: float = 0.0
    perimeter_se: float = 0.0
    area_se: float = 0.0
    smoothness_se: float = 0.0
    compactness_se: float = 0.0
    concavity_se: float = 0.0
    concave_points_se: float = 0.0
    symmetry_se: float = 0.0
    fractal_dimension_se: float = 0.0

    radius_worst: float = 0.0
    texture_worst: float = 0.0
    perimeter_worst: float = 0.0
    area_worst: float = 0.0
    smoothness_worst: float = 0.0
    compactness_worst: float = 0.0
    concavity_worst: float = 0.0
    concave_points_worst: float = 0.0
    symmetry_worst: float = 0.0
    fractal_dimension_worst: float = 0.0


class BreastCancerResponse(BaseModel):
    diagnosis: str
