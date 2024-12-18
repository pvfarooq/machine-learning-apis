from pydantic import BaseModel


class SalaryRequest(BaseModel):
    yoe: float  # years of experience


class SalaryResponse(BaseModel):
    salary: float
