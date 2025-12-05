from pydantic import BaseModel
from typing import List


class PredictRequest(BaseModel):
    image: List[List[float]]  # 28x28 grayscale 0-255


class PredictResponse(BaseModel):
    class_id: int
    label: str
