import torch
from fastapi import APIRouter, Depends
from src.api.schemas import PredictRequest, PredictResponse
from src.api.deps import get_model
from src.ml.transform import image_to_tensor

router = APIRouter()

LABELS = 'T-shirt/top,Trouser,Pullover,Dress,Coat,Sandal,Shirt,Sneaker,Bag,Ankle boot'.split(
    ',')


@router.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest,
            model=Depends(get_model)):
    x = image_to_tensor(req.image).unsqueeze(0)
    with torch.no_grad():
        out = model(x)
    idx = int(out.argmax(1))
    return PredictResponse(class_id=idx, label=LABELS[idx])
