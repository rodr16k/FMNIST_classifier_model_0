import torch
import pathlib
from .model import FashionMNISTModel

MODEL_PATH = pathlib.Path(__file__).parent.parent / \
    'models/trained_models/FMNIST_pytorch_cnn_model.pth'

model = FashionMNISTModel()
model.load_state_dict(torch.load(MODEL_PATH,
                                 map_location='cpu'))
model.eval()
