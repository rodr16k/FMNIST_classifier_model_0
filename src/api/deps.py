from functools import lru_cache
from src.ml.weights import model


@lru_cache(maxsize=1)
def get_model():
    return model
