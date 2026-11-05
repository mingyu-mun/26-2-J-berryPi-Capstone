from .base import CandidateModel, model_input, n_samples
from .neural import CNNBiLSTMModel, MLPModel
from .trees import LightGBMModel, RandomForestModel, XGBoostModel

REGISTRY: dict[str, type[CandidateModel]] = {
    m.name: m for m in (MLPModel, CNNBiLSTMModel, RandomForestModel, XGBoostModel, LightGBMModel)
}


def build_model(name: str, n_classes: int, seed: int = 0, **params) -> CandidateModel:
    if name not in REGISTRY:
        raise KeyError(f"알 수 없는 모델 '{name}'. 사용 가능: {sorted(REGISTRY)}")
    return REGISTRY[name](n_classes, seed, **params)


__all__ = ["CandidateModel", "model_input", "n_samples", "REGISTRY", "build_model", "MLPModel", "CNNBiLSTMModel",
           "RandomForestModel", "XGBoostModel", "LightGBMModel"]
