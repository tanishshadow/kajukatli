import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from xgboost import XGBClassifier, DMatrix


class ModelConfig(BaseModel):
    n_estimators: int = Field(gt=0)
    max_depth: int = Field(gt=0)
    learning_rate: float = Field(gt=0)
    n_jobs: int = Field(gt=0)
    seed: int


def train_model(features: pd.DataFrame, labels: np.ndarray, end_times: np.ndarray,
                train_end: int, config: ModelConfig) -> XGBClassifier:
    assert len(end_times) and np.all(end_times < train_end), "model training accepts train rows only"
    if len(np.unique(labels)) != 2:
        raise ValueError("training requires both classes; no model score will be fabricated")
    model = XGBClassifier(n_estimators=config.n_estimators, max_depth=config.max_depth,
        learning_rate=config.learning_rate, n_jobs=config.n_jobs, random_state=config.seed,
        tree_method="hist", objective="binary:logistic", eval_metric="aucpr",
        scale_pos_weight=float(np.sum(labels == 0) / np.sum(labels == 1)))
    model.fit(features, labels)
    return model


def probabilities(model: XGBClassifier, features: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(features)[:, 1]


def contributions(model: XGBClassifier, features: pd.DataFrame) -> np.ndarray:
    return model.get_booster().predict(DMatrix(features), pred_contribs=True)[:, :-1]
