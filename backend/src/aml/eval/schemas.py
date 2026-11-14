from pydantic import BaseModel, ConfigDict, Field


class Cluster(BaseModel):
    model_config = ConfigDict(extra="forbid")
    edge_ids: list[int]
    accounts: list[int]
    score: float = Field(ge=0)


class BinaryMetrics(BaseModel):
    tp: int
    fp: int
    fn: int
    tn: int
    precision: float
    recall: float
    f1: float
    pr_auc: float | None
