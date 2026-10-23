"""Pick the model by name. The rest of the pipeline only sees the SurvivalModel interface."""

from rtds.forest import ForestModel
from rtds.lstm_model import LSTMModel

VERSIONS = ("forest", "lstm")


def make_model(cfg, version, dataset_index, grid):
    if version == "forest":
        return ForestModel(cfg, dataset_index)
    if version == "lstm":
        return LSTMModel(cfg, dataset_index, grid)
    raise ValueError(f"unknown model version {version!r}; expected one of {VERSIONS}")
