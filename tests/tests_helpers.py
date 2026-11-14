"""Small normalized transaction fixtures, never random transaction samples."""
import pandas as pd


def frame(edges: list[tuple[int, int, int, float, int]]) -> pd.DataFrame:
    return pd.DataFrame({"tx_id": list(range(len(edges))), "src": [e[0] for e in edges],
        "dst": [e[1] for e in edges], "ts": [e[2] for e in edges],
        "amt_out": [float(e[3]) for e in edges], "amt_in": [float(e[3]) for e in edges],
        "cur_out": ["USD"] * len(edges), "cur_in": ["USD"] * len(edges), "fmt": ["ACH"] * len(edges),
        "label": [e[4] for e in edges], "is_self_loop": [e[0] == e[1] for e in edges]})
