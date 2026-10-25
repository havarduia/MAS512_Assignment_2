# Results of main.py, saved once so the presentation figure scripts don't rerun YOLO and the training

from pathlib import Path

import numpy as np

PATH = Path(__file__).resolve().parent / "result/q1_results.npz"


def save(**arrays):
    np.savez(PATH, **arrays)


def load():
    if not PATH.exists():
        raise FileNotFoundError(f"{PATH.name} not found: run main.py first")
    return dict(np.load(PATH))
