"""Иерархический сэмплер режимов: 50/50 источник -> режим ~ fraction_all."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import gen_config as gcfg


class ModeSampler:
    """
    Сэмплирует режим (lam1, lam2, x_min, x_max, step) так, чтобы
    распределение совпадало с экспериментальным:

        1) источник rruff/opxrd с вероятностью SOURCE_WEIGHTS;
        2) внутри источника - пропорционально частоте встречаемости.

    Режимы с неизвестной lambda или шагом исключаются.
    """

    def __init__(self, seed=None):
        rng = np.random.default_rng(seed)
        self._sources = []
        self._tables = []

        for source in ["rruff", "opxrd"]:
            path = gcfg.MODES_RRUFF_CSV if source == "rruff" else gcfg.MODES_OPXRD_CSV
            df = pd.read_csv(path)

            bad = df["lambda_1"].isna() | df["step_mode"].isna()
            dropped = int(bad.sum())
            df = df[~bad]

            lam1 = df["lambda_1"].to_numpy(float)
            lam2 = df["lambda_2"].to_numpy(float)
            xmin = df["x_min_mode"].to_numpy(float)
            xmax = df["x_max_mode"].to_numpy(float)
            step = df["step_mode"].to_numpy(float)

            w = df["fraction_all"].to_numpy(float)
            w = w / w.sum()

            cdf = np.cumsum(w)
            cdf[-1] = 1.0

            self._sources.append(source)
            self._tables.append(
                dict(lam1=lam1, lam2=lam2, xmin=xmin, xmax=xmax,
                     step=step, cdf=cdf, n=len(df), dropped_nan=dropped)
            )

        p = np.array([gcfg.SOURCE_WEIGHTS[s] for s in self._sources], dtype=float)
        self._source_cdf = np.cumsum(p / p.sum())

        self.rng = rng

    def sample(self):
        u = self.rng.random()
        si = int(np.searchsorted(self._source_cdf, u, side="right"))
        si = min(si, len(self._sources) - 1)

        t = self._tables[si]
        j = int(np.searchsorted(t["cdf"], self.rng.random(), side="right"))
        j = min(j, t["n"] - 1)

        lam1 = float(t["lam1"][j])
        lam2 = float(t["lam2"][j]) if np.isfinite(t["lam2"][j]) else None

        return {
            "source": self._sources[si],
            "mode_id": int(j),
            "lambda_primary": round(lam1, 5),
            "lambda_secondary": round(lam2, 5) if lam2 is not None else None,
            "x_min": float(t["xmin"][j]),
            "x_max": float(t["xmax"][j]),
            "step": float(t["step"][j]),
        }


if __name__ == "__main__":
    s = ModeSampler(seed=0)
    from collections import Counter
    src = Counter()
    lams = Counter()
    for _ in range(20000):
        m = s.sample()
        src[m["source"]] += 1
        lams[(m["lambda_primary"], m["lambda_secondary"])] += 1
    print("Источники:", dict(src))
    print("Топ-6 lambda-пар:", lams.most_common(6))
