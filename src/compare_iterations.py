import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]

# KS-статистики топ-признаков по итерациям из outputs/domain_diag*_log.txt
# v1 - исходные параметры, v4 - по-семейные декорации

versions = ["v1", "v2", "v3", "v4"]

pairA = {
    "noise_pt":     [0.895, 0.660, 0.489, 0.30],
    "autocorr_3":   [0.849, 0.941, 0.881, 0.608],
    "max_slope":    [0.809, 0.767, 0.483, 0.399],
    "halo_height":  [0.682, 0.682, 0.713, 0.491],
    "snr":          [0.826, 0.748, 0.674, 0.382],
    "fwhm_median":  [0.30,  0.20,  0.469, 0.15],
    "bg_p10":       [0.727, 0.10,  0.10,  0.10],
}

pairB = {
    "fwhm_median":  [0.726, 0.10,  0.10,  0.10],
    "bg_p10":       [0.26,  0.383, 0.537, 0.329],
    "noise_pt":     [0.575, 0.521, 0.30,  0.25],
    "skew":         [0.30,  0.661, 0.661, 0.628],
    "iqr":          [0.20,  0.30,  0.667, 0.455],
    "gini":         [0.529, 0.515, 0.30,  0.379],
    "frac_low":     [0.30,  0.473, 0.619, 0.366],
}

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

for ax, data, title in [
    (axes[0], pairA, "Пара A: синт-Cu vs RRUFF"),
    (axes[1], pairB, "Пара B: синт-синхротрон vs opXRD"),
]:
    for name, ks in data.items():
        ax.plot(versions, ks, marker="o", lw=1.8, label=name)

    ax.axhline(0.3, color="green", ls="--", lw=1.2, label="цель KS<=0.3")
    ax.set_ylim(0, 1)
    ax.set_title(title)
    ax.set_ylabel("KS-статистика (0=совпадение, 1=разделение)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, ncol=2)

fig.suptitle("Эволюция доменной щели по итерациям калибровки генератора")
plt.tight_layout()

target = BASE / "outputs" / "domain_diag" / "calibration_evolution.png"
target.parent.mkdir(parents=True, exist_ok=True)

plt.savefig(target, dpi=130)

print("saved", target)
