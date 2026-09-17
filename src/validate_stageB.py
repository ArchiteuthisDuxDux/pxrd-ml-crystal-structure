import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config as cfg

GRID = np.linspace(cfg.GRID_LO, cfg.GRID_HI, cfg.GRID_N)

index = pd.read_parquet(cfg.INDEX_PARQUET)
X = np.memmap(cfg.X_INTENSITY_PATH, dtype=np.float16, mode="r",
              shape=(len(index), cfg.GRID_N))
M = np.memmap(cfg.MASK_PATH, dtype=np.uint8, mode="r",
              shape=(len(index), cfg.GRID_N))

print("=" * 70)
print("ИТОГОВАЯ ВАЛИДАЦИЯ ЭТАПА B")
print("=" * 70)
print(f"Строк: {len(index)}")

fail = int(index["preprocess_failed"].sum())
mask_sum = np.asarray(M).sum(axis=1)
print(f"Провалов: {fail} | полностью замаскированных: {int((mask_sum == 0).sum())}")

print("\nПо split_role:")
print(index["split_role"].value_counts().to_string())

print("\nДоля валидной области (маска) по источникам:")
for role, g in index.groupby("dataset_role"):
    frac = mask_sum[g["row_idx"].to_numpy()] / cfg.GRID_N
    print(f"  {role:10s}: медиана {np.median(frac):.1%} | "
          f"p05 {np.percentile(frac, 5):.1%} | мин {frac.min():.1%}")

print("\nМаксимум интенсивности внутри маски (должен быть ~1):")
for role in ["cod", "crystaldb", "rruff", "opxrd"]:
    idx = index[index["dataset_role"] == role]["row_idx"].to_numpy()
    sub = np.random.default_rng(1).choice(idx, size=min(500, len(idx)), replace=False)
    maxes = np.array([X[i][M[i] == 1].max() for i in sub])
    print(f"  {role:10s}: медиана {np.median(maxes):.3f} | "
          f"p05 {np.percentile(maxes, 5):.3f} | p95 {np.percentile(maxes, 95):.3f}")

# --- визуальный контроль: по одному спектру каждого источника ---

fig, axes = plt.subplots(4, 1, figsize=(11, 11))
for ax, role in zip(axes, ["cod", "crystaldb", "rruff", "opxrd"]):
    idx = index[index["dataset_role"] == role]
    row = idx.sample(1, random_state=3).iloc[0]
    i = row["row_idx"]
    m = M[i] == 1
    ax.plot(GRID[m], X[i][m], lw=0.7)
    ax.plot(GRID[~m], X[i][~m], lw=0.7, color="red", alpha=0.3)
    ax.set_title(f"{role}: {row['sample_id']} | λ1={row['lambda_1']} | "
                 f"маска {int(m.sum())}/{cfg.GRID_N} | split={row['split_role']}",
                 fontsize=9)
    ax.set_ylabel("I (norm)")
axes[-1].set_xlabel("2theta, deg")
plt.tight_layout()
plt.savefig("outputs/preview_stageB.png", dpi=130)
print("\nПревью: outputs/preview_stageB.png")

gate = fail == 0 and (mask_sum == 0).sum() == 0
print("\nГЕЙТ:", "ПРОЙДЕН" if gate else "ПРОВАЛЕН")
sys.exit(0 if gate else 1)
