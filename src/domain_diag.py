"""
Доменная диагностика: синтетика vs реал на именованных признаках.

Честные пары (окно [10, 70] градусов, полная маска в окне у обеих сторон):
    A) синтетика-Cu (lambda_1 ~ 1.54056)  <-> RRUFF (Cu)
    B) синтетика-синхротрон (lambda_1 ~ 1.23984) <-> opXRD-синхротрон

Выход (outputs/domain_diag/):
    features.csv, report.txt, roc_pairs.png,
    importance_pairA.png, importance_pairB.png,
    hist_top_pairA.png, hist_top_pairB.png

Запуск: python domain_diag.py
"""

import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split

import config as cfg
from domain_features import FEATURE_NAMES, compute_features

OUT = Path(__file__).parent / "outputs" / "domain_diag"
OUT.mkdir(parents=True, exist_ok=True)

GRID = np.linspace(cfg.GRID_LO, cfg.GRID_HI, cfg.GRID_N)
STEP = GRID[1] - GRID[0]
I0 = int(np.ceil(10.0 / STEP))
I1 = int(70.0 / STEP) + 1


def select_rows(index, M):

    """Маска полного покрытия окна [10,70] + подвыборки пар."""

    cov = np.asarray(M[:, I0:I1]).min(axis=1) == 1
    lam = index["lambda_1"].to_numpy(dtype=float)
    role = index["dataset_role"].to_numpy()
    rng = np.random.default_rng(42)

    def sample(mask, n):
        idx = np.where(mask & cov)[0]

        if len(idx) == 0:
            return np.array([], dtype=int)
        
        if len(idx) <= n:
            return idx
        
        return rng.choice(idx, size=n, replace=False)

    cu_synth = sample((np.abs(lam - 1.54056) < 0.001) & (role != "rruff") & (role != "opxrd"), 6000)
    rruff = sample(role == "rruff", 10**9)
    syn_synth = sample((np.abs(lam - 1.23984) < 0.001) & (role != "rruff") & (role != "opxrd"), 10000)
    opxrd_syn = sample((role == "opxrd") & (np.abs(lam - 1.23984) < 0.001), 10000)

    return {
        "A_synth_cu": cu_synth,
        "A_real_rruff": rruff,
        "B_synth_synch": syn_synth,
        "B_real_opxrd": opxrd_syn,
    }


def compute_batch(X, rows, tag):
    t0 = time.time()
    res = []

    with ThreadPoolExecutor(max_workers=16) as ex:
        for f in ex.map(
            lambda i: compute_features(np.asarray(X[i], dtype=np.float32), I0, I1, STEP),
            rows,
        ):
            res.append(f)
    df = pd.DataFrame(res, columns=FEATURE_NAMES)
    df["row_idx"] = rows
    print(f"  {tag}: {len(rows)} спектров за {time.time()-t0:.0f} c")
    return df


def ks_table(df_real, df_synth):
    rows = []
    for f in FEATURE_NAMES:
        a = df_real[f].dropna().to_numpy()
        b = df_synth[f].dropna().to_numpy()

        if len(a) < 10 or len(b) < 10:
            continue

        ks = ks_2samp(a, b)
        rows.append({
            "feature": f,
            "ks": ks.statistic,
            "ks_p": ks.pvalue,
            "real_p50": np.median(a),
            "synth_p50": np.median(b),
            "median_ratio": (np.median(b) + 1e-9) / (np.median(a) + 1e-9),
        })

    return pd.DataFrame(rows).sort_values("ks", ascending=False)


def run_pair(name, df_real, df_synth, X):
    print(f"\n===== ПАРА {name} =====")
    df_all = pd.concat([df_synth.assign(domain="synth"), df_real.assign(domain="real")],
                       ignore_index=True)
    Xv = df_all[FEATURE_NAMES].fillna(-1).to_numpy()
    yv = (df_all["domain"] == "synth").to_numpy()

    Xtr, Xte, ytr, yte = train_test_split(Xv, yv, test_size=0.3, random_state=42, stratify=yv)
    clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06,
                                         random_state=42)
    clf.fit(Xtr, ytr)
    prob = clf.predict_proba(Xte)[:, 1]
    auc = roc_auc_score(yte, prob)
    print(f"AUC = {auc:.4f}  (n_train={len(ytr)}, n_test={len(yte)})")

    pi = permutation_importance(clf, Xte, yte, scoring="roc_auc",
                                n_repeats=15, random_state=42)
    imp = pd.Series(pi.importances_mean, index=FEATURE_NAMES).sort_values(ascending=False)
    print("Топ-10 важностей (permutation, dAUC):")
    print(imp.head(10).round(4).to_string())

    ks = ks_table(df_real, df_synth)
    print("\nТоп-10 KS-расхождений:")
    print(ks.head(10)[["feature", "ks", "real_p50", "synth_p50", "median_ratio"]]
          .round(4).to_string(index=False))

    # ---------- графики ----------

    fpr, tpr, _ = roc_curve(yte, prob)
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    ax.plot(fpr, tpr, lw=2, label=f"AUC={auc:.3f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR"); ax.legend()
    ax.set_title(f"ROC: {name}")
    fig.tight_layout()
    fig.savefig(OUT / f"roc_{name}.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    top = imp.head(12)[::-1]
    colors = ["#c0392b" if v > 0.01 else "#7f8c8d" for v in top]
    ax.barh(top.index, top.values, color=colors)
    ax.set_xlabel("permutation importance (dAUC)")
    ax.set_title(f"Что выдаёт синтетику: {name}")
    fig.tight_layout()
    fig.savefig(OUT / f"importance_{name}.png", dpi=130)
    plt.close(fig)

    top6 = imp.head(6).index.tolist()
    fig, axes = plt.subplots(2, 3, figsize=(14, 7))

    for ax, f in zip(axes.ravel(), top6):
        lo, hi = np.nanpercentile(df_all[f], [1, 99])
        bins = np.linspace(lo, hi, 40)
        ax.hist(df_real[f].dropna(), bins=bins, alpha=0.55, density=True, label="real")
        ax.hist(df_synth[f].dropna(), bins=bins, alpha=0.55, density=True, label="synth")
        ax.set_title(f"{f} (KS={ks.set_index('feature').loc[f, 'ks']:.2f})", fontsize=10)
        ax.legend(fontsize=8)

    fig.suptitle(f"Распределения топ-признаков: {name}")
    fig.tight_layout()
    fig.savefig(OUT / f"hist_top_{name}.png", dpi=120)
    plt.close(fig)

    if auc < 0.65:
        verdict = "щель мизерная - можно учить головы"

    elif auc < 0.90:
        verdict = "заметная щель - калибровать ручки генератора"

    else:
        verdict = "огромная щель - искать баг уровня 'настроек SG'"

    print(f"ВЕРДИКТ: {verdict}")

    return auc, imp, ks, verdict, df_all


def main():
    index = pd.read_parquet(cfg.INDEX_PARQUET)
    X = np.memmap(cfg.X_INTENSITY_PATH, dtype=np.float16, mode="r",
                  shape=(len(index), cfg.GRID_N))
    M = np.memmap(cfg.MASK_PATH, dtype=np.uint8, mode="r",
                  shape=(len(index), cfg.GRID_N))

    print("Отбор честных пар (полная маска в [10,70])...")
    sets = select_rows(index, M)

    for k, v in sets.items():
        print(f"  {k}: {len(v)}")

    feats = {}

    for tag, rows in sets.items():
        feats[tag] = compute_batch(X, rows, tag)

    all_df = pd.concat(
        [feats[k].assign(pair_tag=k) for k in sets],
        ignore_index=True,
    )
    all_df.to_csv(OUT / "features.csv", index=False)

    aucA, impA, ksA, vA, _ = run_pair(
        "pairA_synthCu_vs_RRUFF",
        feats["A_real_rruff"], feats["A_synth_cu"], X,
    )
    aucB, impB, ksB, vB, _ = run_pair(
        "pairB_synthSynch_vs_opXRD",
        feats["B_real_opxrd"], feats["B_synth_synch"], X,
    )

    with open(OUT / "report.txt", "w", encoding="utf-8") as f:
        f.write(f"pairA (synth-Cu vs RRUFF): AUC={aucA:.4f} -> {vA}\n")
        f.write(f"pairB (synth-synch vs opXRD): AUC={aucB:.4f} -> {vB}\n\n")
        f.write("=== pairA importances ===\n" + impA.round(4).to_string() + "\n\n")
        f.write("=== pairA KS ===\n" + ksA.round(4).to_string(index=False) + "\n\n")
        f.write("=== pairB importances ===\n" + impB.round(4).to_string() + "\n\n")
        f.write("=== pairB KS ===\n" + ksB.round(4).to_string(index=False) + "\n")

    print(f"\nОтчёт и графики: {OUT}")

    return 0

if __name__ == "__main__":
    sys.exit(main())
