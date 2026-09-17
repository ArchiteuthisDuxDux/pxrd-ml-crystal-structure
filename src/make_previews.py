import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

base = Path(__file__).resolve().parents[1]
df = pd.read_parquet(base / "data" / "clean" / "df_synth_summary_final_clean.parquet")
raw = base / "data" / "interim"
out = base / "outputs"

for role in ["cod", "crystaldb"]:
    sub = df[df["dataset_role"] == role].sample(2, random_state=7)
    fig, axes = plt.subplots(2, 1, figsize=(11, 6))

    for ax, (_, r) in zip(axes, sub.iterrows()):
        arr = np.load(raw / r["raw_spectrum_path"])
        x, y = arr[:, 0], arr[:, 1]
        ax.plot(x, y, lw=0.7)
        lam = r["primary_wavelength"]
        comp = r["phase_compositions"][0] if r["phase_compositions"] else "?"
        ax.set_title(
            f"{r['source']} | {comp} | SG {r['spacegroup_number']} | "
            f"lam={lam:.4f} | D={r['crystallite_size_nm']:.0f} nm",
            fontsize=9,
        )
        ax.set_xlabel("2theta, deg")
        ax.set_ylabel("counts")

    plt.tight_layout()

    f = out / f"preview_{role}.png"
    
    plt.savefig(f, dpi=130)
    plt.close()
    print("saved", f)
