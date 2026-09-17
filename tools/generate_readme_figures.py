from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from PIL import Image


PROJECT = Path(__file__).resolve().parents[1]
OUTPUT = PROJECT / "docs" / "assets" / "readme"
OUTPUT.mkdir(parents=True, exist_ok=True)

INK = "#20242B"
MUTED = "#68717D"
GRID = "#DDE3E8"
PAPER = "#FFFFFF"
MINT = "#4BD0A0"
MINT_DARK = "#159A71"
BLUE = "#4B82D0"
AMBER = "#F2A93B"
RED = "#D95F59"
PURPLE = "#8A6FD1"
LIGHT_MINT = "#E9F8F3"
LIGHT_BLUE = "#EAF1FB"
LIGHT_AMBER = "#FFF4DF"
LIGHT_RED = "#FBEDEC"

mpl.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 16,
        "axes.titleweight": "bold",
        "axes.labelcolor": INK,
        "axes.edgecolor": GRID,
        "axes.facecolor": PAPER,
        "figure.facecolor": PAPER,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "text.color": INK,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.7,
        "grid.alpha": 0.8,
        "legend.frameon": False,
    }
)


def new_figure(title: str, subtitle: str = "") -> tuple[plt.Figure, plt.Axes]:
    fig = plt.figure(figsize=(9.6, 5.4), dpi=100)
    ax = fig.add_axes([0.08, 0.13, 0.88, 0.70])
    title_size = 18 if len(title) <= 67 else 16
    fig.text(0.06, 0.925, title, fontsize=title_size, fontweight="bold", color=INK)
    if subtitle:
        fig.text(0.06, 0.875, subtitle, fontsize=9.5, color=MUTED)
    return fig, ax


def save(fig: plt.Figure, filename: str) -> None:
    path = OUTPUT / filename
    fig.savefig(path, dpi=100, facecolor=PAPER, bbox_inches=None, pad_inches=0)
    plt.close(fig)
    with Image.open(path) as image:
        if image.size != (960, 540):
            raise RuntimeError(f"{filename}: expected 960x540, got {image.size}")


def box(
    ax: plt.Axes,
    xy: tuple[float, float],
    width: float,
    height: float,
    title: str,
    body: str,
    face: str = LIGHT_MINT,
    edge: str = MINT_DARK,
    title_size: float = 10,
    body_size: float = 8.5,
) -> None:
    x, y = xy
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        linewidth=1.3,
        edgecolor=edge,
        facecolor=face,
    )
    ax.add_patch(patch)
    ax.text(x + width / 2, y + height * 0.66, title, ha="center", va="center", fontsize=title_size, fontweight="bold")
    ax.text(x + width / 2, y + height * 0.32, body, ha="center", va="center", fontsize=body_size, color=MUTED, linespacing=1.25)


def arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float], color: str = MUTED) -> None:
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=13, linewidth=1.25, color=color))


def load_table(relative: str) -> pd.DataFrame:
    return pd.read_parquet(PROJECT / relative)


rruff = load_table("data/clean/df_rruff_clean.parquet")
opxrd = load_table("data/clean/df_opxrd_clean.parquet")
synth = load_table("data/clean/df_synth_summary_final_clean.parquet")
rruff_sg = load_table("data/clean/ft_pool_rruff_with_rruff_sg.parquet")
opxrd_ft = load_table("data/clean/ft_pool_opxrd.parquet")
combined_ft = load_table("data/clean/ft_pool_combined_with_rruff_sg.parquet")


def figure_pipeline() -> None:
    fig, ax = new_figure(
        "From raw diffraction patterns to a leakage-controlled multitask model",
        "The repository is executable in notebook order; each stage writes explicit artifacts consumed by the next stage.",
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    stages = [
        ("RAW", "RRUFF · opXRD\nCOD · crystalDB", LIGHT_BLUE, BLUE),
        ("PREPARE", "parse · clean\nlabel masks", LIGHT_MINT, MINT_DARK),
        ("GENERATE", "physics-based\nsynthetic PXRD", LIGHT_AMBER, AMBER),
        ("PRETRAIN", "460,133\nsynthetic patterns", "#F1EDFB", PURPLE),
        ("FINE-TUNE", "nested grouped CV\nreal measurements", LIGHT_RED, RED),
        ("EVALUATE", "specialists · combined\ncross-source tests", LIGHT_MINT, MINT_DARK),
    ]
    xs = np.linspace(0.01, 0.84, len(stages))
    for i, ((label, body, face, edge), x) in enumerate(zip(stages, xs)):
        box(ax, (x, 0.42), 0.145, 0.32, label, body, face, edge, title_size=9.5, body_size=7.7)
        if i < len(stages) - 1:
            arrow(ax, (x + 0.147, 0.58), (xs[i + 1] - 0.006, 0.58), edge)
    ax.text(0.01, 0.22, "24 ordered master notebooks", fontsize=12, fontweight="bold")
    ax.text(0.01, 0.13, "Setup → EDA → parquet pools → synthetic generator → preprocessing → pretraining → fine-tuning → ablations → transfer", color=MUTED, fontsize=9)
    ax.text(0.99, 0.02, "DS_XRD_project", ha="right", color=MUTED, fontsize=8)
    save(fig, "01_pipeline_overview.png")


def figure_dataset_scale() -> None:
    fig, ax = new_figure(
        "Dataset scale and supervision roles",
        "Only the labeled real subset is used for fine-tuning; the large unlabeled opXRD tail is kept separate.",
    )
    labels = ["RRUFF\nreal", "opXRD\nlabeled", "opXRD\nunlabeled", "COD + crystalDB\nsynthetic"]
    values = [len(rruff), len(opxrd_ft), int((opxrd.phase_count == 0).sum()), len(synth)]
    colors = [MINT_DARK, BLUE, "#A8B1BB", PURPLE]
    bars = ax.bar(labels, values, color=colors, width=0.62)
    ax.set_yscale("log")
    ax.set_ylabel("number of patterns · log scale")
    ax.set_ylim(700, 800000)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value * 1.14, f"{value:,}", ha="center", va="bottom", fontweight="bold")
    ax.text(0.99, 0.03, "Real FT pool: 1,359 + 2,116 = 3,475 rows", transform=ax.transAxes, ha="right", color=MUTED, fontsize=9)
    save(fig, "02_dataset_scale.png")


def read_spectrum(relative: str) -> np.ndarray:
    path = PROJECT / "data" / "interim" / relative
    if not path.exists():
        raise FileNotFoundError(
            f"Missing derived spectrum {path}. Run notebooks 02 and 03 before regenerating README figures."
        )
    array = np.load(path)
    if array.ndim != 2:
        raise ValueError(f"Unexpected spectrum shape {array.shape} for {path}")
    if array.shape[0] == 2:
        return array
    if array.shape[1] == 2:
        return array.T
    raise ValueError(f"Unexpected spectrum shape {array.shape} for {path}")


def representative_rows(frame: pd.DataFrame, count: int = 6) -> pd.DataFrame:
    valid = frame[frame.raw_spectrum_path.notna()].sort_values(["x_max", "step"])
    indices = np.linspace(0, len(valid) - 1, count, dtype=int)
    return valid.iloc[indices]


def figure_profiles(frame: pd.DataFrame, source_name: str, filename: str, color: str) -> None:
    fig, ax = new_figure(
        f"{source_name}: representative experimental diffraction profiles",
        "Raw intensity is displayed after per-pattern max normalization; preprocessing later uses a common 0–90° grid and sqrt scaling.",
    )
    for offset, (_, row) in enumerate(representative_rows(frame).iterrows()):
        x, y = read_spectrum(str(row.raw_spectrum_path))
        y = np.nan_to_num(y.astype(float), nan=0.0)
        y -= np.nanmin(y)
        y /= max(float(np.nanmax(y)), 1e-12)
        ax.plot(x, y + offset * 1.08, color=color, linewidth=0.9, alpha=0.9)
    ax.set_xlim(0, 90)
    ax.set_xlabel(r"diffraction angle $2\theta$ (degrees)")
    ax.set_ylabel("normalized intensity + offset")
    ax.set_yticks([])
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    save(fig, filename)


def figure_acquisition(frame: pd.DataFrame, source_name: str, filename: str, color: str) -> None:
    fig = plt.figure(figsize=(9.6, 5.4), dpi=100)
    fig.text(0.06, 0.925, f"{source_name}: acquisition geometry is not uniform", fontsize=18, fontweight="bold")
    fig.text(0.06, 0.875, "Variable angular coverage and native sampling are retained as masks after resampling.", fontsize=9.5, color=MUTED)
    ax1 = fig.add_axes([0.08, 0.16, 0.40, 0.62])
    ax2 = fig.add_axes([0.56, 0.16, 0.40, 0.62])
    xmax = pd.to_numeric(frame.x_max, errors="coerce").dropna().clip(upper=180)
    step = pd.to_numeric(frame.step, errors="coerce").dropna()
    step = step[(step > 0) & (step < step.quantile(0.995))]
    ax1.hist(xmax, bins=32, color=color, alpha=0.88, edgecolor=PAPER)
    ax1.axvline(90, color=RED, linestyle="--", linewidth=1.4, label="model limit: 90°")
    ax1.set_xlabel(r"maximum measured $2\theta$ (degrees)")
    ax1.set_ylabel("patterns")
    ax1.legend(fontsize=8)
    ax2.hist(step, bins=32, color=color, alpha=0.88, edgecolor=PAPER)
    ax2.set_xlabel(r"native angular step $\Delta 2\theta$ (degrees)")
    ax2.set_ylabel("patterns")
    fig.text(0.28, 0.085, f"median max angle: {xmax.median():.1f}°", ha="center", fontsize=9, color=MUTED)
    fig.text(0.76, 0.085, f"median step: {step.median():.4f}°", ha="center", fontsize=9, color=MUTED)
    save(fig, filename)


SYSTEM_ORDER = ["triclinic", "monoclinic", "orthorhombic", "tetragonal", "trigonal", "hexagonal", "cubic"]


def normalized_system(value: object) -> str:
    if pd.isna(value):
        return "missing"
    text = str(value).strip().lower().replace("_", " ")
    aliases = {"rhombohedral": "trigonal"}
    return aliases.get(text, text)


def figure_labels(frame: pd.DataFrame, source_name: str, filename: str, color: str) -> None:
    fig = plt.figure(figsize=(9.6, 5.4), dpi=100)
    title = f"{source_name}: label balance and target availability"
    fig.text(0.06, 0.925, title, fontsize=18 if len(title) <= 67 else 16, fontweight="bold")
    fig.text(0.06, 0.875, "Missing labels are handled by task-specific masks; they are never imputed as ground truth.", fontsize=9.5, color=MUTED)
    ax1 = fig.add_axes([0.12, 0.16, 0.44, 0.62])
    ax2 = fig.add_axes([0.68, 0.16, 0.27, 0.62])
    systems = frame.crystal_system.map(normalized_system)
    counts = systems.value_counts()
    order = [name for name in SYSTEM_ORDER if name in counts.index]
    if "missing" in counts.index:
        order.append("missing")
    values = [int(counts.get(name, 0)) for name in order]
    ax1.barh(order[::-1], values[::-1], color=color)
    ax1.set_xlabel("rows")
    for y, value in enumerate(values[::-1]):
        ax1.text(value, y, f"  {value:,}", va="center", fontsize=8)
    targets = ["lattice", "space group", "crystal system", "elements"]
    mask_cols = ["head_mask_lattice", "head_mask_spacegroup", "head_mask_crystal_system", "head_mask_elements"]
    fallback_cols = ["has_lattice", "has_spacegroup", "crystal_system", "has_elements"]
    coverage = []
    for mask_col, fallback_col in zip(mask_cols, fallback_cols):
        if mask_col in frame:
            coverage.append(100 * pd.to_numeric(frame[mask_col], errors="coerce").fillna(0).astype(bool).mean())
        elif fallback_col == "crystal_system":
            coverage.append(100 * frame[fallback_col].notna().mean())
        else:
            coverage.append(100 * pd.to_numeric(frame[fallback_col], errors="coerce").fillna(0).astype(bool).mean())
    bars = ax2.barh(targets[::-1], coverage[::-1], color=[color] * 4)
    ax2.set_yticks([])
    ax2.set_xlim(0, 100)
    ax2.set_xlabel("label coverage (%)")
    for bar, value, label in zip(bars, coverage[::-1], targets[::-1]):
        y = bar.get_y() + bar.get_height() / 2
        ax2.text(1.5, y, label, va="center", fontsize=7.7, color=PAPER, fontweight="bold")
        if value > 82:
            ax2.text(value - 1.5, y, f"{value:.1f}%", va="center", ha="right", fontsize=8, color=PAPER)
        else:
            ax2.text(value + 1.5, y, f"{value:.1f}%", va="center", fontsize=8)
    save(fig, filename)


def figure_target_distributions() -> None:
    fig = plt.figure(figsize=(9.6, 5.4), dpi=100)
    fig.text(0.06, 0.925, "Real and synthetic target domains are not identical", fontsize=18, fontweight="bold")
    fig.text(0.06, 0.875, "Synthetic pretraining provides scale; real-data fine-tuning is required to adapt the target distribution.", fontsize=9.5, color=MUTED)
    ax1 = fig.add_axes([0.08, 0.16, 0.40, 0.62])
    ax2 = fig.add_axes([0.56, 0.16, 0.40, 0.62])
    frames = [(rruff, "RRUFF", MINT_DARK), (opxrd_ft, "opXRD labeled", BLUE), (synth.sample(min(len(synth), 100000), random_state=17), "synthetic", PURPLE)]
    for frame, label, color in frames:
        a = pd.to_numeric(frame.lattice_a, errors="coerce").dropna()
        a = a[(a > 0) & (a < a.quantile(0.995))]
        ax1.hist(a, bins=45, density=True, histtype="step", linewidth=1.7, color=color, label=label)
        volume = pd.to_numeric(frame.lattice_a, errors="coerce") * pd.to_numeric(frame.lattice_b, errors="coerce") * pd.to_numeric(frame.lattice_c, errors="coerce")
        volume = volume[(volume > 0) & np.isfinite(volume)]
        ax2.hist(np.log10(volume), bins=45, density=True, histtype="step", linewidth=1.7, color=color, label=label)
    ax1.set_xlabel(r"lattice parameter $a$ (Å)")
    ax1.set_ylabel("density")
    ax2.set_xlabel(r"$\log_{10}(abc)$ proxy (Å³)")
    ax2.set_ylabel("density")
    ax1.legend(fontsize=8)
    ax2.legend(fontsize=8)
    save(fig, "09_target_distributions.png")


def figure_generator_physics() -> None:
    fig, ax = new_figure(
        "Synthetic generator: physics + randomized nuisance effects",
        "Peak positions and structure amplitudes follow physics; instrumental and sample effects are randomized for domain coverage.",
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    box(ax, (0.01, 0.54), 0.29, 0.31, "1 · Bragg condition", "$2d_{hkl}\\sin\\theta=n\\lambda$\npeak positions", LIGHT_BLUE, BLUE, 11, 9.5)
    box(ax, (0.355, 0.54), 0.29, 0.31, "2 · Structure factor", "$F_{hkl}=\\sum_j o_j f_j(s)e^{2\\pi i\\mathbf{h}\\cdot\\mathbf{r}_j}$\nrelative amplitudes", "#F1EDFB", PURPLE, 11, 8)
    box(ax, (0.70, 0.54), 0.29, 0.31, "3 · Powder intensity", "$I_{hkl}\\propto m_{hkl}|F_{hkl}|^2LP(\\theta)$\nreflection weights", LIGHT_MINT, MINT_DARK, 11, 8.5)
    arrow(ax, (0.305, 0.695), (0.347, 0.695), BLUE)
    arrow(ax, (0.65, 0.695), (0.692, 0.695), PURPLE)
    ax.text(0.02, 0.36, "Randomized experimental effects", fontsize=12, fontweight="bold")
    chips = [("Kα doublet", BLUE), ("Caglioti FWHM", MINT_DARK), ("size broadening", PURPLE), ("texture", AMBER), ("background", RED), ("noise", MUTED)]
    widths = [0.14, 0.16, 0.16, 0.13, 0.13, 0.13]
    x = 0.02
    for (label, color), width in zip(chips, widths):
        box(ax, (x, 0.16), width, 0.13, label, "", PAPER, color, 8.3, 1)
        x += width + 0.012
    save(fig, "10_generator_physics.png")


def figure_generator_validation() -> None:
    fig, ax = new_figure(
        "Generator validation against an independent diffraction engine",
        "A saved 190-structure check compares the custom engine with pymatgen XRDCalculator.",
    )
    labels = ["median profile\ncorrelation", "mean profile\ncorrelation", "profiles with\nr ≥ 0.97"]
    values = [99.75, 99.52, 99.50]
    colors = [MINT_DARK, BLUE, PURPLE]
    bars = ax.bar(labels, values, color=colors, width=0.6)
    ax.set_ylim(99.25, 99.90)
    ax.set_ylabel("profile agreement (%)")
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.03, f"{value:.2f}%", ha="center", fontweight="bold")
    ax.text(0.99, 0.04, "n = 190 structures · independent pymatgen reference", transform=ax.transAxes, ha="right", fontsize=8.5, color=MUTED)
    save(fig, "11_generator_validation.png")


def figure_domain_shift() -> None:
    report = pd.read_csv(PROJECT / "outputs/domain_check_report.csv")
    fig = plt.figure(figsize=(9.6, 5.4), dpi=100)
    fig.text(0.06, 0.925, "Synthetic-to-real domain gap remains measurable", fontsize=18, fontweight="bold")
    fig.text(0.06, 0.875, "A classifier separates wavelength-matched synthetic and real patterns almost perfectly; pretraining alone is insufficient.", fontsize=9.5, color=MUTED)
    ax1 = fig.add_axes([0.08, 0.16, 0.50, 0.62])
    ax2 = fig.add_axes([0.65, 0.16, 0.30, 0.62])
    labels = [f"{row.pair}\n{row['mode']}" for _, row in report.iterrows()]
    auc = report.auc.to_numpy()
    bars = ax1.barh(labels[::-1], auc[::-1], color=[MINT_DARK, MINT, BLUE, "#88AEE4", PURPLE, "#B1A2E0"][::-1])
    ax1.set_xlim(0.995, 1.0002)
    ax1.set_xlabel("synthetic-vs-real ROC AUC")
    for bar, value in zip(bars, auc[::-1]):
        ax1.text(value - 0.00005, bar.get_y() + bar.get_height() / 2, f"{value:.4f}", ha="right", va="center", fontsize=8, color=INK)
    feature_counts: dict[str, int] = {}
    for text in report.top_features:
        for feature in str(text).split(";"):
            feature_counts[feature.strip()] = feature_counts.get(feature.strip(), 0) + 1
    top = sorted(feature_counts.items(), key=lambda item: (-item[1], item[0]))[:6]
    ax2.barh([name for name, _ in top][::-1], [count for _, count in top][::-1], color=RED)
    ax2.set_xlabel("times in top-feature list")
    ax2.set_title("Domain cues", fontsize=11)
    save(fig, "12_domain_shift.png")


def figure_architecture() -> None:
    fig, ax = new_figure(
        "XRDNet V2: one shared representation, four supervised heads",
        "Each loss is multiplied by a per-row availability mask, so incomplete labels remain usable without manufacturing targets.",
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    box(ax, (0.01, 0.40), 0.17, 0.34, "Input", "4096 intensity bins\n+ validity mask\n+ wavelength metadata", LIGHT_BLUE, BLUE, 11, 8.6)
    box(ax, (0.27, 0.40), 0.20, 0.34, "1D CNN backbone", "residual blocks\nwidth ×3\n11.3M parameters", "#F1EDFB", PURPLE, 11, 8.6)
    box(ax, (0.56, 0.40), 0.16, 0.34, "Latent", "1536-dimensional\nshared embedding", LIGHT_MINT, MINT_DARK, 11, 8.6)
    arrow(ax, (0.185, 0.57), (0.26, 0.57), BLUE)
    arrow(ax, (0.48, 0.57), (0.55, 0.57), PURPLE)
    heads = [
        ("Lattice", "a,b,c, α,β,γ,V", BLUE, LIGHT_BLUE),
        ("Space group", "230 classes", PURPLE, "#F1EDFB"),
        ("Crystal system", "7 classes", MINT_DARK, LIGHT_MINT),
        ("Elements", "multi-label", AMBER, LIGHT_AMBER),
    ]
    ys = [0.72, 0.51, 0.30, 0.09]
    for (label, body, edge, face), y in zip(heads, ys):
        box(ax, (0.81, y), 0.18, 0.15, label, body, face, edge, 9, 7.5)
        arrow(ax, (0.725, 0.57), (0.80, y + 0.075), edge)
    ax.text(0.01, 0.16, r"$\mathcal{L}=m_{lat}\mathcal{L}_{lat}+m_{sg}\mathcal{L}_{sg}+m_{sys}\mathcal{L}_{sys}+m_{el}\mathcal{L}_{el}$", fontsize=12, color=INK)
    save(fig, "13_model_architecture.png")


def figure_timeline() -> None:
    fig, ax = new_figure(
        "Experiment chronology: successful results and negative results are both retained",
        "The project evolved through explicit baselines, leakage repair, source specialists, label enrichment, and controlled ablations.",
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    stages = [
        ("V1", "synthetic\nbaseline", BLUE),
        ("V2", "larger\nbackbone", PURPLE),
        ("FT v1", "random split\nleakage found", RED),
        ("FT v2", "GroupKFold\n+ replay", MINT_DARK),
        ("Pseudo", "rejected\nexperiment", RED),
        ("Specialists", "RRUFF-only\nopXRD-only", BLUE),
        ("No replay", "combined\nreal FT", AMBER),
        ("SG enrich", "RRUFF + IMA\nexact names", PURPLE),
        ("Final", "paired CV\n+ transfer", MINT_DARK),
    ]
    y = 0.52
    ax.plot([0.04, 0.96], [y, y], color=GRID, linewidth=3)
    xs = np.linspace(0.05, 0.95, len(stages))
    for i, ((name, body, color), x) in enumerate(zip(stages, xs)):
        ax.scatter([x], [y], s=145, color=color, zorder=3, edgecolor=PAPER, linewidth=2)
        above = i % 2 == 0
        text_y = 0.76 if above else 0.19
        va = "bottom" if above else "top"
        ax.plot([x, x], [y + (0.02 if above else -0.02), text_y - (0.03 if above else -0.03)], color=color, linewidth=1)
        ax.text(x, text_y, name, ha="center", va=va, fontsize=9, fontweight="bold", color=color)
        ax.text(x, text_y - (0.055 if above else -0.055), body, ha="center", va=va, fontsize=7.2, color=MUTED, linespacing=1.2)
    save(fig, "14_experiment_timeline.png")


def metric_value(path: str, metric: str, value_column: str) -> float:
    frame = pd.read_csv(PROJECT / "outputs" / path)
    return float(frame.loc[frame.metric == metric, value_column].iloc[0])


def figure_model_comparison() -> None:
    metrics = ["sys_acc", "sg_acc", "sg_top5", "el_f1_micro"]
    labels = ["System", "SG top-1", "SG top-5", "Elements F1"]
    combined = pd.read_csv(PROJECT / "outputs/ft_combined_no_replay_control_with_rruff_sg_cv_summary.csv")
    combined = combined[combined.scope == "combined"].set_index("metric")
    real = pd.read_csv(PROJECT / "outputs/real_only_v2_with_rruff_sg_cv_summary.csv").set_index("metric")
    fig = plt.figure(figsize=(9.6, 5.4), dpi=100)
    fig.text(0.06, 0.925, "Final comparison on the SG-enriched real pool", fontsize=18, fontweight="bold")
    fig.text(0.06, 0.875, "Pretraining gives large system/SG gains while element F1 is nearly tied; all values use the same five outer grouped folds.", fontsize=9.5, color=MUTED)
    ax = fig.add_axes([0.08, 0.16, 0.88, 0.62])
    x = np.arange(len(metrics))
    width = 0.25
    zero = [100 * combined.loc[m, "zero_shot"] for m in metrics]
    real_values = [100 * real.loc[m, "real_only"] for m in metrics]
    final = [100 * combined.loc[m, "after_combined_ft"] for m in metrics]
    bars1 = ax.bar(x - width, zero, width, label="zero-shot V2", color="#A8B1BB")
    bars2 = ax.bar(x, real_values, width, label="real-only control", color=BLUE)
    bars3 = ax.bar(x + width, final, width, label="pretrain → combined FT", color=MINT_DARK)
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 82)
    ax.set_ylabel("grouped-CV score (%)")
    ax.legend(ncol=3, loc="upper left", fontsize=8.5)
    for bars in (bars1, bars2, bars3):
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.2, f"{bar.get_height():.1f}", ha="center", fontsize=7.5)
    save(fig, "15_final_model_comparison.png")


def figure_specialists() -> None:
    metrics = ["sys_acc", "sg_acc", "sg_top5", "el_f1_micro"]
    labels = ["System", "SG top-1", "SG top-5", "Elements F1"]
    r = pd.read_csv(PROJECT / "outputs/ft_rruff_only_with_rruff_sg_cv_summary.csv").set_index("metric")
    o = pd.read_csv(PROJECT / "outputs/ft_opxrd_only_cv_summary.csv").set_index("metric")
    fig = plt.figure(figsize=(9.6, 5.4), dpi=100)
    fig.text(0.06, 0.925, "Source-specific models solve different measurement domains", fontsize=18, fontweight="bold")
    fig.text(0.06, 0.875, "Scores are shown within each source's own grouped CV; they are not a common test-set ranking.", fontsize=9.5, color=MUTED)
    ax1 = fig.add_axes([0.08, 0.17, 0.40, 0.60])
    ax2 = fig.add_axes([0.56, 0.17, 0.40, 0.60])
    for ax, table, title, color in [(ax1, r, "RRUFF-only", MINT_DARK), (ax2, o, "opXRD-only", BLUE)]:
        zero = [100 * table.loc[m, "zero_shot"] for m in metrics]
        tuned = [100 * table.loc[m, "after_source_ft"] for m in metrics]
        x = np.arange(len(metrics))
        ax.bar(x - 0.18, zero, 0.36, color="#A8B1BB", label="zero-shot")
        ax.bar(x + 0.18, tuned, 0.36, color=color, label="source FT")
        ax.set_title(title, fontsize=12)
        ax.set_xticks(x, labels, rotation=18, ha="right", fontsize=8)
        ax.set_ylim(0, 90)
        ax.set_ylabel("score (%)")
        ax.legend(fontsize=8)
    save(fig, "16_source_specialists.png")


def figure_cross_source() -> None:
    data = pd.read_csv(PROJECT / "outputs/cross_source_transfer_metrics.csv")
    metrics = ["system_accuracy", "sg_top1", "sg_top5", "elements_micro_f1"]
    labels = ["System", "SG top-1", "SG top-5", "Elements F1"]
    fig, ax = new_figure(
        "Cross-source transfer exposes the real domain shift",
        "Each source specialist is evaluated directly on the other source, without adaptation.",
    )
    x = np.arange(len(metrics))
    width = 0.34
    first = [100 * float(data.iloc[0][m]) for m in metrics]
    second = [100 * float(data.iloc[1][m]) for m in metrics]
    b1 = ax.bar(x - width / 2, first, width, color=MINT_DARK, label="RRUFF-only → opXRD")
    b2 = ax.bar(x + width / 2, second, width, color=BLUE, label="opXRD-only → RRUFF")
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 55)
    ax.set_ylabel("target-domain score (%)")
    ax.legend(fontsize=8.5)
    for bars in (b1, b2):
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1, f"{bar.get_height():.1f}", ha="center", fontsize=8)
    save(fig, "17_cross_source_transfer.png")


def figure_leakage() -> None:
    fig, ax = new_figure(
        "Leakage control through canonical group keys",
        "The split is leakage-free only with respect to this explicit key; the key is useful but not a universal identity function.",
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    box(ax, (0.02, 0.56), 0.38, 0.28, "opXRD group", "phase_compositions + '|' + round(lattice_a, 2)\nkeeps repeated formulations together", LIGHT_BLUE, BLUE, 10.5, 8.5)
    box(ax, (0.02, 0.16), 0.38, 0.28, "RRUFF group", "'rruff_mineral|' + mineral_name.casefold()\nkeeps mineral-name variants together", LIGHT_MINT, MINT_DARK, 10.5, 8.5)
    box(ax, (0.52, 0.56), 0.20, 0.28, "Outer CV", "5-fold GroupKFold\nunbiased evaluation", "#F1EDFB", PURPLE, 10.5, 8.5)
    box(ax, (0.78, 0.56), 0.20, 0.28, "Inner split", "group-aware tuning\ninside train only", LIGHT_AMBER, AMBER, 10.5, 8.5)
    arrow(ax, (0.41, 0.70), (0.51, 0.70), BLUE)
    arrow(ax, (0.73, 0.70), (0.77, 0.70), PURPLE)
    ax.text(0.52, 0.39, "What the key does not guarantee", fontsize=11, fontweight="bold", color=RED)
    limitations = [
        "composition strings are not fully canonicalized",
        "only lattice a is rounded; polymorph identity is imperfect",
        "mineral names are weaker than persistent structure IDs",
        "global cross-source / real-synthetic overlap is not excluded",
    ]
    for i, text in enumerate(limitations):
        ax.text(0.54, 0.31 - i * 0.075, f"• {text}", fontsize=7.8, color=MUTED)
    save(fig, "18_leakage_control.png")


def figure_ablation() -> None:
    summary = pd.read_csv(PROJECT / "outputs/ft_single_phase_ablation_with_rruff_sg_paired_summary.csv")
    subset = summary[(summary.eval_set == "single_phase") & (summary.scope == "combined")]
    metrics = ["sys_acc", "sg_acc", "sg_top5", "el_f1_micro"]
    available = subset[subset.metric.isin(metrics)].set_index("metric")
    fig = plt.figure(figsize=(9.6, 5.4), dpi=100)
    fig.text(0.06, 0.925, "Single-phase-only FT does not replace all-real FT", fontsize=18, fontweight="bold")
    fig.text(0.06, 0.875, "Paired folds and an equal optimizer-step budget isolate the data-selection effect.", fontsize=9.5, color=MUTED)
    ax1 = fig.add_axes([0.08, 0.17, 0.48, 0.60])
    ax2 = fig.add_axes([0.65, 0.17, 0.30, 0.60])
    labels = ["System", "SG top-1", "SG top-5", "Elements F1"]
    if len(available) == len(metrics):
        diffs = [100 * float(available.loc[m, "difference_single_minus_all"]) for m in metrics]
        ax1.barh(labels[::-1], diffs[::-1], color=[MINT_DARK if value >= 0 else RED for value in diffs[::-1]])
        ax1.axvline(0, color=INK, linewidth=1)
        ax1.set_xlabel("single-phase − all-real (percentage points)")
    else:
        ax1.text(0.5, 0.5, "Paired summary not available", transform=ax1.transAxes, ha="center", va="center", color=MUTED)
        ax1.axis("off")
    diagnostic = summary[(summary.eval_set == "multiphase_diagnostic") & (summary.scope == "combined")].set_index("metric")
    diag_metrics = ["sys_acc", "sg_acc", "sg_top5", "el_f1_micro"]
    single = [100 * float(diagnostic.loc[m, "single_phase_ft"]) for m in diag_metrics]
    all_real = [100 * float(diagnostic.loc[m, "all_real_ft"]) for m in diag_metrics]
    x = np.arange(len(diag_metrics))
    ax2.bar(x - 0.18, single, 0.36, color=AMBER, label="single-phase FT")
    ax2.bar(x + 0.18, all_real, 0.36, color=MINT_DARK, label="all-real FT")
    ax2.set_title("Multiphase diagnostic", fontsize=11)
    ax2.set_xticks(x, ["Sys", "SG1", "SG5", "El F1"], fontsize=8)
    ax2.set_ylim(0, 110)
    ax2.set_ylabel("score (%)")
    ax2.legend(fontsize=7.5)
    save(fig, "19_single_phase_ablation.png")


def figure_sg_enrichment() -> None:
    with open(PROJECT / "outputs/rruff_spacegroup_mapping_summary.json", encoding="utf-8") as handle:
        summary = json.load(handle)
    fig = plt.figure(figsize=(9.6, 5.4), dpi=100)
    fig.text(0.06, 0.925, "RRUFF SG enrichment: exact names, explicit ambiguity", fontsize=18, fontweight="bold")
    fig.text(0.06, 0.875, "No fuzzy matching is used; unresolved and ambiguous records remain masked rather than forced into labels.", fontsize=9.5, color=MUTED)
    ax1 = fig.add_axes([0.15, 0.16, 0.39, 0.62])
    ax2 = fig.add_axes([0.63, 0.16, 0.32, 0.62])
    statuses = summary["status_counts"]
    order = [
        "usable_unique_spacegroup",
        "usable_after_crystal_system_filter",
        "matched_ambiguous_spacegroup",
        "matched_unresolved_symbol",
        "matched_no_spacegroup",
        "unmatched_name",
        "matched_crystal_system_conflict",
    ]
    label_map = {
        "usable_unique_spacegroup": "unique SG",
        "usable_after_crystal_system_filter": "resolved by system",
        "matched_ambiguous_spacegroup": "ambiguous SG",
        "matched_unresolved_symbol": "unresolved symbol",
        "matched_no_spacegroup": "no SG in IMA",
        "unmatched_name": "name unmatched",
        "matched_crystal_system_conflict": "system conflict",
    }
    vals = [statuses[k] for k in order]
    colors = [MINT_DARK, MINT, AMBER, RED, "#C9A2A0", "#A8B1BB", PURPLE]
    ax1.barh([label_map[k] for k in order][::-1], vals[::-1], color=colors[::-1])
    ax1.set_xlabel("RRUFF IDs")
    for y, value in enumerate(vals[::-1]):
        ax1.text(value, y, f"  {value:,}", va="center", fontsize=8)
    ax2.axis("off")
    box(ax2, (0.05, 0.62), 0.90, 0.26, "1,105 / 1,359", "usable RRUFF space-group labels\n81.31% coverage", LIGHT_MINT, MINT_DARK, 18, 9)
    box(ax2, (0.05, 0.28), 0.90, 0.24, "93", "represented space groups\nafter enrichment", LIGHT_BLUE, BLUE, 18, 9)
    ax2.text(0.5, 0.12, "1,295 exact + 48 normalized exact-name matches", ha="center", color=MUTED, fontsize=8.5)
    save(fig, "20_rruff_sg_enrichment.png")


def write_manifest() -> None:
    rows = []
    for path in sorted(OUTPUT.glob("*.png")):
        with Image.open(path) as image:
            rows.append({"file": path.name, "width": image.width, "height": image.height, "bytes": path.stat().st_size})
    (OUTPUT / "figure_manifest.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")


def main() -> None:
    figure_pipeline()
    figure_dataset_scale()
    figure_profiles(rruff, "RRUFF", "03_rruff_profiles.png", MINT_DARK)
    figure_acquisition(rruff, "RRUFF", "04_rruff_acquisition.png", MINT_DARK)
    figure_labels(rruff_sg, "RRUFF", "05_rruff_labels.png", MINT_DARK)
    figure_profiles(opxrd, "opXRD", "06_opxrd_profiles.png", BLUE)
    figure_acquisition(opxrd, "opXRD", "07_opxrd_acquisition.png", BLUE)
    figure_labels(opxrd_ft, "opXRD labeled subset", "08_opxrd_labels.png", BLUE)
    figure_target_distributions()
    figure_generator_physics()
    figure_generator_validation()
    figure_domain_shift()
    figure_architecture()
    figure_timeline()
    figure_model_comparison()
    figure_specialists()
    figure_cross_source()
    figure_leakage()
    figure_ablation()
    figure_sg_enrichment()
    write_manifest()
    print(f"Generated {len(list(OUTPUT.glob('*.png')))} figures in {OUTPUT}")


if __name__ == "__main__":
    main()
