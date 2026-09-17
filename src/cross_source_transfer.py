"""Cross-domain evaluation of the two historical source-only FT checkpoints.

The experiment is an out-of-domain stress test, not cross-validation:
each final checkpoint was trained on all labelled rows of its own source and
is evaluated once on the other source. Metrics are calculated only for rows
with the corresponding ground-truth label.

Outputs:
    outputs/cross_source_transfer_metrics.csv
    outputs/cross_source_transfer_metrics.png
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


BASE = Path(__file__).resolve().parents[1]
DATA = BASE / "data" / "preprocessed"
OUT = BASE / "outputs"
CKPT_DIR = BASE / "checkpoints"
GRID_N = 4096
WIDTH = 3
BATCH_SIZE = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AMP = DEVICE.type == "cuda"

EXPERIMENTS = [
    {
        "model": "RRUFF-only (SG-enriched)",
        "training_domain": "RRUFF",
        "target_domain": "opXRD",
        "checkpoint": CKPT_DIR / "ft_rruff_only_with_rruff_sg_final.pt",
        "config": OUT / "ft_rruff_only_with_rruff_sg_final_config.json",
        "target_pool": BASE / "data" / "clean" / "ft_pool_opxrd.parquet",
    },
    {
        "model": "opXRD-only",
        "training_domain": "opXRD",
        "target_domain": "RRUFF",
        "checkpoint": CKPT_DIR / "ft_opxrd_only_final.pt",
        "config": OUT / "ft_opxrd_only_final_config.json",
        "target_pool": BASE / "data" / "clean" / "ft_pool_rruff_with_rruff_sg.parquet",
    },
]


def to_list(value):
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, list) else []
        except Exception:
            return []
    if isinstance(value, (list, tuple, np.ndarray)):
        return [item for item in value if isinstance(item, str)]
    return []


def prepare_target_frame(path, row_of, systems):
    frame = pd.read_parquet(path).copy()
    frame["row_idx"] = frame["sample_id"].map(row_of)
    assert frame["row_idx"].notna().all(), f"Missing preprocessed rows: {path.name}"
    frame["primary_wavelength"] = frame["primary_wavelength"].fillna(1.5406)
    if "secondary_wavelength" not in frame:
        frame["secondary_wavelength"] = np.nan
    frame["crystal_system"] = frame["crystal_system"].replace({"rhombohedral": "trigonal"})
    frame["elements"] = frame["elements_list"].apply(to_list)

    abc = frame[["lattice_a", "lattice_b", "lattice_c"]].to_numpy(np.float64)
    angles = frame[["alpha", "beta", "gamma"]].to_numpy(np.float64)
    cosines = [np.cos(np.radians(angles[:, i])) for i in range(3)]
    ca, cb, cg = cosines
    volume_term = 1.0 - ca**2 - cb**2 - cg**2 + 2 * ca * cb * cg
    frame["volume"] = abc.prod(axis=1) * np.sqrt(np.clip(volume_term, 1e-12, None))
    frame["lat6"] = list(np.hstack([np.log(abc), angles]))
    frame["has_system_label"] = frame["crystal_system"].isin(systems)
    return frame.reset_index(drop=True)


def build_labels(frame, vocab, element_index, systems, lat_mean, lat_std, vol_mean, vol_std):
    lam1 = frame["primary_wavelength"].to_numpy(np.float32)
    lam2 = frame["secondary_wavelength"].to_numpy(np.float32)
    lam = np.stack(
        [lam1 / 1.54, np.nan_to_num(lam2) / 1.54, np.isfinite(lam2).astype(np.float32)],
        axis=1,
    )

    lat6 = np.stack(frame["lat6"].to_numpy())
    has_lat = ~np.isnan(lat6).any(axis=1)
    lat_mask = has_lat.astype(np.float32)
    lat6 = (lat6 - lat_mean) / lat_std

    volume_raw = frame["volume"].to_numpy(np.float64)
    volume = (np.log(volume_raw) - vol_mean) / vol_std

    sg_raw = frame["spacegroup_number"].to_numpy(float)
    sg_mask = np.isfinite(sg_raw).astype(np.float32)
    sg = np.nan_to_num(sg_raw, nan=1.0).astype(np.int64) - 1

    system_index = {system: i for i, system in enumerate(systems)}
    system_raw = frame["crystal_system"].map(system_index)
    system_mask = system_raw.notna().to_numpy(np.float32)
    system = system_raw.fillna(0).to_numpy(np.int64)

    elements = np.zeros((len(frame), len(vocab)), np.float32)
    element_mask = np.zeros(len(frame), np.float32)
    for row, item_list in enumerate(frame["elements"]):
        if item_list:
            element_mask[row] = 1.0
            for element in item_list:
                index = element_index.get(element)
                if index is not None:
                    elements[row, index] = 1.0

    return {
        "lam": lam,
        "lat6": lat6.astype(np.float32),
        "lat_mask": lat_mask,
        "volume": volume.astype(np.float32),
        "volume_mask": lat_mask.copy(),
        "sg": sg,
        "sg_mask": sg_mask,
        "system": system,
        "system_mask": system_mask,
        "elements": elements,
        "element_mask": element_mask,
    }


class SpectrumDataset(Dataset):
    def __init__(self, frame, x_memmap, mask_memmap, label_kwargs):
        self.rows = frame["row_idx"].to_numpy(np.int64)
        self.x_memmap = x_memmap
        self.mask_memmap = mask_memmap
        self.labels = build_labels(frame, **label_kwargs)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        x = np.empty((2, GRID_N), np.float32)
        x[0] = self.x_memmap[self.rows[index]]
        x[1] = self.mask_memmap[self.rows[index]]
        labels = self.labels
        return (
            torch.from_numpy(x),
            torch.from_numpy(labels["lam"][index]),
            torch.from_numpy(labels["lat6"][index]),
            torch.tensor(labels["lat_mask"][index]),
            torch.tensor(labels["sg"][index]),
            torch.tensor(labels["sg_mask"][index]),
            torch.tensor(labels["system"][index]),
            torch.tensor(labels["system_mask"][index]),
            torch.from_numpy(labels["elements"][index]),
            torch.tensor(labels["element_mask"][index]),
            torch.tensor(labels["volume"][index]),
            torch.tensor(labels["volume_mask"][index]),
        )


class ResBlock(nn.Module):
    def __init__(self, channels_in, channels_out, stride=1):
        super().__init__()
        self.conv1 = nn.Conv1d(channels_in, channels_out, 3, stride=stride, padding=1, bias=False)
        self.n1 = nn.GroupNorm(8, channels_out)
        self.conv2 = nn.Conv1d(channels_out, channels_out, 3, padding=1, bias=False)
        self.n2 = nn.GroupNorm(8, channels_out)
        if channels_in == channels_out and stride == 1:
            self.skip = nn.Identity()
        else:
            self.skip = nn.Sequential(
                nn.Conv1d(channels_in, channels_out, 1, stride=stride, bias=False),
                nn.GroupNorm(8, channels_out),
            )

    def forward(self, x):
        hidden = F.gelu(self.n1(self.conv1(x)))
        hidden = self.n2(self.conv2(hidden))
        return F.gelu(hidden + self.skip(x))


class XRDNetV2(nn.Module):
    def __init__(self, n_elements, width=WIDTH):
        super().__init__()
        channels = [32 * width, 48 * width, 64 * width, 96 * width, 128 * width, 192 * width, 256 * width]
        self.stem = nn.Sequential(
            nn.Conv1d(2, channels[0], 15, padding=7, bias=False),
            nn.GroupNorm(8, channels[0]),
            nn.GELU(),
        )
        self.blocks = nn.Sequential(*[ResBlock(channels[i], channels[i + 1], stride=2) for i in range(6)])
        self.lam_mlp = nn.Sequential(nn.Linear(3, 16 * width), nn.GELU(), nn.Linear(16 * width, 16 * width))
        self.trunk = nn.Sequential(
            nn.Linear(channels[-1] + 16 * width, 512 * width),
            nn.GELU(),
            nn.Linear(512 * width, 512 * width),
            nn.GELU(),
        )
        self.head_lat = nn.Linear(512 * width, 6)
        self.head_vol = nn.Linear(512 * width, 1)
        self.head_sg = nn.Linear(512 * width, 230)
        self.head_sys = nn.Linear(512 * width, 7)
        self.head_el = nn.Linear(512 * width, n_elements)

    def forward(self, x, lam):
        features = self.blocks(self.stem(x))
        mask = F.adaptive_avg_pool1d(x[:, 1:2], features.shape[-1]).clamp_min(1e-3)
        pooled = (features * mask).sum(-1) / mask.sum(-1)
        latent = self.trunk(torch.cat([pooled, self.lam_mlp(lam)], dim=1))
        return {
            "lat": self.head_lat(latent),
            "vol": self.head_vol(latent).squeeze(-1),
            "sg": self.head_sg(latent),
            "system": self.head_sys(latent),
            "elements": self.head_el(latent),
        }


@torch.no_grad()
def evaluate(model, loader, threshold, lat_mean, lat_std, vol_mean, vol_std):
    model.eval()
    records = []
    for batch in loader:
        batch = [value.to(DEVICE, non_blocking=AMP) for value in batch]
        with torch.autocast(device_type=DEVICE.type, dtype=torch.float16, enabled=AMP):
            output = model(batch[0], batch[1])
        _, _, lat, lat_mask, sg, sg_mask, system, system_mask, elements, element_mask, volume, _ = batch

        lat_pred = output["lat"].float().cpu().numpy() * lat_std + lat_mean
        lat_true = lat.float().cpu().numpy() * lat_std + lat_mean
        lat_pred[:, :3] = np.exp(lat_pred[:, :3])
        lat_true[:, :3] = np.exp(lat_true[:, :3])
        volume_pred = np.exp(output["vol"].float().cpu().numpy() * vol_std + vol_mean)
        volume_true = np.exp(volume.float().cpu().numpy() * vol_std + vol_mean)
        element_probability = torch.sigmoid(output["elements"]).float().cpu().numpy()
        element_true = elements.float().cpu().numpy()

        for i in range(len(lat)):
            has_lattice = bool(lat_mask[i].item() > 0)
            predicted_elements = frozenset(np.where(element_probability[i] > threshold)[0])
            true_elements = frozenset(np.where(element_true[i] > 0.5)[0]) if element_mask[i] > 0 else frozenset()
            errors = np.abs(lat_pred[i, :3] - lat_true[i, :3]) if has_lattice else np.full(3, np.nan)
            records.append(
                {
                    "lat_mask": float(lat_mask[i]),
                    "sg_mask": float(sg_mask[i]),
                    "system_mask": float(system_mask[i]),
                    "element_mask": float(element_mask[i]),
                    "system_ok": float(system_mask[i] > 0 and output["system"][i].argmax().item() == system[i].item()),
                    "sg_ok": float(sg_mask[i] > 0 and output["sg"][i].argmax().item() == sg[i].item()),
                    "sg_top5": float(sg_mask[i] > 0 and sg[i].item() in output["sg"][i].topk(5).indices.tolist()),
                    "mae_a": errors[0],
                    "mae_b": errors[1],
                    "mae_c": errors[2],
                    "mae_angle": float(np.abs(lat_pred[i, 3:] - lat_true[i, 3:]).mean()) if has_lattice else np.nan,
                    "mae_volume": abs(volume_pred[i] - volume_true[i]) if has_lattice else np.nan,
                    "predicted_elements": predicted_elements,
                    "true_elements": true_elements,
                }
            )

    frame = pd.DataFrame(records)
    metrics = {"n_rows": len(frame), "elements_threshold": threshold}
    for mask, metrics_to_mean in [
        ("system_mask", {"system_n": None, "system_accuracy": "system_ok"}),
        ("sg_mask", {"sg_n": None, "sg_top1": "sg_ok", "sg_top5": "sg_top5"}),
        ("lat_mask", {"lattice_n": None, "mae_a": "mae_a", "mae_b": "mae_b", "mae_c": "mae_c", "mae_angle": "mae_angle", "mae_volume": "mae_volume"}),
    ]:
        valid = frame[frame[mask] > 0]
        for name, column in metrics_to_mean.items():
            metrics[name] = len(valid) if column is None else (valid[column].mean() if len(valid) else np.nan)

    valid_elements = frame[frame["element_mask"] > 0]
    metrics["elements_n"] = len(valid_elements)
    if len(valid_elements):
        true_positive = sum(len(row.predicted_elements & row.true_elements) for row in valid_elements.itertuples())
        false_positive = sum(len(row.predicted_elements - row.true_elements) for row in valid_elements.itertuples())
        false_negative = sum(len(row.true_elements - row.predicted_elements) for row in valid_elements.itertuples())
        precision = true_positive / max(true_positive + false_positive, 1)
        recall = true_positive / max(true_positive + false_negative, 1)
        metrics["elements_micro_f1"] = 2 * precision * recall / max(precision + recall, 1e-9)
        metrics["elements_exact"] = float((valid_elements["predicted_elements"] == valid_elements["true_elements"]).mean())
    else:
        metrics["elements_micro_f1"] = np.nan
        metrics["elements_exact"] = np.nan
    return metrics


def plot_metrics(results):
    display = results.set_index("model_to_target")
    metrics = ["system_accuracy", "sg_top1", "sg_top5", "elements_micro_f1"]
    labels = ["System accuracy", "SG top-1", "SG top-5", "Elements micro-F1"]
    fig, axis = plt.subplots(figsize=(10, 4.8))
    x = np.arange(len(metrics))
    width = 0.34
    for offset, (_, row) in zip([-width / 2, width / 2], display.iterrows()):
        values = [row[name] if np.isfinite(row[name]) else np.nan for name in metrics]
        bars = axis.bar(x + offset, values, width, label=row.name)
        for bar, value in zip(bars, values):
            if np.isfinite(value):
                axis.text(bar.get_x() + bar.get_width() / 2, value + 0.02, f"{value:.3f}", ha="center", fontsize=8)
            else:
                axis.text(bar.get_x() + bar.get_width() / 2, 0.02, "N/A", ha="center", fontsize=8, color="#555555")
    axis.set_xticks(x, labels)
    axis.set_ylim(0, 1.05)
    axis.set_ylabel("Score")
    axis.set_title("Cross-domain transfer: source-only checkpoints")
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "cross_source_transfer_metrics.png", dpi=160)
    plt.close(fig)


def main():
    stats = json.loads((OUT / "pretrain_stats.json").read_text(encoding="utf-8"))
    vocab = stats["vocab"]
    systems = stats["systems"]
    label_kwargs = {
        "vocab": vocab,
        "element_index": {element: index for index, element in enumerate(vocab)},
        "systems": systems,
        "lat_mean": np.asarray(stats["lat_mean"]),
        "lat_std": np.asarray(stats["lat_std"]),
        "vol_mean": stats["vol_mean"],
        "vol_std": stats["vol_std"],
    }
    index = pd.read_parquet(DATA / "index_preprocessed.parquet")
    row_of = dict(zip(index["sample_id"], index["row_idx"]))
    x_memmap = np.memmap(DATA / "X_intensity.f16", dtype=np.float16, mode="r", shape=(len(index), GRID_N))
    mask_memmap = np.memmap(DATA / "M_mask.u8", dtype=np.uint8, mode="r", shape=(len(index), GRID_N))

    print(f"Device: {DEVICE.type}")
    result_rows = []
    for experiment in EXPERIMENTS:
        for path in (experiment["checkpoint"], experiment["config"], experiment["target_pool"]):
            if not path.exists():
                raise FileNotFoundError(path)
        config = json.loads(experiment["config"].read_text(encoding="utf-8"))
        target = prepare_target_frame(experiment["target_pool"], row_of, systems)
        dataset = SpectrumDataset(target, x_memmap, mask_memmap, label_kwargs)
        loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=AMP, num_workers=0)

        model = XRDNetV2(len(vocab)).to(DEVICE)
        state = torch.load(experiment["checkpoint"], map_location=DEVICE, weights_only=True)
        model.load_state_dict(state)
        metrics = evaluate(
            model,
            loader,
            threshold=float(config["elements_threshold"]),
            lat_mean=label_kwargs["lat_mean"],
            lat_std=label_kwargs["lat_std"],
            vol_mean=label_kwargs["vol_mean"],
            vol_std=label_kwargs["vol_std"],
        )
        result_rows.append(
            {
                "model_to_target": f"{experiment['model']} -> {experiment['target_domain']}",
                "model": experiment["model"],
                "training_domain": experiment["training_domain"],
                "target_domain": experiment["target_domain"],
                "checkpoint": experiment["checkpoint"].name,
                **metrics,
            }
        )
        print(f"{experiment['model']} -> {experiment['target_domain']}: {metrics}")

    results = pd.DataFrame(result_rows)
    results.to_csv(OUT / "cross_source_transfer_metrics.csv", index=False)
    plot_metrics(results)
    print("Saved: outputs/cross_source_transfer_metrics.csv")
    print("Saved: outputs/cross_source_transfer_metrics.png")


if __name__ == "__main__":
    main()
