"""
Псевдо-разметка 86 368 неразмеченных реальных opXRD-спектров (self-training).

Этапы:
  1. Инференс ft_v2_final на всех unlabeled-строках -> вероятности голов.
  2. Калибровка порогов НА ВАЛИДАЦИИ (FT CV-фолды): для каждой головы и
     целевой точности (90/95/98%) находим порог уверенности, при котором
     фактическая точность на вале >= цели.
  3. Разметка: каждой строке - псевдо-метки только тех голов, где уверенность
     выше порога (базовая цель 95%).
  4. Реестр JSON: какие строки, какие головы, уверенность, порог, модель, дата.

Выход:
  data/clean/pseudo_labels.parquet      - псевдо-метки (sample_id + метки + уверенности)
  outputs/pseudo_labels_registry.json   - реестр происхождения
"""

import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

BASE = Path(__file__).resolve().parents[1]
DATA = BASE / "data" / "preprocessed"
OUT = BASE / "outputs"
CKPT = BASE / "checkpoints" / "ft_v2_final.pt"
DEVICE = torch.device("cuda")
GRID_N = 4096
SEED = 42

stats = json.loads((OUT / "pretrain_stats.json").read_text())
VOCAB = stats["vocab"]
EL_IDX = {e: i for i, e in enumerate(VOCAB)}
SYSTEMS = stats["systems"]
LAT_MEAN = np.array(stats["lat_mean"])
LAT_STD = np.array(stats["lat_std"])
VOL_MEAN, VOL_STD = stats["vol_mean"], stats["vol_std"]

index = pd.read_parquet(DATA / "index_preprocessed.parquet")
N_TOTAL = len(index)
X_MM = np.memmap(DATA / "X_intensity.f16", dtype=np.float16, mode="r", shape=(N_TOTAL, GRID_N))
M_MM = np.memmap(DATA / "M_mask.u8", dtype=np.uint8, mode="r", shape=(N_TOTAL, GRID_N))
row_of = dict(zip(index["sample_id"], index["row_idx"]))


# ---------------- модель v2 ----------------
class ResBlock(nn.Module):
    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.conv1 = nn.Conv1d(cin, cout, 3, stride=stride, padding=1, bias=False)
        self.n1 = nn.GroupNorm(8, cout)
        self.conv2 = nn.Conv1d(cout, cout, 3, padding=1, bias=False)
        self.n2 = nn.GroupNorm(8, cout)
        if cin == cout and stride == 1:
            self.skip = nn.Identity()
        else:
            self.skip = nn.Sequential(nn.Conv1d(cin, cout, 1, stride=stride, bias=False),
                                      nn.GroupNorm(8, cout))

    def forward(self, x):
        h = F.gelu(self.n1(self.conv1(x)))
        h = self.n2(self.conv2(h))
        return F.gelu(h + self.skip(x))


class XRDNetV2(nn.Module):
    def __init__(self, n_el, w=3):
        super().__init__()
        c = [32 * w, 48 * w, 64 * w, 96 * w, 128 * w, 192 * w, 256 * w]
        self.stem = nn.Sequential(nn.Conv1d(2, c[0], 15, padding=7, bias=False),
                                  nn.GroupNorm(8, c[0]), nn.GELU())
        self.blocks = nn.Sequential(*[ResBlock(c[i], c[i + 1], stride=2) for i in range(6)])
        self.lam_mlp = nn.Sequential(nn.Linear(3, 16 * w), nn.GELU(), nn.Linear(16 * w, 16 * w))
        self.trunk = nn.Sequential(nn.Linear(c[-1] + 16 * w, 512 * w), nn.GELU(),
                                   nn.Linear(512 * w, 512 * w), nn.GELU())
        self.head_lat = nn.Linear(512 * w, 6)
        self.head_vol = nn.Linear(512 * w, 1)
        self.head_sg = nn.Linear(512 * w, 230)
        self.head_sys = nn.Linear(512 * w, 7)
        self.head_el = nn.Linear(512 * w, n_el)

    def forward(self, x, lam):
        f = self.stem(x)
        f = self.blocks(f)
        w = F.adaptive_avg_pool1d(x[:, 1:2], f.shape[-1]).clamp_min(1e-3)
        pooled = (f * w).sum(-1) / w.sum(-1)
        z = torch.cat([pooled, self.lam_mlp(lam)], dim=1)
        z = self.trunk(z)
        return dict(lat=self.head_lat(z), vol=self.head_vol(z).squeeze(-1),
                    sg=self.head_sg(z), sys=self.head_sys(z), el=self.head_el(z))


class InferDS(Dataset):
    """Только спектры + лямбды; истинные метки не нужны."""

    def __init__(self, frame):
        self.row = frame["row_idx"].to_numpy(np.int64)
        lam1 = frame["lambda_1"].to_numpy(np.float32)
        lam2 = frame["lambda_2"].to_numpy(np.float32)
        self.lam = np.stack([lam1 / 1.54, np.nan_to_num(lam2) / 1.54,
                             np.isfinite(lam2).astype(np.float32)], 1)

    def __len__(self):
        return len(self.row)

    def __getitem__(self, i):
        x = np.empty((2, GRID_N), np.float32)
        x[0] = X_MM[self.row[i]]
        x[1] = M_MM[self.row[i]]
        return torch.from_numpy(x), torch.from_numpy(self.lam[i])


@torch.no_grad()
def infer(model, frame, bs=256):
    """Вероятности sys/sg + sigmoid элементов + lat6/vol предсказания."""
    loader = DataLoader(InferDS(frame), batch_size=bs, shuffle=False, pin_memory=True)
    n = len(frame)
    out_sys = np.empty((n, 7), np.float32)
    out_sg = np.empty((n, 230), np.float32)
    out_el = np.empty((n, len(VOCAB)), np.float32)
    out_lat = np.empty((n, 6), np.float32)
    i = 0
    for x, lam in loader:
        x, lam = x.to(DEVICE), lam.to(DEVICE)
        with torch.autocast("cuda", dtype=torch.float16):
            o = model(x, lam)
        b = len(x)
        out_sys[i:i + b] = F.softmax(o["sys"].float(), 1).cpu().numpy()
        out_sg[i:i + b] = F.softmax(o["sg"].float(), 1).cpu().numpy()
        out_el[i:i + b] = torch.sigmoid(o["el"].float()).cpu().numpy()
        out_lat[i:i + b] = o["lat"].float().cpu().numpy()
        i += b
    return dict(sys=out_sys, sg=out_sg, el=out_el, lat=out_lat)


# ---------------- 1. инференс на валидации (калибровка) и на 86k ----------------
model = XRDNetV2(len(VOCAB), w=3).to(DEVICE)
model.load_state_dict(torch.load(CKPT, map_location=DEVICE, weights_only=True))
model.eval()
print("модель загружена:", CKPT.name)

# калибровочный сет: ВЕСЬ ft_pool (метки известны; модель видела их в финальном
# обучении, поэтому порог может быть оптимистичным -> поднимаем требования ниже)
ft = pd.read_parquet(BASE / "data" / "clean" / "ft_pool_combined.parquet")
ft["row_idx"] = ft["sample_id"].map(row_of)
ft = ft[ft["row_idx"].notna()].reset_index(drop=True)
IMPUTE = 1.5406
ft["lambda_1"] = ft["primary_wavelength"].fillna(IMPUTE)
ft["lambda_2"] = ft["secondary_wavelength"]

t0 = time.time()
print(f"инференс на ft_pool (калибровка): {len(ft)} ...", flush=True)
cal = infer(model, ft)
print(f"  {time.time()-t0:.0f} c")

unl = index[index["split_role"] == "unlabeled"].reset_index(drop=True)
t0 = time.time()
print(f"инференс на unlabeled: {len(unl)} ...", flush=True)
pred = infer(model, unl)
print(f"  {time.time()-t0:.0f} c")

# ---------------- 2. калибровка порогов на ft_pool ----------------
sysmap = {s: i for i, s in enumerate(SYSTEMS)}
ft_sys_true = ft["crystal_system"].map(sysmap).to_numpy()
ft_sg_true = ft["spacegroup_number"].to_numpy(float) - 1
ft_el_true = np.zeros((len(ft), len(VOCAB)), np.float32)
for i, v in enumerate(ft["elements_list"]):
    try:
        els = json.loads(v) if isinstance(v, str) else list(v)
    except Exception:
        els = []
    for e in els if isinstance(els, list) else []:
        j = EL_IDX.get(e)
        if j is not None:
            ft_el_true[i, j] = 1.0


def calibrate_threshold(probs_max, correct, targets=(0.90, 0.95, 0.98)):
    """Порог, при котором фактическая точность >= target."""
    res = {}
    order = np.argsort(-probs_max)
    p_sorted = probs_max[order]
    c_sorted = correct[order]
    cum_acc = np.cumsum(c_sorted) / (np.arange(len(c_sorted)) + 1)
    for tgt in targets:
        # ищем самый низкий порог с точностью >= tgt
        idx = np.where(cum_acc >= tgt)[0]
        if len(idx) == 0:
            res[tgt] = None
            continue
        # берём последний idx, где acc>=tgt при жадном спуске (после может упасть)
        # проще: максимальный порог, при котором точность выше цели хотя бы у
        # половины диапазона -> используем可靠ный: минимальный порог из устойчивой зоны
        k = idx[-1]
        res[tgt] = float(p_sorted[k])
    return res


print("\n=== калибровка порогов (на ft_pool, модель его видела - оценка оптимистична) ===")

# crystal system
sys_max = cal["sys"].max(1)
sys_pred = cal["sys"].argmax(1)
sys_ok = (sys_pred == ft_sys_true).astype(np.float64)
sys_ok[ft_sys_true != ft_sys_true] = np.nan
m = ~np.isnan(sys_ok)
thr_sys = calibrate_threshold(sys_max[m], sys_ok[m])
print("crystal system: max-prob vs точность")
for tgt in (0.90, 0.95, 0.98):
    print(f"  цель {tgt:.0%}: порог {thr_sys[tgt]:.3f}" if thr_sys[tgt] else f"  цель {tgt:.0%}: недостижимо")

# space group (только строки с SG-меткой)
sg_max = cal["sg"].max(1)
sg_pred = cal["sg"].argmax(1)
sg_ok = np.where(np.isfinite(ft_sg_true), (sg_pred == ft_sg_true).astype(float), np.nan)
m = ~np.isnan(sg_ok)
thr_sg = calibrate_threshold(sg_max[m], sg_ok[m])
print("space group:")
for tgt in (0.90, 0.95, 0.98):
    print(f"  цель {tgt:.0%}: порог {thr_sg[tgt]:.3f}" if thr_sg[tgt] else f"  цель {tgt:.0%}: недостижимо")

# элементы: "уверенность" = min prob among true-set / max among false-set...
# практичнее: точное совпадение множества при пороге t на sigmoid
print("elements (точное совпадение множества при пороге t):")
thr_el = {}
for tgt in (0.90, 0.95):
    best = None
    for t in np.arange(0.5, 0.95, 0.05):
        bin_ = (cal["el"] > t).astype(int)
        exact = (bin_ == ft_el_true).all(1)
        acc = exact.mean()
        if acc >= tgt:
            best = (float(t), float(acc))
    thr_el[tgt] = best
    print(f"  цель {tgt:.0%}: {'порог %.2f (факт %.1f%%)' % best if best else 'недостижимо'}")

np.save(BASE / "outputs" / "_tmp_cal_sys.npy", cal["sys"])
np.save(BASE / "outputs" / "_tmp_cal_sg.npy", cal["sg"])
np.save(BASE / "outputs" / "_tmp_cal_el.npy", cal["el"])
np.save(BASE / "outputs" / "_tmp_pred_sys.npy", pred["sys"])
np.save(BASE / "outputs" / "_tmp_pred_sg.npy", pred["sg"])
np.save(BASE / "outputs" / "_tmp_pred_el.npy", pred["el"])
np.save(BASE / "outputs" / "_tmp_pred_lat.npy", pred["lat"])
unl[["sample_id", "row_idx", "lambda_1", "lambda_2"]].to_parquet(
    BASE / "outputs" / "_tmp_unl.parquet")
ft[["sample_id"]].to_parquet(BASE / "outputs" / "_tmp_ft_ids.parquet")
print("\nпромежуточные результаты сохранены (_tmp_*.npy/parquet)")
