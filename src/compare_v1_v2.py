"""
Финальное сравнение v1 vs v2 на ОДНОМ и том же синтетическом вале (9 237 строк,
те же сплиты) одним оценщиком. Плюс zero-shot обоих пре-трейнов на реальном FT-пуле.
"""

import json
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
DEVICE = torch.device("cuda")
GRID_N = 4096

stats = json.loads((OUT / "pretrain_stats.json").read_text())
VOCAB = stats["vocab"]
EL_IDX = {e: i for i, e in enumerate(VOCAB)}
LAT_MEAN = np.array(stats["lat_mean"])
LAT_STD = np.array(stats["lat_std"])
VOL_MEAN, VOL_STD = stats["vol_mean"], stats["vol_std"]
SYSTEMS = stats["systems"]

index = pd.read_parquet(DATA / "index_preprocessed.parquet")
N_TOTAL = len(index)
X_MM = np.memmap(DATA / "X_intensity.f16", dtype=np.float16, mode="r", shape=(N_TOTAL, GRID_N))
M_MM = np.memmap(DATA / "M_mask.u8", dtype=np.uint8, mode="r", shape=(N_TOTAL, GRID_N))
row_of = dict(zip(index["sample_id"], index["row_idx"]))


def to_list(v):

    if isinstance(v, str):
        try:
            return json.loads(v)
        
        except Exception:
            return []
        
    if isinstance(v, (list, tuple, np.ndarray)):
        return [e for e in v if isinstance(e, str)]
    
    return []


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


class Net(nn.Module):
    def __init__(self, n_el, w):
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
        z = self.trunk(torch.cat([pooled, self.lam_mlp(lam)], 1))

        return dict(lat=self.head_lat(z), vol=self.head_vol(z).squeeze(-1),
                    sg=self.head_sg(z), sys=self.head_sys(z), el=self.head_el(z))


class DS(Dataset):
    def __init__(self, frame):
        self.row = frame["row_idx"].to_numpy(np.int64)
        lam1 = frame["lambda_1"].to_numpy(np.float32)
        lam2 = frame["lambda_2"].to_numpy(np.float32) if "lambda_2" in frame \
            else frame["secondary_wavelength"].to_numpy(np.float32)
        self.lam = np.stack([lam1 / 1.54, np.nan_to_num(lam2) / 1.54,
                             np.isfinite(lam2).astype(np.float32)], 1)
        lat6 = np.stack(frame["lat6"].to_numpy())
        self.latm = (~np.isnan(lat6).any(1)).astype(np.float32)
        self.lat6 = (lat6 - LAT_MEAN) / LAT_STD
        self.vol = (frame["logV"].to_numpy(np.float32) - VOL_MEAN) / VOL_STD
        sg = frame["spacegroup_number"].to_numpy(float)
        self.sgm = np.isfinite(sg)
        self.sg = np.nan_to_num(sg).astype(np.int64) - 1
        sysmap = {s: i for i, s in enumerate(SYSTEMS)}
        sys_raw = frame["crystal_system"].map(sysmap)
        self.sysm = sys_raw.notna().to_numpy(np.float32)
        self.sys = sys_raw.fillna(0).to_numpy(np.int64)
        n = len(frame)
        self.el = np.zeros((n, len(VOCAB)), np.float32)
        self.elm = np.zeros(n, np.float32)

        for i, els in enumerate(frame["elements"]):
            if len(els):
                self.elm[i] = 1
                for e in els:
                    j = EL_IDX.get(e)
                    if j is not None:
                        self.el[i, j] = 1.0

    def __len__(self):
        return len(self.row)

    def __getitem__(self, i):
        x = np.empty((2, GRID_N), np.float32)
        x[0] = X_MM[self.row[i]]
        x[1] = M_MM[self.row[i]]

        return (torch.from_numpy(x), torch.from_numpy(self.lam[i]),
                torch.from_numpy(self.lat6[i].astype(np.float32)),
                torch.tensor(self.latm[i]), torch.tensor(self.sg[i]),
                torch.tensor(self.sgm[i] * (self.sgm[i] > 0)),  # sgm как float
                torch.tensor(self.sys[i]), torch.tensor(self.sysm[i]),
                torch.from_numpy(self.el[i]), torch.tensor(self.elm[i]),
                torch.tensor(self.vol[i]), torch.tensor(self.latm[i]))


@torch.no_grad()

def evaluate(model, loader):
    model.eval()
    sg_ok = sg_n = sys_ok = sys_n = 0
    tp = fp = fn = 0
    lat_abs = np.zeros(6)
    lat_n = 0

    for x, lam, lat, latm, sg, sgm, sys_, sysm, el, elm, vol, volm in loader:
        x, lam = x.to(DEVICE), lam.to(DEVICE)

        with torch.autocast("cuda", dtype=torch.float16):
            out = model(x, lam)
        sg, sgm, sys_, sysm = sg.cuda(), sgm.cuda(), sys_.cuda(), sysm.cuda()
        el = el.cuda()
        m = sgm > 0

        if m.any():
            sg_ok += (out["sg"][m].argmax(1) == sg[m].long()).sum().item()
            sg_n += int(m.sum())
        m = sysm > 0

        if m.any():
            sys_ok += (out["sys"][m].argmax(1) == sys_[m].long()).sum().item()
            sys_n += int(m.sum())

        pred = (out["el"] > 0).float()
        tp += float((pred * el).sum().item())
        fp += float((pred * (1 - el)).sum().item())
        fn += float(((1 - pred) * el).sum().item())
        m = latm > 0

        if m.any():
            p = out["lat"][m].float().cpu().numpy() * LAT_STD + LAT_MEAN
            t = lat[m].numpy() * LAT_STD + LAT_MEAN
            p[:, :3] = np.exp(p[:, :3])
            t[:, :3] = np.exp(t[:, :3])
            lat_abs += np.abs(p - t).sum(0)
            lat_n += int(m.sum())

    res = dict(
        sg_acc=sg_ok / max(sg_n, 1), sg_n=sg_n,
        sys_acc=sys_ok / max(sys_n, 1),
        el_f1=2 * (tp / max(tp + fp, 1)) * (tp / max(tp + fn, 1))
        / max(tp / max(tp + fp, 1) + tp / max(tp + fn, 1), 1e-9),
    )

    if lat_n:
        mae = lat_abs / lat_n
        res.update(mae_a=mae[0], mae_b=mae[1], mae_c=mae[2], mae_ang=mae[3:].mean(),
                   lat_n=lat_n)
        
    return res


# ---------------- синтетический вал (тот же сплит) ----------------

splits = pd.read_parquet(OUT / "splits_pretrain.parquet")
syn = pd.read_parquet(BASE / "data" / "clean" / "df_synth_summary_final_clean.parquet")
pre = index[index["split_role"] == "pretrain"][["sample_id", "row_idx", "lambda_1", "lambda_2"]]
df = pre.merge(syn, on="sample_id").merge(splits, on="sample_id")
val_df = df[df["split"] == "val"].reset_index(drop=True)

abc = val_df[["lattice_a", "lattice_b", "lattice_c"]].to_numpy(np.float64)
ang = val_df[["alpha", "beta", "gamma"]].to_numpy(np.float64)
ca, cb, cg = (np.cos(np.radians(ang[:, i])) for i in range(3))
t = 1.0 - ca**2 - cb**2 - cg**2 + 2 * ca * cb * cg
val_df["V"] = abc.prod(1) * np.sqrt(np.clip(t, 1e-12, None))
val_df["lat6"] = list(np.hstack([np.log(abc), ang]))
val_df["logV"] = np.log(val_df["V"])
val_df["elements"] = val_df["elements_list"].apply(to_list)

loader = DataLoader(DS(val_df), batch_size=192, shuffle=False, pin_memory=True)

for tag, ckpt, w in [
    ("v1 (1.37M)", "pretrain_v1_full_best.pt", 1),
    ("v2 (11.3M)", "pretrain_v2_full_best.pt", 3),
]:
    model = Net(len(VOCAB), w).to(DEVICE)
    model.load_state_dict(torch.load(BASE / "checkpoints" / ckpt,
                                     map_location=DEVICE, weights_only=True))
    m = evaluate(model, loader)
    print(f"=== пре-трейн {tag} | синтетический вал n={len(val_df)} ===")
    print(f"  SG top-1:     {m['sg_acc']*100:5.1f}%  (n={m['sg_n']})")
    print(f"  crystal sys:  {m['sys_acc']*100:5.1f}%")
    print(f"  elements F1:  {m['el_f1']*100:5.1f}%")
    print(f"  MAE a/b/c:    {m['mae_a']:.2f}/{m['mae_b']:.2f}/{m['mae_c']:.2f} A | углы {m['mae_ang']:.2f} deg")
    print()
    del model
    torch.cuda.empty_cache()
