"""
Диагностика пре-трейна: что модель выучила, а где уперлась в сложность задачи.

Ключевые вопросы:
  1. Разбивка по кристаллическим системам: кубическая решётка читается из
     положений пиков (a = lambda*d*sqrt(h^2+k^2+l^2)),
     триклинная - это задача индексирования, исторически сложная.
     Если кубик предсказан хорошо, а триклин плохо - это свойство задачи.
  2. In-distribution разрыв: те же метрики на ТРЕЙН-выборке.
     Если там метрики сильно выше - модель способна, вопрос в обобщении.
     Если там тоже скромно - недоученность/ёмкость.
  3. SG top-5: насколько близко модель подходит к правильной группе.
"""

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
CKPT = BASE / "checkpoints" / "pretrain_v1_full_last.pt"
DEVICE = torch.device("cuda")
GRID_N = 4096

stats = json.loads((BASE / "outputs" / "pretrain_stats.json").read_text())
VOCAB = stats["vocab"]
SYSTEMS = stats["systems"]
LAT_MEAN = np.array(stats["lat_mean"])
LAT_STD = np.array(stats["lat_std"])
VOL_MEAN, VOL_STD = stats["vol_mean"], stats["vol_std"]
EL_IDX = {e: i for i, e in enumerate(VOCAB)}

index = pd.read_parquet(DATA / "index_preprocessed.parquet")
pre = index[index["split_role"] == "pretrain"]
splits = pd.read_parquet(BASE / "outputs" / "splits_pretrain.parquet")
syn = pd.read_parquet(BASE / "data" / "clean" / "df_synth_summary_final_clean.parquet")[
    ["sample_id", "lattice_a", "lattice_b", "lattice_c",
     "alpha", "beta", "gamma", "spacegroup_number", "crystal_system", "elements_list"]
]
df = pre.merge(syn, on="sample_id", how="left").merge(splits, on="sample_id")


def to_list(v):
    if isinstance(v, str):
        try:
            return json.loads(v
                              )
        except Exception:
            return []
        
    if isinstance(v, (list, tuple, np.ndarray)):
        return list(v)
    
    return []


df["elements"] = df["elements_list"].apply(to_list)

abc = df[["lattice_a", "lattice_b", "lattice_c"]].to_numpy(np.float64)
ang = df[["alpha", "beta", "gamma"]].to_numpy(np.float64)
ca, cb, cg = (np.cos(np.radians(ang[:, i])) for i in range(3))
t = 1.0 - ca**2 - cb**2 - cg**2 + 2 * ca * cb * cg
V = abc.prod(1) * np.sqrt(np.clip(t, 1e-12, None))
df["lat6"] = list(np.hstack([np.log(abc), ang]))
df["logV"] = np.log(V)


class DS(Dataset):
    def __init__(self, frame):
        self.row = frame["row_idx"].to_numpy(np.int64)
        lam1 = frame["lambda_1"].to_numpy(np.float32)
        lam2 = frame["lambda_2"].to_numpy(np.float32)
        self.lam = np.stack([lam1 / 1.54, np.nan_to_num(lam2) / 1.54,
                             np.isfinite(lam2).astype(np.float32)], 1)
        self.lat6 = (np.stack(frame["lat6"].to_numpy()) - LAT_MEAN) / LAT_STD
        sg = frame["spacegroup_number"].to_numpy(float)
        self.sgm = np.isfinite(sg)
        self.sg = np.nan_to_num(sg).astype(np.int64) - 1
        sysmap = {s: i for i, s in enumerate(SYSTEMS)}
        self.sys = frame["crystal_system"].map(sysmap).fillna(-1).to_numpy(np.int64)
        n = len(frame)
        self.el = np.zeros((n, len(VOCAB)), np.float32)
        for i, els in enumerate(frame["elements"]):
            for e in els:
                j = EL_IDX.get(e)
                if j is not None:
                    self.el[i, j] = 1.0
        self.vol = (frame["logV"].to_numpy(np.float32) - VOL_MEAN) / VOL_STD

    def __len__(self):
        return len(self.row)

    def __getitem__(self, i):
        x = np.empty((2, GRID_N), np.float32)
        x[0] = X_MM[self.row[i]]
        x[1] = M_MM[self.row[i]]

        return (torch.from_numpy(x), torch.from_numpy(self.lam[i]),
                torch.from_numpy(self.lat6[i]), torch.tensor(self.sg[i]),
                torch.tensor(self.sys[i]), torch.from_numpy(self.el[i]),
                torch.tensor(self.vol[i]), torch.tensor(self.row[i]))


N_TOTAL = len(index)
X_MM = np.memmap(DATA / "X_intensity.f16", dtype=np.float16, mode="r", shape=(N_TOTAL, GRID_N))
M_MM = np.memmap(DATA / "M_mask.u8", dtype=np.uint8, mode="r", shape=(N_TOTAL, GRID_N))


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


class XRDNet(nn.Module):
    def __init__(self, n_el):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv1d(2, 32, 15, padding=7, bias=False),
                                  nn.GroupNorm(8, 32), nn.GELU())
        chans = [(32, 48), (48, 64), (64, 96), (96, 128), (128, 192), (192, 256)]
        self.blocks = nn.Sequential(*[ResBlock(ci, co, stride=2) for ci, co in chans])
        self.lam_mlp = nn.Sequential(nn.Linear(3, 16), nn.GELU(), nn.Linear(16, 16))
        self.trunk = nn.Sequential(nn.Linear(256 + 16, 512), nn.GELU(),
                                   nn.Linear(512, 512), nn.GELU())
        self.head_lat = nn.Linear(512, 6)
        self.head_vol = nn.Linear(512, 1)
        self.head_sg = nn.Linear(512, 230)
        self.head_sys = nn.Linear(512, 7)
        self.head_el = nn.Linear(512, n_el)

    def forward(self, x, lam):
        f = self.stem(x)
        f = self.blocks(f)
        w = F.adaptive_avg_pool1d(x[:, 1:2], f.shape[-1]).clamp_min(1e-3)
        pooled = (f * w).sum(-1) / w.sum(-1)
        z = torch.cat([pooled, self.lam_mlp(lam)], dim=1)
        z = self.trunk(z)
        return dict(lat=self.head_lat(z), vol=self.head_vol(z).squeeze(-1),
                    sg=self.head_sg(z), sys=self.head_sys(z), el=self.head_el(z))


model = XRDNet(len(VOCAB)).to(DEVICE)
model.load_state_dict(torch.load(CKPT, map_location=DEVICE))
model.eval()


@torch.no_grad()
def collect(frame, tag, n=6000):
    if len(frame) > n:
        frame = frame.sample(n, random_state=0)
    loader = DataLoader(DS(frame), batch_size=256, shuffle=False, pin_memory=True)
    recs = []

    for x, lam, lat6, sg, sys_, el, vol, row in loader:
        x, lam = x.to(DEVICE), lam.to(DEVICE)

        with torch.autocast("cuda", dtype=torch.float16):
            out = model(x, lam)
        lat_pred = out["lat"].float().cpu().numpy() * LAT_STD + LAT_MEAN
        lat_pred[:, :3] = np.exp(lat_pred[:, :3])
        lat_true = lat6.numpy() * LAT_STD + LAT_MEAN
        lat_true[:, :3] = np.exp(lat_true[:, :3])

        for i in range(len(x)):
            recs.append(dict(
                row=int(row[i]), tag=tag,
                system=SYSTEMS[sys_[i].item()] if sys_[i].item() >= 0 else None,
                sg_true=int(sg[i].item()) + 1, sg_ok=bool(out["sg"][i].argmax().item() == sg[i].item()),
                sg_top5=bool(sg[i].item() in out["sg"][i].topk(5).indices.tolist()),
                sys_ok=bool(out["sys"][i].argmax().item() == sys_[i].item()),
                mae_a=abs(lat_pred[i, 0] - lat_true[i, 0]),
                mae_b=abs(lat_pred[i, 1] - lat_true[i, 1]),
                mae_c=abs(lat_pred[i, 2] - lat_true[i, 2]),
                mae_ang=np.abs(lat_pred[i, 3:] - lat_true[i, 3:]).mean(),
            ))
    return recs


val_df = df[df["split"] == "val"]
train_df = df[df["split"] == "train"]
print(f"оценка: val {min(6000, len(val_df))} / train {min(6000, len(train_df))} строк")
recs = collect(val_df, "val (новые соединения)")
recs += collect(train_df, "train (виденные формулы)")
r = pd.DataFrame(recs)

print()
print("=" * 78)
print("IN-DISTRIBUTION GAP (та же архитектура, разные данные)")
print("=" * 78)
for tag, g in r.groupby("tag"):
    print(f"\n[{tag}] n={len(g)}")
    print(f"  sg top-1 {g['sg_ok'].mean():.3f} | top-5 {g['sg_top5'].mean():.3f} | "
          f"sys_acc {g['sys_ok'].mean():.3f}")
    print(f"  MAE a/b/c: {g['mae_a'].mean():.2f}/{g['mae_b'].mean():.2f}/{g['mae_c'].mean():.2f} A | "
          f"углы {g['mae_ang'].mean():.2f} deg | медиана MAE a {g['mae_a'].median():.2f} A")

print()
print("=" * 78)
print("РАЗБИВКА ПО КРИСТАЛЛИЧЕСКИМ СИСТЕМАМ (val, новые соединения)")
print("=" * 78)
order = ["cubic", "hexagonal", "tetragonal", "trigonal",
         "orthorhombic", "monoclinic", "triclinic"]
va = r[r["tag"].str.startswith("val")]
rows = []
for s in order:
    g = va[va["system"] == s]
    if not len(g):
        continue
    rows.append(dict(system=s, n=len(g),
                     sys_acc=g["sys_ok"].mean(),
                     sg_top1=g["sg_ok"].mean(), sg_top5=g["sg_top5"].mean(),
                     mae_a=g["mae_a"].mean(), mae_a_med=g["mae_a"].median(),
                     mae_ang=g["mae_ang"].mean()))
tbl = pd.DataFrame(rows)
print(tbl.round(3).to_string(index=False))

fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
t = tbl.set_index("system")
t["mae_a"].plot.bar(ax=axes[0], color="#2980b9")
axes[0].set_title("MAE a, A (среднее) по системам")
axes[0].set_ylabel("A")
t["sys_acc"].plot.bar(ax=axes[1], color="#27ae60")
axes[1].set_title("Crystal system accuracy по системам")
axes[1].set_ylim(0, 1)
t["sg_top5"].plot.bar(ax=axes[2], color="#8e44ad")
axes[2].set_title("SG top-5 accuracy по системам")
axes[2].set_ylim(0, 1)
plt.tight_layout()
plt.savefig(BASE / "outputs" / "diag_pretrain_breakdown.png", dpi=130)
print("\nграфик: outputs/diag_pretrain_breakdown.png")
