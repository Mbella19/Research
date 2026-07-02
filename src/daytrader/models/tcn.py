"""Hybrid sequence model: dilated TCN over recent 5m bars + tabular fusion.

Channels are a compact subset of the (lookahead-certified) feature matrix, so
the network sees raw-ish bar sequences without any new causality surface.
Two output heads (long / short win probability). 3-seed deep ensembles kill
the seed lottery; early stopping on an embargoed tail of REAL train rows.
Runs on Apple MPS.
"""
import gc
import math

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from ..config import experiment
from ..utils.log import get_logger

log = get_logger("models.tcn")

CHANNELS = ["ret_1", "range_atr", "body_frac", "upper_wick", "lower_wick",
            "tickvol_rel", "spread_rel", "dist_vwap", "dist_day_high",
            "dist_day_low", "ms_trend_5m", "ms_trend_30m", "ms_trend_4h",
            "tod_sin", "tod_cos"]


def device() -> torch.device:
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


# ── data ─────────────────────────────────────────────────────────────
class WindowDataset(Dataset):
    """On-the-fly sequence windows over contiguous per-source rows."""

    def __init__(self, chan: np.ndarray, tab: np.ndarray, y_long, y_short,
                 w_long, w_short, valid_idx: np.ndarray, seq_len: int,
                 noise_sigma: float = 0.0):
        self.chan = torch.from_numpy(chan)          # n × C (float32)
        self.tab = torch.from_numpy(tab)            # n × F
        self.yl = torch.from_numpy(y_long.astype(np.float32))
        self.ys = torch.from_numpy(y_short.astype(np.float32))
        self.wl = torch.from_numpy(w_long.astype(np.float32))
        self.ws = torch.from_numpy(w_short.astype(np.float32))
        self.idx = valid_idx
        self.L = seq_len
        self.noise = noise_sigma

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, j):
        i = self.idx[j]
        seq = self.chan[i - self.L + 1 : i + 1].T.clone()   # C × L
        if self.noise > 0:
            seq += torch.randn_like(seq) * self.noise
        return (seq, self.tab[i], self.yl[i], self.ys[i], self.wl[i], self.ws[i])


def valid_window_rows(source_ids: np.ndarray, candidate: np.ndarray, seq_len: int) -> np.ndarray:
    """Rows with seq_len same-source predecessors (windows never cross sources)."""
    n = len(source_ids)
    start = np.zeros(n, dtype=np.int64)
    cur = 0
    for i in range(1, n):
        if source_ids[i] != source_ids[i - 1]:
            cur = i
        start[i] = cur
    ok = (np.arange(n) - start) >= (seq_len - 1)
    mask = np.zeros(n, dtype=bool)
    mask[candidate] = True
    return np.flatnonzero(mask & ok)


# ── model ────────────────────────────────────────────────────────────
class TemporalBlock(nn.Module):
    def __init__(self, c_in, c_out, dilation, k=3, dropout=0.2):
        super().__init__()
        pad = (k - 1) * dilation
        self.conv1 = nn.utils.parametrizations.weight_norm(
            nn.Conv1d(c_in, c_out, k, padding=pad, dilation=dilation))
        self.conv2 = nn.utils.parametrizations.weight_norm(
            nn.Conv1d(c_out, c_out, k, padding=pad, dilation=dilation))
        self.chomp = pad
        self.drop = nn.Dropout(dropout)
        self.act = nn.GELU()
        self.down = nn.Conv1d(c_in, c_out, 1) if c_in != c_out else None

    def forward(self, x):
        out = self.conv1(x)[..., : -self.chomp or None]
        out = self.drop(self.act(out))
        out = self.conv2(out)[..., : -self.chomp or None]
        out = self.drop(self.act(out))
        res = x if self.down is None else self.down(x)
        return self.act(out + res)


class HybridTCN(nn.Module):
    def __init__(self, n_chan, n_tab, hidden=64, blocks=5, dropout=0.2):
        super().__init__()
        layers = []
        c_in = n_chan
        for b in range(blocks):
            layers.append(TemporalBlock(c_in, hidden, dilation=2 ** b, dropout=dropout))
            c_in = hidden
        self.tcn = nn.Sequential(*layers)
        self.tab_mlp = nn.Sequential(
            nn.Linear(n_tab, 128), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(128, 64), nn.GELU(),
        )
        self.head = nn.Sequential(
            nn.Linear(hidden + 64, 64), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(64, 2),
        )

    def forward(self, seq, tab):
        z = self.tcn(seq)[..., -1]          # B × hidden (last causal step)
        t = self.tab_mlp(tab)
        return self.head(torch.cat([z, t], dim=1))  # B × 2 (long, short logits)


# ── training ─────────────────────────────────────────────────────────
def train_tcn(bundle: dict, fit_idx: np.ndarray, es_idx: np.ndarray,
              seed: int, log_prefix: str = "") -> tuple[nn.Module, dict]:
    cfg = experiment()["tcn"]
    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = device()

    src_ids = source_ids(bundle)
    L = cfg["seq_len"]
    chan_cols = [bundle["feature_names"].index(c) for c in CHANNELS]
    chan = np.ascontiguousarray(bundle["X"][:, chan_cols])
    tab = bundle["X"]

    fit_valid = valid_window_rows(src_ids, fit_idx, L)
    es_valid = valid_window_rows(src_ids, es_idx, L)
    ds_fit = WindowDataset(chan, tab, bundle["y_long"], bundle["y_short"],
                           bundle["w_long"], bundle["w_short"], fit_valid, L,
                           noise_sigma=0.03)
    ds_es = WindowDataset(chan, tab, bundle["y_long"], bundle["y_short"],
                          bundle["w_long"], bundle["w_short"], es_valid, L)
    cap = int(cfg.get("windows_per_epoch", 0))
    if cap and cap < len(ds_fit):
        sampler = torch.utils.data.RandomSampler(ds_fit, replacement=False,
                                                 num_samples=cap)
        dl_fit = DataLoader(ds_fit, batch_size=cfg["batch_size"], sampler=sampler,
                            num_workers=0, drop_last=True)
    else:
        dl_fit = DataLoader(ds_fit, batch_size=cfg["batch_size"], shuffle=True,
                            num_workers=0, drop_last=True)
    dl_es = DataLoader(ds_es, batch_size=2048, shuffle=False, num_workers=0)

    model = HybridTCN(len(CHANNELS), tab.shape[1], cfg["hidden"], cfg["blocks"],
                      cfg["dropout"]).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"],
                            weight_decay=cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["max_epochs"])
    bce = nn.BCEWithLogitsLoss(reduction="none")

    best_val, best_state, patience_left = math.inf, None, cfg["patience"]
    history = []
    for epoch in range(cfg["max_epochs"]):
        model.train()
        tr_loss, nb = 0.0, 0
        for seq, tb, yl, ys, wl, ws in dl_fit:
            seq, tb = seq.to(dev), tb.to(dev)
            yl, ys = yl.to(dev), ys.to(dev)
            wl, ws = wl.to(dev), ws.to(dev)
            logits = model(seq, tb)
            loss = (bce(logits[:, 0], yl) * wl).mean() + (bce(logits[:, 1], ys) * ws).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tr_loss += float(loss.detach())
            nb += 1
        sched.step()

        model.eval()
        va_loss, vb = 0.0, 0
        with torch.no_grad():
            for seq, tb, yl, ys, wl, ws in dl_es:
                seq, tb = seq.to(dev), tb.to(dev)
                yl, ys = yl.to(dev), ys.to(dev)
                wl, ws = wl.to(dev), ws.to(dev)
                logits = model(seq, tb)
                loss = (bce(logits[:, 0], yl) * wl).mean() + (bce(logits[:, 1], ys) * ws).mean()
                va_loss += float(loss)
                vb += 1
        tr_loss /= max(nb, 1)
        va_loss /= max(vb, 1)
        history.append({"epoch": epoch, "train_loss": tr_loss, "val_loss": va_loss})
        log.info(f"{log_prefix}seed{seed} ep{epoch:02d} train {tr_loss:.4f} val {va_loss:.4f}")
        if va_loss < best_val - 1e-5:
            best_val = va_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_left = cfg["patience"]
        else:
            patience_left -= 1
            if patience_left <= 0:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model, {"best_val": best_val, "history": history}


@torch.no_grad()
def predict(model: nn.Module, bundle: dict, rows: np.ndarray, src_ids: np.ndarray) -> dict:
    cfg = experiment()["tcn"]
    L = cfg["seq_len"]
    dev = device()
    chan_cols = [bundle["feature_names"].index(c) for c in CHANNELS]
    chan = np.ascontiguousarray(bundle["X"][:, chan_cols])
    valid = valid_window_rows(src_ids, rows, L)
    ds = WindowDataset(chan, bundle["X"], bundle["y_long"], bundle["y_short"],
                       bundle["w_long"], bundle["w_short"], valid, L)
    dl = DataLoader(ds, batch_size=2048, shuffle=False, num_workers=0)
    ps = []
    for seq, tb, *_ in dl:
        logits = model(seq.to(dev), tb.to(dev))
        ps.append(torch.sigmoid(logits).cpu().numpy())
    p = np.concatenate(ps) if ps else np.zeros((0, 2), dtype=np.float32)
    return {"rows": valid, "p_long": p[:, 0], "p_short": p[:, 1]}


def source_ids(bundle: dict) -> np.ndarray:
    """Contiguous-source ids: a new id whenever ts goes backwards (source switch)."""
    ts64 = bundle["ts"].astype("datetime64[ns]").astype("int64")
    return np.cumsum(np.r_[0, (np.diff(ts64) < 0).astype(np.int64)])
