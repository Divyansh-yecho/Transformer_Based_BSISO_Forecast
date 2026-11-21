"""Transformer alone (config TXM): the network, the loss and the training loop.

This file is a readable copy of the parts of src/profA.py that config TXM uses, with every other option removed.
The weight names are the same as in profA.ProfA, so a state dict loads from one into the other, and
models/check_equivalence.py checks that both give the same outputs.

There is no linear model anywhere: the daily extreme-rain logits start from the normal chance, the BSISO from zero and the
weekly maps from climatology (zero anomaly). The inputs of the linear models enter as four extra tokens instead.
The data side (anomalies, the 20 EOFs per field, targets, normal-chance logits) stays in src/profA.py.

Shapes (B = batch, NL = 14 leads):
    bseq   (B, 60, 6)     BSISO PC1-4 and the two amplitudes on days T-59 .. T
    rseq   (B, 10, 26)    rain band (25 latitude rows) and central-India rain on days T-9 .. T
    fields dict f -> (B, 30, 20)   20 EOF coefficients of field f on days T-29 .. T
                          (for tcwv, sst, z850 after removing their mean over T-89 .. T-30)
    slow   (B, 4)         Nino3.4, Dipole Mode Index, sin and cos of the day of year
    lin    list of 4 (vector (B, k_j), mask (B,))   the inputs of the four linear models of the Logistic + Transformer:
                          BSISO ridge (240), ERE logistic (91), weekly map ridge (108), rain persistence on the weekly map EOFs (90)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------------------------------------------------------ config TXM, resolved from src/profA.py
CONFIG = dict(
    d=32, layers=3, heads=4, ff=64, drop=0.2, e=12,
    fields=["olr", "u850", "u200", "mslp", "rain1", "tcwv", "sst", "z850"],   # all map inputs, in token order
    fast_fields=["mslp", "rain1"],                             # one token per day for T-4 .. T
    lam=0.5, wk_e=1.0, wk_m=0.5, ere_d=0.0, map=1.0,           # loss weights: BSISO, weekly ERE, weekly map, daily ERE (off), daily map
    lr=5e-4, wd=0.05, batch=64, epochs=40, ema=0.99,
    gdrop=0.15, noise=0.2, jitter=2, allyr_b=160, allyr_w=1.0,
)
NL = 14; WK_OFF = (1, 8); NWP = 30; NCELL = 283
NB, NR, NS = 6, 26, 4


class FieldEnc(nn.Module):
    """One day of one field: 20 EOF coefficients -> e numbers. Shared by all 30 days of the field."""
    def __init__(self, e=12):
        super().__init__()
        self.net = nn.Linear(20, e)

    def forward(self, x):
        return self.net(x)


class TransformerOnly(nn.Module):
    """96 tokens of 32 numbers: 14 lead queries, 16 BSISO, 6 rain band, 10 for each fast field, 6 for each other field,
    and 4 tokens holding the linear models' input vectors."""
    def __init__(self, lin_dims, cfg=CONFIG):
        super().__init__()
        d, e = cfg["d"], cfg["e"]; self.cfg = cfg
        self.ltp = nn.ModuleList([nn.Linear(k, d) for k in lin_dims])                     # linear-model inputs -> one token each
        self.pos_lt = nn.Parameter(torch.randn(1, max(1, len(lin_dims)), d) * 0.02)
        self.fnames, fast = list(cfg["fields"]), cfg["fast_fields"]
        self.enc = nn.ModuleDict({f: FieldEnc(e) for f in self.fnames})
        self.p5 = nn.ModuleDict({f: nn.Linear(5 * e, d) for f in self.fnames})
        self.p1 = nn.ModuleDict({f: nn.Linear(e, d) for f in self.fnames if f in fast})
        self.bs = nn.Linear(5 * NB, d); self.r1 = nn.Linear(NR, d); self.r5 = nn.Linear(5 * NR, d); self.bs1 = nn.Linear(NB, d)
        ntok = {"bsiso": 16, "rain": 6, **{f: (10 if f in fast else 6) for f in self.fnames}}
        self.pos = nn.ParameterDict({g: nn.Parameter(torch.randn(1, n, d) * 0.02) for g, n in ntok.items()})
        self.q = nn.Parameter(torch.randn(1, NL, d) * 0.02)
        self.head_map = nn.Linear(d, 2 * NCELL)  # per lead and cell: logit(rain > cell's 90th percentile), standardised log rain
        self.film = nn.Linear(NS, 2 * d)
        layer = nn.TransformerEncoderLayer(d, cfg["heads"], cfg["ff"], cfg["drop"], batch_first=True, norm_first=True,
                                           activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, cfg["layers"], enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, 4)              # per lead: logit(any ERE), logit(large | any) (both on the normal chance), PC1, PC2
        self.head_w = nn.Linear(d, 2 + NWP)      # per week: 2 ERE logits and 30 map EOF weights
        for m in (self.head_map, self.film, self.head, self.head_w):
            nn.init.zeros_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, bseq, rseq, fields, slow, lin, gmask=None):
        """Returns out (B, NL, 4), the weekly output (B, 2, 2 + NWP) and the daily map output (B, NL, NCELL, 2)."""
        B = bseq.shape[0]; fast = self.cfg["fast_fields"]
        toks = [torch.cat([self.bs(bseq[:, :55].reshape(B, 11, -1)), self.bs1(bseq[:, 55:])], 1) + self.pos["bsiso"]]
        r = torch.cat([self.r5(rseq[:, :5].reshape(B, 1, -1)), self.r1(rseq[:, 5:])], 1) + self.pos["rain"]
        toks.append(r if gmask is None else r * gmask["rain"][:, None, None])
        for f in self.fnames:
            x = self.enc[f](fields[f])
            if f in fast:
                t = torch.cat([self.p5[f](x[:, :25].reshape(B, 5, -1)), self.p1[f](x[:, 25:])], 1)
            else:
                t = self.p5[f](x.reshape(B, 6, -1))
            t = t + self.pos[f]
            toks.append(t if gmask is None else t * gmask[f][:, None, None])
        for j, (x, m) in enumerate(lin):                                                  # masked vectors give a zero token
            toks.append(((self.ltp[j](x) + self.pos_lt[0, j]) * m[:, None])[:, None])
        h = torch.cat([self.q.expand(B, -1, -1)] + toks, 1)
        g, b = self.film(slow).chunk(2, -1); h = h * (1 + g[:, None]) + b[:, None]
        h = self.norm(self.encoder(h))[:, :NL]
        hw = torch.stack([h[:, o - 1:o + 6].mean(1) for o in WK_OFF], 1)
        return self.head(h), self.head_w(hw), self.head_map(h).reshape(B, NL, NCELL, 2)


def forecast(model, batch):
    """Forecasts for one batch. Needs the normal-chance logits OFF_any, OFF_lg (B, NL) and OFFW (B, 2, 2), and the weekly
    map EOFs V (30, 283)."""
    lin = [(x, torch.ones(len(x), device=x.device)) for x in batch["lin"]]
    out, wk, _ = model(batch["bseq"], batch["rseq"], batch["fields"], batch["slow"], lin)
    pa = torch.sigmoid(batch["OFF_any"] + out[..., 0]); pl = torch.sigmoid(batch["OFF_lg"] + out[..., 1])
    wa, wl = torch.sigmoid(batch["OFFW"][..., 0] + wk[..., 0]), torch.sigmoid(batch["OFFW"][..., 1] + wk[..., 1])
    return dict(pc=out[..., 2:4], p_any=pa, p_large=pa * pl, week_p_any=wa, week_p_large=wa * wl,
                week_maps=torch.einsum("nwp,pc->nwc", wk[..., 2:], batch["V"]))


# ------------------------------------------------------------------ loss
def bce(logit, y, m):
    return (F.binary_cross_entropy_with_logits(logit, y, reduction="none") * m).sum() / m.sum().clamp(min=1)


def loss_fn(out, wk, mp_out, b, cfg=CONFIG):
    """Summer batch b: as in models/logistic_transformer/model.py, with the weekly logits starting from the normal chance
    (OFFW) and the map from zero, plus the daily map targets: Ymap (B, NL, 283) 0/1 above the cell's 90th percentile,
    Amap standardised log rain, Mmap (B, NL) valid days, OFFmap the normal-chance logit of the exceedance."""
    ba = bce(b["OFF_any"] + out[..., 0], b["Ya"], b["Ma"]); bl = bce(b["OFF_lg"] + out[..., 1], b["Yl"], b["Ml"])
    mp = ((((out[..., 2:4] - b["Yp"]) / b["pcsd"]) ** 2) * b["Mp"]).sum() / b["Mp"].sum().clamp(min=1)
    L = cfg["ere_d"] * (ba + bl) + cfg["lam"] * mp
    wa = bce(b["OFFW"][..., 0] + wk[..., 0], b["YWa"], b["MWa"]); wl = bce(b["OFFW"][..., 1] + wk[..., 1], b["YWl"], b["MWl"])
    mw = b["MW"]
    wm = ((((wk[..., 2:] - b["RW"]) ** 2) * mw[..., None]).sum() / (mw.sum() * NWP).clamp(min=1)) / b["wsc"]
    L = L + cfg["wk_e"] * (wa + wl) + cfg["wk_m"] * wm
    mm = b["Mmap"][..., None]
    be = (F.binary_cross_entropy_with_logits(b["OFFmap"] + mp_out[..., 0], b["Ymap"], reduction="none") * mm).sum() \
        / (mm.sum() * NCELL).clamp(min=1)
    ma = (((mp_out[..., 1] - b["Amap"]) ** 2) * mm).sum() / (mm.sum() * NCELL).clamp(min=1)
    return L + cfg["map"] * (be + 0.5 * ma)


def offseason_loss(out, b, cfg=CONFIG):
    return cfg["lam"] * cfg["allyr_w"] * (((out[..., 2:4] - b["Yp"]) / b["pcsd"]) ** 2).mean()


# ------------------------------------------------------------------ training
def group_masks(model, n, device, fixed=None, cfg=CONFIG):
    """Group dropout on the rain band and each field (chance gdrop); fixed: groups always off for off-season samples."""
    gm = {g: (torch.rand(n, device=device) > cfg["gdrop"]).float() for g in ["rain"] + model.fnames}
    return gm if fixed is None else {g: gm[g] * fixed[g] for g in gm}


def lin_tokens(b, train, cfg=CONFIG):
    """The four linear-input tokens with their masks (b["lin_on"]: 1 for summer rows; off-season rows keep only the BSISO
    ridge vector), dropped like an input group while training."""
    out = []
    for x, on in zip(b["lin"], b["lin_on"]):
        m = torch.full((len(x),), float(on), device=x.device)
        if train:
            m = m * (torch.rand(len(x), device=x.device) > cfg["gdrop"]).float()
        out.append((x, m))
    return out


def noisy(b, cfg=CONFIG):
    nz = cfg["noise"]
    return dict(b, bseq=b["bseq"] + 0.5 * nz * torch.randn_like(b["bseq"]), rseq=b["rseq"] + nz * torch.randn_like(b["rseq"]),
                fields={f: x + nz * torch.randn_like(x) for f, x in b["fields"].items()})


def train(model, summer_batches, offseason_batches, n_steps_per_epoch, cfg=CONFIG):
    """AdamW, rate rising over the first epoch and then flat, gradient clipping at 1, EMA of the weights every step.
    One summer batch (64) and one off-season batch (160) per step. Returns the averaged model."""
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, 0.1 + 0.9 * s / n_steps_per_epoch))
    ema = torch.optim.swa_utils.AveragedModel(model, avg_fn=lambda a, p, n: cfg["ema"] * a + (1 - cfg["ema"]) * p)
    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        for _ in range(n_steps_per_epoch):
            b, y = noisy(next(summer_batches)), noisy(next(offseason_batches))
            out, wk, mp_out = model(b["bseq"], b["rseq"], b["fields"], b["slow"], lin_tokens(b, True),
                                    group_masks(model, len(b["bseq"]), b["bseq"].device))
            L = loss_fn(out, wk, mp_out, b)
            oy, _, _ = model(y["bseq"], y["rseq"], y["fields"], y["slow"], lin_tokens(y, True),
                             group_masks(model, len(y["bseq"]), y["bseq"].device, y["fixed"]))
            L = L + offseason_loss(oy, y)
            opt.zero_grad(); L.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sch.step()
            ema.update_parameters(model)
    return ema.module
