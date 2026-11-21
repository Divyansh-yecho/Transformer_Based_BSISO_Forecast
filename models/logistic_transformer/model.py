"""Logistic + Transformer (config W3N): the network, the frozen linear parts, the loss and the training loop.

This file is a readable copy of the parts of src/profA.py that config W3N uses, with every other option removed.
The weight names are the same as in profA.ProfA, so a state dict loads from one into the other, and
models/check_equivalence.py checks that both give the same outputs.

What stays in src/profA.py is the data side: anomalies, the 20 EOFs per field, the targets, the extreme-rain thresholds,
the normal-chance logits and the weekly ridge / logistic parts (functions build and weekly_fit). All of them are
computed from the training years of each cross-validation fold.

Shapes (B = batch, NL = 14 leads):
    bseq   (B, 60, 6)     BSISO PC1-4 and the two amplitudes on days T-59 .. T
    rseq   (B, 10, 26)    rain band (25 latitude rows) and central-India rain on days T-9 .. T
    fields dict f -> (B, 30, 20)   20 EOF coefficients of field f on days T-29 .. T
    slow   (B, 4)         Nino3.4, Dipole Mode Index, sin and cos of the day of year
    SK     (B, NL, 94)    inputs of the frozen extreme-rain logistic part, per lead
    SKB    (B, 240)       inputs of the frozen BSISO ridge part (PC1-4 on T-29 .. T, OLR EOFs at lags 0, 5, .., 25 days)
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------------------------------------------------------ config W3N, resolved from src/profA.py
CONFIG = dict(
    d=32, layers=3, heads=4, ff=64, drop=0.2,                 # encoder: width 32, 3 layers, 4 heads, feed-forward 64
    e=12,                                                      # each day of a field: 20 EOF coefficients -> 12 numbers
    fields=["u850", "mslp", "rain1", "tcwv", "sst", "z850"],   # map inputs, in token order (rain1 = IMD 1° rain map)
    fast_fields=["mslp", "rain1"],                             # these get one token per day for T-4 .. T
    lam=0.5, wk_e=1.0, wk_m=0.5, ere_d=0.0,                    # loss weights: BSISO, weekly ERE, weekly map, daily ERE (off)
    pen_b=0.25, pen_e=0.3, pen_w=0.3, pen_m=0.25,              # penalties on the size of the transformer's own correction
    lr=1.5e-4, wd=0.05, batch=64, epochs=12, ema=0.99,         # AdamW, flat rate after a one-epoch rise, weight averaging
    gdrop=0.15, noise=0.2, jitter=2,                           # group dropout, input noise, backward window shift (days)
    allyr_b=160, allyr_w=1.0,                                  # off-season BSISO-only samples per step and their weight
    l2_ere=300.0, chain_pen=1.0, alpha_bsiso=100.0,            # penalties of the frozen linear parts
)
NL = 14                 # leads (days)
WK_OFF = (1, 8)         # first lead of week 1 and week 2
NWP = 30                # EOF patterns of the weekly rain-anomaly maps
NB, NR, NS = 6, 26, 4   # numbers per day: BSISO, rain band; size of the slow state


class FieldEnc(nn.Module):
    """One day of one field: 20 EOF coefficients -> e numbers. Shared by all 30 days of the field."""
    def __init__(self, e=12):
        super().__init__()
        self.net = nn.Linear(20, e)

    def forward(self, x):
        return self.net(x)


class LogisticTransformer(nn.Module):
    """The transformer and its frozen linear parts.

    Tokens (80 of 32 numbers): 14 learned lead queries, 16 BSISO (11 five-day blocks + 5 days), 6 rain band (1 + 5),
    10 for each fast field (5 + 5), 6 for each other field (6 five-day blocks). Every group adds a learned position vector.
    """
    def __init__(self, n_ere_inputs=94, n_bsiso_inputs=240, cfg=CONFIG):
        super().__init__()
        d, e = cfg["d"], cfg["e"]; self.cfg = cfg
        self.fnames, fast = list(cfg["fields"]), cfg["fast_fields"]
        self.enc = nn.ModuleDict({f: FieldEnc(e) for f in self.fnames})
        self.p5 = nn.ModuleDict({f: nn.Linear(5 * e, d) for f in self.fnames})           # five-day block -> token
        self.p1 = nn.ModuleDict({f: nn.Linear(e, d) for f in self.fnames if f in fast})   # one day -> token (fast fields)
        self.bs = nn.Linear(5 * NB, d); self.bs1 = nn.Linear(NB, d)                       # BSISO: block token, daily token
        self.r5 = nn.Linear(5 * NR, d); self.r1 = nn.Linear(NR, d)                        # rain band: block token, daily token
        ntok = {"bsiso": 16, "rain": 6, **{f: (10 if f in fast else 6) for f in self.fnames}}
        self.pos = nn.ParameterDict({g: nn.Parameter(torch.randn(1, n, d) * 0.02) for g, n in ntok.items()})
        self.q = nn.Parameter(torch.randn(1, NL, d) * 0.02)                              # lead queries T+1 .. T+14
        self.film = nn.Linear(NS, 2 * d)                                                  # slow state -> scale and shift
        layer = nn.TransformerEncoderLayer(d, cfg["heads"], cfg["ff"], cfg["drop"], batch_first=True, norm_first=True,
                                           activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, cfg["layers"], enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, 4)              # per lead: d logit(any ERE), d logit(large | any), d PC1, d PC2
        self.head_w = nn.Linear(d, 2 + NWP)      # per week: 2 ERE logits and 30 map EOF weights
        for m in (self.film, self.head, self.head_w):                                     # zero start: epoch 0 = linear parts
            nn.init.zeros_(m.weight); nn.init.zeros_(m.bias)
        # frozen linear parts, set by fit_linear_parts (closed form) and never trained
        self.register_buffer("skip_e", torch.zeros(NL, n_ere_inputs, 2)); self.register_buffer("skip_eb", torch.zeros(NL, 2))
        self.register_buffer("skip_b", torch.zeros(NL, n_bsiso_inputs, 2)); self.register_buffer("skip_bb", torch.zeros(NL, 2))

    # -------------------------------------------------------------- the transformer's own correction
    def correction(self, bseq, rseq, fields, slow, gmask=None):
        """Returns out (B, NL, 4) and the weekly correction (B, 2, 2 + NWP). gmask: group -> (B,) 0/1 (group dropout)."""
        B = bseq.shape[0]; fast = self.cfg["fast_fields"]
        toks = [torch.cat([self.bs(bseq[:, :55].reshape(B, 11, -1)), self.bs1(bseq[:, 55:])], 1) + self.pos["bsiso"]]
        r = torch.cat([self.r5(rseq[:, :5].reshape(B, 1, -1)), self.r1(rseq[:, 5:])], 1) + self.pos["rain"]
        toks.append(r if gmask is None else r * gmask["rain"][:, None, None])
        for f in self.fnames:
            x = self.enc[f](fields[f])                                                    # (B, 30, e)
            if f in fast:
                t = torch.cat([self.p5[f](x[:, :25].reshape(B, 5, -1)), self.p1[f](x[:, 25:])], 1)
            else:
                t = self.p5[f](x.reshape(B, 6, -1))
            t = t + self.pos[f]
            toks.append(t if gmask is None else t * gmask[f][:, None, None])
        h = torch.cat([self.q.expand(B, -1, -1)] + toks, 1)                               # (B, 80, d)
        g, b = self.film(slow).chunk(2, -1); h = h * (1 + g[:, None]) + b[:, None]        # FiLM
        h = self.norm(self.encoder(h))[:, :NL]                                            # keep the 14 lead states
        hw = torch.stack([h[:, o - 1:o + 6].mean(1) for o in WK_OFF], 1)                  # week w: mean of its 7 lead states
        return self.head(h), self.head_w(hw)

    def forward(self, bseq, rseq, fields, slow, SK, SKB, gmask=None):
        """Daily output with the frozen linear parts added, the transformer's own correction, and the weekly correction.
        Daily ERE logits still need the normal-chance offset; the weekly outputs are added to the weekly linear parts
        (see forecast)."""
        raw, wk = self.correction(bseq, rseq, fields, slow, gmask)
        e_ = (torch.einsum("nkf,kfo->nko", SK, self.skip_e) if SK is not None else 0.0) + self.skip_eb
        b_ = torch.einsum("nf,kfo->nko", SKB, self.skip_b) + self.skip_bb
        out = torch.cat([raw[..., :2] + e_, raw[..., 2:4] + b_], -1)
        return out, raw, wk


def forecast(model, batch):
    """Forecasts for one batch. batch holds the inputs (see the module docstring) and the frozen linear parts of the data side:
    OFF_any, OFF_lg (B, NL) normal-chance logits; LW (B, 2, 2) weekly logistic logits; GW (B, 2, 30) weekly ridge EOF weights;
    LOCW (B, 2, 283) damped recent rain per cell; V (30, 283) the weekly map EOFs."""
    out, _, wk = model(batch["bseq"], batch["rseq"], batch["fields"], batch["slow"], batch["SK"], batch["SKB"])
    pa = torch.sigmoid(batch["OFF_any"] + out[..., 0]); pl = torch.sigmoid(batch["OFF_lg"] + out[..., 1])
    wa, wl = torch.sigmoid(batch["LW"][..., 0] + wk[..., 0]), torch.sigmoid(batch["LW"][..., 1] + wk[..., 1])
    maps = batch["LOCW"] + torch.einsum("nwp,pc->nwc", batch["GW"] + wk[..., 2:], batch["V"])
    return dict(pc=out[..., 2:4], p_any=pa, p_large=pa * pl, week_p_any=wa, week_p_large=wa * wl, week_maps=maps)


# ------------------------------------------------------------------ frozen linear parts (closed form, training rows only)
def fit_off(X, y, off, pen):
    """Logistic regression with a fixed offset (the normal-chance logit) and a ridge penalty per column; Newton steps."""
    X1 = np.column_stack([np.ones(len(X)), X]); w = np.zeros(X1.shape[1]); R_ = np.diag(np.r_[0.0, pen])
    for _ in range(60):
        p = 1 / (1 + np.exp(-np.clip(off + X1 @ w, -30, 30)))
        st = np.linalg.solve((X1 * (p * (1 - p))[:, None]).T @ X1 + R_, X1.T @ (p - y) + R_ @ w); w -= st
        if np.abs(st).max() < 1e-7:
            break
    return w


def fit_linear_parts(model, SK, Ya, Ma, La, Yl, Ml, Ll, SKB, Yp, Mp, cfg=CONFIG):
    """Sets skip_e / skip_b to their closed-form solution (numpy arrays of the training rows).
    ERE, per lead: logistic with the normal-chance offset La / Ll on SK (penalty l2_ere; chain_pen for the 3 BSISO-chain columns).
    BSISO, per lead: ridge (alpha_bsiso) on SKB; pass the off-season rows stacked under the summer rows, as in training."""
    F_ = SK.shape[2]; We = np.zeros((NL, F_, 2)); be = np.zeros((NL, 2))
    for k in range(NL):
        pen = np.r_[np.full(F_ - 3, cfg["l2_ere"]), np.full(3, cfg["chain_pen"])]
        for j, (Y, M, O) in enumerate(((Ya, Ma, La), (Yl, Ml, Ll))):
            m = M[:, k] > 0; X = SK[m, k]; nz = X.std(0) > 0                  # all-zero columns (map EOFs beyond lead 10) stay 0
            w = fit_off(X[:, nz], Y[m, k], O[m, k], pen[nz]); be[k, j] = w[0]; We[k, nz, j] = w[1:]
    Fb = SKB.shape[1]; Wb = np.zeros((NL, Fb, 2)); bb = np.zeros((NL, 2))
    for k in range(NL):
        m = (Mp[:, k] > 0).all(1); X1 = np.column_stack([np.ones(m.sum()), SKB[m]])
        w = np.linalg.solve(X1.T @ X1 + cfg["alpha_bsiso"] * np.diag(np.r_[0.0, np.ones(Fb)]), X1.T @ Yp[m, k])
        bb[k] = w[0]; Wb[k] = w[1:]
    t = lambda a: torch.as_tensor(a, dtype=torch.float32, device=model.skip_e.device)
    with torch.no_grad():
        model.skip_e.copy_(t(We)); model.skip_eb.copy_(t(be)); model.skip_b.copy_(t(Wb)); model.skip_bb.copy_(t(bb))


# ------------------------------------------------------------------ loss
def bce(logit, y, m):
    return (F.binary_cross_entropy_with_logits(logit, y, reduction="none") * m).sum() / m.sum().clamp(min=1)


def loss_fn(out, raw, wk, b, cfg=CONFIG):
    """Summer batch b: Ya/Ma, Yl/Ml (B, NL) daily ERE targets and masks with offsets OFF_any/OFF_lg; Yp/Mp (B, NL, 2) observed
    PC1/PC2 and mask; pcsd (2,) their standard deviations; YWa/MWa, YWl/MWl (B, 2) weekly classes; LW, GW, RW (target EOF
    weights), MW (B, 2) valid weeks; wsc the residual variance of the weekly map target (loss scale)."""
    ba = bce(b["OFF_any"] + out[..., 0], b["Ya"], b["Ma"]); bl = bce(b["OFF_lg"] + out[..., 1], b["Yl"], b["Ml"])
    mp = ((((out[..., 2:4] - b["Yp"]) / b["pcsd"]) ** 2) * b["Mp"]).sum() / b["Mp"].sum().clamp(min=1)
    L = cfg["ere_d"] * (ba + bl) + cfg["lam"] * mp                                      # daily ERE weight 0: left to the linear part
    wa = bce(b["LW"][..., 0] + wk[..., 0], b["YWa"], b["MWa"]); wl = bce(b["LW"][..., 1] + wk[..., 1], b["YWl"], b["MWl"])
    mw = b["MW"]
    wm = ((((b["GW"] + wk[..., 2:] - b["RW"]) ** 2) * mw[..., None]).sum() / (mw.sum() * NWP).clamp(min=1)) / b["wsc"]
    L = L + cfg["wk_e"] * (wa + wl) + cfg["wk_m"] * wm
    L = L + cfg["pen_w"] * (wk[..., :2] ** 2).mean() + cfg["pen_m"] * (wk[..., 2:] ** 2).mean() / b["wsc"]
    L = L + cfg["pen_b"] * ((raw[..., 2:4] / b["pcsd"]) ** 2).mean() + cfg["pen_e"] * (raw[..., :2] ** 2).mean()
    return L


def offseason_loss(out, raw, b, cfg=CONFIG):
    """Off-season batch (BSISO and OLR inputs only, the rest masked): BSISO loss and its size penalty."""
    return cfg["lam"] * cfg["allyr_w"] * (((out[..., 2:4] - b["Yp"]) / b["pcsd"]) ** 2).mean() \
        + cfg["pen_b"] * ((raw[..., 2:4] / b["pcsd"]) ** 2).mean()


# ------------------------------------------------------------------ training
def group_masks(model, n, device, fixed=None, cfg=CONFIG):
    """Group dropout: each input group (rain band and each field) is switched off for a sample with chance gdrop.
    fixed: group -> 0/1 for groups that are always off (off-season samples have no rain or ERA5 summer inputs)."""
    gm = {g: (torch.rand(n, device=device) > cfg["gdrop"]).float() for g in ["rain"] + model.fnames}
    return gm if fixed is None else {g: gm[g] * fixed[g] for g in gm}


def noisy(b, cfg=CONFIG):
    """Input noise while training: half on the BSISO, full on the rain band and the field EOFs. The backward window shift
    (jitter) is applied where the 30-day windows are cut, on the data side."""
    nz = cfg["noise"]
    return dict(b, bseq=b["bseq"] + 0.5 * nz * torch.randn_like(b["bseq"]), rseq=b["rseq"] + nz * torch.randn_like(b["rseq"]),
                fields={f: x + nz * torch.randn_like(x) for f, x in b["fields"].items()})


def train(model, summer_batches, offseason_batches, n_steps_per_epoch, cfg=CONFIG):
    """AdamW on the transformer only (the linear parts are buffers), rate rising over the first epoch and then flat,
    gradient clipping at 1, weights averaged every step (EMA). Each step uses one summer batch (64 rows) and one
    off-season batch (160 rows). Returns the averaged model, which gives the forecasts."""
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, 0.1 + 0.9 * s / n_steps_per_epoch))
    ema = torch.optim.swa_utils.AveragedModel(model, avg_fn=lambda a, p, n: cfg["ema"] * a + (1 - cfg["ema"]) * p)
    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        for _ in range(n_steps_per_epoch):
            b, y = noisy(next(summer_batches)), noisy(next(offseason_batches))
            out, raw, wk = model(b["bseq"], b["rseq"], b["fields"], b["slow"], b["SK"], b["SKB"],
                                 group_masks(model, len(b["bseq"]), b["bseq"].device))
            L = loss_fn(out, raw, wk, b)
            oy, rawy, _ = model(y["bseq"], y["rseq"], y["fields"], y["slow"], None, y["SKB"],
                                group_masks(model, len(y["bseq"]), y["bseq"].device, y["fixed"]))
            L = L + offseason_loss(oy, rawy, y)
            opt.zero_grad(); L.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sch.step()
            ema.update_parameters(model)
    return ema.module
