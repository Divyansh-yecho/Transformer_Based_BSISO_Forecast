"""Checks that the readable model files give exactly the same network as src/profA.py.

For each model it reads the config from src/profA.py, builds the original ProfA network (taken out of src/profA.py without
running the training script) and the readable version, copies the weights of the original into the readable one, and
compares the outputs on random inputs. It also checks that every setting in the readable CONFIG matches profA.py.

usage: python models/check_equivalence.py
"""
import ast, os, sys
import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, "..")
sys.path.insert(0, os.path.join(HERE, "logistic_transformer")); import model as LT          # noqa: E402
sys.modules.pop("model"); sys.path[0] = os.path.join(HERE, "transformer_only"); import model as TO   # noqa: E402

SRC = open(os.path.join(ROOT, "src", "profA.py"), encoding="utf-8").read()
lines = SRC.split("\n")


def resolved_config(name):
    """The CFGS dictionary of profA.py (pure dictionaries) merged into its base settings, as profA.py does."""
    i0 = next(i for i, l in enumerate(lines) if l.startswith("CFGS = ")); i1 = next(i for i, l in enumerate(lines) if l.startswith("CFG = dict(dict(enc="))
    ns = {}; exec("\n".join(lines[i0:i1]), ns)
    base = dict(enc="eof", d=48, layers=3, lam=0.5, lr=5e-4, drop=0.1, gdrop=0.15, wd=0.05, mu=0.01, epochs=60, batch=64, hp=False,
                film_enso=True, noise=0.0, jitter=0, swa=0, mreg=0.0, ec=False, ecw=False, map=0.0, map2ere=False, ana=False)
    return dict(base, **ns["CFGS"][name])


def original_classes():
    """FieldEnc and ProfA from profA.py, with the few module-level constants they read."""
    tree = ast.parse(SRC); want = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name in ("FieldEnc", "ProfA")]
    ns = dict(torch=torch, nn=nn, np=np, NL=14, NCELL=283, NWP=30, WK_OFF=(1, 8), FAST_F=["mslp", "rain1"], CIC=np.zeros(283, bool),
              REGIONS={}, WK_CKS=(1, 2), NYM=30)
    exec(compile(ast.Module(body=want, type_ignores=[]), "profA_classes", "exec"), ns)
    return ns["ProfA"]


def randomise(m):
    torch.manual_seed(1)
    with torch.no_grad():
        for p in m.parameters():
            p.copy_(torch.randn_like(p) * 0.1)                     # also the zero-initialised heads, so the test is not trivial


def inputs(fields, B=5):
    g = torch.Generator().manual_seed(2)
    r = lambda *s: torch.randn(*s, generator=g)
    return r(B, 60, 6), r(B, 10, 26), {f: r(B, 30, 20) for f in fields}, r(B, 4), r


def check_config(name, mine):
    cfg = resolved_config(name)
    for k in ["d", "layers", "drop", "lam", "wk_e", "wk_m", "ere_d", "map", "pen_b", "pen_e", "pen_w", "pen_m", "lr", "wd", "batch",
              "epochs", "ema", "gdrop", "noise", "jitter", "allyr_b", "allyr_w"]:
        if k in mine:
            assert abs(float(cfg.get(k, 0.0)) - float(mine[k])) < 1e-12, f"{name}: {k} = {cfg.get(k)} in profA.py, {mine[k]} here"
    all_f = ["olr", "u850", "u200", "mslp", "rain1", "tcwv", "sst", "z850"]
    assert [f for f in all_f if f not in cfg.get("drop_fields", [])] == mine["fields"], f"{name}: field list"
    return cfg


ProfA = original_classes()
report = []

# ------------------------------------------------------------------ Logistic + Transformer (W3N)
cfg = check_config("W3N", LT.CONFIG)
orig = ProfA(cfg, {f: (1,) for f in LT.CONFIG["fields"]})
orig.skip_e = nn.Parameter(torch.zeros(14, 94, 2)); orig.skip_eb = nn.Parameter(torch.zeros(14, 2))
orig.skip_b = nn.Parameter(torch.zeros(14, 240, 2)); orig.skip_bb = nn.Parameter(torch.zeros(14, 2))
randomise(orig); orig.eval()
mine = LT.LogisticTransformer(94, 240); res = mine.load_state_dict(orig.state_dict(), strict=False); mine.eval()
assert not res.missing_keys and res.unexpected_keys == ["pos_lt"], res      # pos_lt: unused by W3N, created by ProfA anyway
bseq, rseq, fields, slow, r = inputs(LT.CONFIG["fields"]); SK, SKB = r(5, 14, 94), r(5, 240)
with torch.no_grad():
    raw0 = orig(bseq, rseq, {f: orig.enc[f](x) for f, x in fields.items()}, slow); wk0 = orig.last_w     # as run_model in profA.py
    out0 = torch.cat([raw0[..., :2] + torch.einsum("nkf,kfo->nko", SK, orig.skip_e) + orig.skip_eb,
                      raw0[..., 2:4] + torch.einsum("nf,kfo->nko", SKB, orig.skip_b) + orig.skip_bb], -1)
    out1, raw1, wk1 = mine(bseq, rseq, fields, slow, SK, SKB)
d = max((out0 - out1).abs().max().item(), (raw0[..., :4] - raw1).abs().max().item(), (wk0 - wk1).abs().max().item())
n_tr = sum(p.numel() for p in mine.parameters()); n_fz = sum(b.numel() for b in mine.buffers())
report.append(f"Logistic + Transformer (W3N): largest output difference {d:.2e}; {n_tr:,} trained weights + {n_fz:,} frozen linear "
              f"weights = {n_tr + n_fz:,} (59,500 in the run records)")
assert d < 1e-5 and n_tr + n_fz == 59500

# ------------------------------------------------------------------ Transformer alone (TXM)
cfg = check_config("TXM", TO.CONFIG)
lin_dims = [240, 91, 108, 90]                      # sizes of the four linear-input vectors in the final runs
orig = ProfA(cfg, {f: (1,) for f in TO.CONFIG["fields"]}, lt_dims=lin_dims); randomise(orig); orig.eval()
mine = TO.TransformerOnly(lin_dims); mine.load_state_dict(orig.state_dict(), strict=True); mine.eval()
bseq, rseq, fields, slow, r = inputs(TO.CONFIG["fields"])
lin = [(r(5, k), (torch.rand(5) > 0.3).float()) for k in lin_dims]
with torch.no_grad():
    orig.lt_in = lin; out0 = orig(bseq, rseq, {f: orig.enc[f](x) for f, x in fields.items()}, slow); orig.lt_in = None
    wk0, mp0 = orig.last_w, orig.last_map
    out1, wk1, mp1 = mine(bseq, rseq, fields, slow, lin)
d = max((out0 - out1).abs().max().item(), (wk0 - wk1).abs().max().item(), (mp0 - mp1).abs().max().item())
n_tr = sum(p.numel() for p in mine.parameters())
report.append(f"Transformer alone (TXM): largest output difference {d:.2e}; {n_tr:,} weights (90,746 in the run records)")
assert d < 1e-5 and n_tr == 90746

print("\n".join(report)); print("both models match src/profA.py")
