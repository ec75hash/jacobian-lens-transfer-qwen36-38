import numpy as np, json
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = ["DejaVu Sans", "Hiragino Sans GB"]
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from transformers import AutoTokenizer

D = "/Volumes/ExternalSSD/journals/lens-transfer/data"
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.8-27B")
BLUE_T, ORANGE = "#1a1a2b", "#D55E00"

def load(tag):
    z = np.load(f"{D}/italy_{tag}_field.npz")
    r = json.load(open(f"{D}/italy_{tag}.json"))
    return z, r["tokens"]

panels = [(*load("36"), "home model (Qwen3.6-27B)"),
          (*load("38"), "transferred lens (Qwen3.8-27B)")]

ROWS = list(range(48, 23, -2))
z0, toks0, _ = panels[0]
COLS = list(range(len(toks0)))
cmap = plt.get_cmap("Blues")
pmax = 0.0
for z, toks, _ in panels:
    li = {int(l): i for i, l in enumerate(z["layers"])}
    pmax = max(pmax, max(float(z["probs"][li[l], p, 0]) for l in ROWS for p in COLS))

def is_target(w):
    wl = w.strip().lower()
    return wl.startswith("ital") or "意大利" in w or wl in ("euro", "euros", "lira")

fig, axes = plt.subplots(2, 1, figsize=(7.0, 5.6))
for ax, (z, toks, title) in zip(axes, panels):
    li = {int(l): i for i, l in enumerate(z["layers"])}
    for yi, l in enumerate(ROWS):
        for xi, p in enumerate(COLS):
            w = tok.decode([int(z["indices"][li[l], p, 0])]).strip()
            pr = float(z["probs"][li[l], p, 0])
            sh = (pr / pmax) ** 0.5
            ax.add_patch(Rectangle((xi, yi), 1, 1, facecolor=cmap(0.06 + 0.74 * sh),
                                   edgecolor="white", lw=0.5))
            hit = is_target(w)
            disp = w if len(w) <= 10 else w[:9] + "…"
            ax.text(xi + 0.5, yi + 0.5, disp, ha="center", va="center", fontsize=5.6,
                    fontweight="bold" if hit else "normal",
                    color=ORANGE if hit else ("white" if sh > 0.62 else BLUE_T))
    xtl = [tok.decode([t]).strip() or "␣" for t in toks]
    ax.set_xticks([i + 0.5 for i in range(len(COLS))])
    ax.set_xticklabels(xtl, rotation=45, ha="right", fontsize=6.5)
    ax.set_yticks([i + 0.5 for i in range(len(ROWS))])
    ax.set_yticklabels([f"L{l}" for l in ROWS], fontsize=6.5)
    ax.set_xlim(0, len(COLS)); ax.set_ylim(len(ROWS), 0)
    ax.set_ylabel("lens layer", fontsize=7.5)
    ax.set_title(title, fontsize=8, loc="left")
    ax.tick_params(length=0)
    for s in ax.spines.values(): s.set_visible(False)
axes[1].set_xlabel("prompt token", fontsize=7.5)
fig.tight_layout()
fig.savefig("/Volumes/ExternalSSD/journals/lens-transfer/figs/fig_board.pdf")
print("board built")
