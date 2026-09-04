"""Rebuild fig_concept.pdf (Figure 1). 2026-08-27: question text removed; CJK font for the Chinese token."""
import os, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch
from matplotlib import font_manager
HERE = os.path.dirname(os.path.abspath(__file__))
GREY, EDGE, BLUE, ORANGE = "#e6e6ea", "#9a9aa2", "#1f77b4", "#e07b1c"
cjk = font_manager.FontProperties(family=["Hiragino Sans GB", "Arial Unicode MS"], size=14)
fig, (a, b) = plt.subplots(1, 2, figsize=(15.5, 6.4), gridspec_kw={"width_ratios": [1, 1.1]})
for ax in (a, b):
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")
def stack(ax, x, y0=1.4, n=8, w=1.6, h=0.55, gap=0.18, hi=None):
    ys = []
    for i in range(n):
        y = y0 + i * (h + gap); ys.append(y)
        ax.add_patch(Rectangle((x, y), w, h, facecolor=(BLUE if i == hi else GREY), edgecolor=EDGE, lw=1))
    return ys
# (a) Jacobian readout
a.set_title("(a) Jacobian readout", fontsize=15, pad=14)
ys = stack(a, 0.8, hi=3); hl = ys[3]
a.text(1.6, 0.75, "model layers", ha="center", fontsize=14)
a.text(2.55, hl + 0.12, r"$h_\ell$", fontsize=16)
a.add_patch(FancyBboxPatch((4.7, 7.3), 3.0, 0.9, boxstyle="round,pad=0.02", facecolor="#eef4fb", edgecolor=BLUE, lw=1.5))
a.text(6.2, 7.75, "final-layer basis", ha="center", va="center", fontsize=14)
a.add_patch(FancyArrowPatch((2.5, hl + 0.35), (4.7, 7.7), connectionstyle="arc3,rad=-0.35", arrowstyle="-|>", mutation_scale=18, color=BLUE, lw=2))
a.text(3.55, 6.55, r"$J_\ell$", color=BLUE, fontsize=16); a.text(2.45, 5.35, "fitted transport", color=BLUE, fontsize=12)
a.add_patch(FancyArrowPatch((5.4, 7.3), (5.4, 6.25), arrowstyle="-|>", mutation_scale=18, color="black", lw=1.8))
a.text(5.65, 6.78, "unembed", fontsize=12, va="center")
a.text(4.9, 5.55, "1. Italy", fontsize=14); a.text(4.9, 4.85, "2. 意大利", fontproperties=cjk)
a.text(4.9, 4.15, "3. euro", fontsize=14); a.text(5.05, 3.45, "⋮", fontsize=14)
a.text(6.2, 0.75, "ranked vocabulary readout", ha="center", fontsize=12, color="#777777")
# (b) Cross-checkpoint transfer
b.set_title("(b) Cross-checkpoint transfer", fontsize=15, pad=14)
ysh = stack(b, 0.8); yss = stack(b, 7.4); hh = ysh[3]
b.add_patch(Rectangle((0.62, hh - 0.14), 1.96, 0.83, fill=False, edgecolor=BLUE, lw=2.2))
b.add_patch(Rectangle((7.22, hh - 0.14), 1.96, 0.83, fill=False, edgecolor=ORANGE, lw=2.2))
b.text(1.6, hh + 0.85, "lens fitted here", ha="center", va="center", color=BLUE, fontsize=12, bbox=dict(facecolor="white", edgecolor="none", pad=2.0))
b.add_patch(FancyArrowPatch((2.75, hh + 0.28), (7.05, hh + 0.28), arrowstyle="-|>", mutation_scale=20, color="black", lw=2))
b.text(4.9, hh + 0.75, "same lens,", ha="center", fontsize=13); b.text(4.9, hh - 0.35, "no refit", ha="center", fontsize=13)
b.text(1.6, 0.55, "Qwen3.6-27B\n(home)", ha="center", fontsize=14); b.text(8.2, 0.55, "Qwen3.8-27B\n(successor)", ha="center", fontsize=14)
fig.tight_layout()
fig.savefig(os.path.join(HERE, "fig_concept.pdf")); fig.savefig(os.path.join(HERE, "fig_concept.png"), dpi=150)
print("wrote fig_concept.pdf/png")
