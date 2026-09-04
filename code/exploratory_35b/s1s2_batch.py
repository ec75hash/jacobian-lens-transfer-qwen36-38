#!/usr/bin/env python3
"""S1/S2 transfer smoke over capture_activations dumps: latent-token rank per layer (S1, 40 two-hop
items) and true-next-token median rank per layer (S2, one paragraph). J-lens vs logit lens.
Usage: s1s2_batch.py --gguf <shards...> --lens lens.pt --tokenizer <dir> --mh-cap /workspace/cap/mh40
       --mh-items mh40.json --s2-cap /workspace/cap/s2/danube --out results.json
"""
import argparse, json, os, sys
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lens_decode import load_unembed_from_gguf, load_lens, rmsnorm, layer_path, read_capture

ap = argparse.ArgumentParser()
ap.add_argument("--gguf", nargs="+", required=True); ap.add_argument("--lens", required=True)
ap.add_argument("--tokenizer", required=True); ap.add_argument("--mh-cap", required=True)
ap.add_argument("--mh-items", required=True); ap.add_argument("--s2-cap", default=None)
ap.add_argument("--out", required=True); ap.add_argument("--label", default="")
a = ap.parse_args()
dev = torch.device("cuda")
W, Nw, eps, winfo = load_unembed_from_gguf(a.gguf); W, Nw = W.to(dev), Nw.to(dev)
J, layers, d, n_prompts = load_lens(a.lens)
Jd = {l: J[l].to(dev, torch.float32) for l in layers}
from tokenizers import Tokenizer
tok = Tokenizer.from_file(os.path.join(a.tokenizer, "tokenizer.json"))

def logits(h, l):            # h [T,d] cpu f32 -> jlens logits, logit-lens logits (both [T,V] on dev)
    h = h.to(dev)
    zj = rmsnorm(h @ Jd[l].T, Nw, eps) @ W.T
    zl = rmsnorm(h, Nw, eps) @ W.T
    return zj, zl

def rank(z, tid):
    return (z > z[:, tid:tid + 1]).sum(-1) + 1   # [T] 1-based

items = json.load(open(a.mh_items))
per_item = []
S1j = {l: [] for l in layers}; S1l = {l: [] for l in layers}; top1 = {l: 0 for l in layers}
for pid, prompt, latent in items:
    cap = os.path.join(a.mh_cap, pid)
    if not os.path.isdir(cap):
        print("missing", cap); continue
    meta, ptoks, gtoks, text = read_capture(cap)
    n_p = len(ptoks); pos = n_p - 1
    enc = tok.encode(latent, add_special_tokens=False).ids
    tid = enc[0]
    rj, rl, pj = {}, {}, {}
    for l in layers:
        H = torch.from_numpy(np.load(layer_path(cap, l))[pos:pos + 1])
        zj, zl = logits(H, l)
        rj[l] = int(rank(zj, tid)[0]); rl[l] = int(rank(zl, tid)[0])
        pj[l] = float(torch.softmax(zj[0], -1)[tid])
        S1j[l].append(rj[l]); S1l[l].append(rl[l])
        if rj[l] == 1: top1[l] += 1
    best = min(rj, key=rj.get); peak = max(pj, key=pj.get)
    per_item.append({"id": pid, "latent": latent, "tid": tid, "answer": text.strip()[:30], "best_layer": best,
                     "best_rank": rj[best], "peak_prob_layer": peak, "peak_prob": pj[peak],
                     "rank_jlens": rj, "rank_logitlens": rl})
    print(f"{pid} {latent!r:16} ans={text.strip()[:18]!r:22} best L{best}={rj[best]:>5}  L24={rj[24]:>5} L28={rj[28]:>5} L32={rj[32]:>5} L36={rj[36]:>5}   ll L24={rl[24]:>6}  peakP L{peak}={pj[peak]:.3f}", flush=True)

med = {l: float(np.median(S1j[l])) for l in layers}; medl = {l: float(np.median(S1l[l])) for l in layers}
print("\nS1 median latent rank (n=%d)" % len(per_item))
for l in layers:
    if l % 2 == 0 or l == 39:
        print(f"  L{l:02d}  jlens {med[l]:>7.1f}   logit-lens {medl[l]:>8.1f}   top1 {top1[l]}/{len(per_item)}")
band = [med[l] for l in range(24, 37)]
verdict = "PASS" if (min(band) < 1000 and max(top1[l] for l in range(28, 39)) >= 0) else "FAIL"

res = {"label": a.label, "gguf": a.gguf, "lens": a.lens, "lens_n_prompts": n_prompts, "unembed": winfo,
       "s1": {"n": len(per_item), "median_rank_jlens": med, "median_rank_logitlens": medl, "top1_count": top1,
              "band24_36_min_median": min(band), "items": per_item}}

if a.s2_cap and os.path.isdir(a.s2_cap):
    meta, ptoks, gtoks, text = read_capture(a.s2_cap)
    ids = [t["token_id"] for t in ptoks] + [t["token_id"] for t in gtoks]
    T = len(ptoks) + len(gtoks)
    s2j, s2l = {}, {}
    nxt = torch.tensor(ids[17:T], device=dev)  # true next token for positions 16..T-2
    for l in layers:
        H = torch.from_numpy(np.load(layer_path(a.s2_cap, l))[16:T - 1])
        zj, zl = logits(H, l)
        rj = (zj > zj.gather(1, nxt[:, None])).sum(-1) + 1
        rl = (zl > zl.gather(1, nxt[:, None])).sum(-1) + 1
        s2j[l] = float(rj.float().median()); s2l[l] = float(rl.float().median())
    print("\nS2 next-token median rank (positions 16..%d)" % (T - 2))
    for l in (4, 8, 12, 16, 20, 24, 28, 32, 36, 38):
        if l in s2j: print(f"  L{l:02d}  jlens {s2j[l]:>8.1f}   logit-lens {s2l[l]:>9.1f}")
    print("  home-model README (3.6-35B, WikiText): L8 182  L16 106  L24 61  L32 47")
    res["s2"] = {"n_positions": T - 17, "median_rank_jlens": s2j, "median_rank_logitlens": s2l, "n_prompt": len(ptoks)}
json.dump(res, open(a.out, "w"), indent=1)
print("wrote", a.out)
