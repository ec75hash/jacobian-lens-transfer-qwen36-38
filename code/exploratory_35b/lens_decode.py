#!/usr/bin/env python3
"""lens_decode.py — apply a Jacobian lens to llama.cpp residual dumps (no HF model needed).

Input : a capture_activations output dir  <cap>/{layers/l_out-N.npy, prompt_tokens.json,
        generated_tokens.json, metadata.txt}   (l_out-N = [n_tokens, d_model] fp32,
        positions = prompt tokens then generated tokens, in order)
Weights: output_norm.weight + output.weight read from the SAME GGUF the model ran from.
Lens   : lens.pt  {"J": {layer: [d,d]}, "source_layers", "d_model", "n_prompts"}
         source layer l  <->  llama.cpp tensor l_out-l  (residual after block l)

Outputs (in --out): <tag>_field.npz {indices int32[L,T,256], probs f16[L,T,256], layers int32[L]}
                    <tag>.json      {model, prompt, n_prompt, n_gen, tokens, text, s0, ranks...}
Modes / measurements:
  --s0            decoder identity: argmax(unembed(l_out-39)) vs the model's own next token
  --rank-token S  rank of token S (exact string, e.g. " Italy") at --pos (default: last prompt pos)
  --next-ranks    rank of the true next token per layer (median over positions >= 16)  [S2]
  --logit-lens    also compute the classic logit lens (skip J) for the rank measurements
"""
import argparse, json, os, sys, time
import numpy as np
import torch

TOPK = 256


# ----------------------------------------------------------------------------- GGUF weights
def _bf16_to_f32(u16: np.ndarray) -> np.ndarray:
    return (u16.astype(np.uint32) << 16).view(np.float32)


def load_unembed_from_gguf(gguf_paths):
    """Return (W_out [V,d] f32 torch, norm_w [d] f32 torch, eps) from a (possibly split) GGUF."""
    from gguf import GGUFReader
    import gguf as gguf_mod
    found = {}
    eps = None
    for p in gguf_paths:
        r = GGUFReader(p)
        for kv in r.fields.values():
            if kv.name.endswith("attention.layer_norm_rms_epsilon") and eps is None:
                eps = float(kv.parts[kv.data[0]][0])
        for t in r.tensors:
            if t.name in ("output.weight", "output_norm.weight", "token_embd.weight"):
                found[t.name] = (t, p)
    if "output_norm.weight" not in found:
        raise SystemExit(f"output_norm.weight not found in {gguf_paths}")
    if "output.weight" not in found and "token_embd.weight" not in found:
        raise SystemExit("neither output.weight nor token_embd.weight found (tied?)")
    tied = "output.weight" not in found
    tW, _ = found["token_embd.weight" if tied else "output.weight"]
    tN, _ = found["output_norm.weight"]

    def to_f32(t):
        tt = t.tensor_type
        name = gguf_mod.GGMLQuantizationType(tt).name
        shape = [int(s) for s in reversed(t.shape.tolist())]  # gguf shape is reversed
        if name == "F32":
            arr = np.array(t.data, dtype=np.float32).reshape(shape)
        elif name == "F16":
            arr = np.array(t.data, dtype=np.float16).astype(np.float32).reshape(shape)
        elif name == "BF16":
            raw = np.array(t.data)
            arr = _bf16_to_f32(raw.view(np.uint16).reshape(-1)).reshape(shape)
        else:
            arr = gguf_mod.quants.dequantize(np.array(t.data), tt).astype(np.float32).reshape(shape)
        return torch.from_numpy(np.ascontiguousarray(arr)), name

    W, wtype = to_f32(tW)
    Nw, ntype = to_f32(tN)
    if W.ndim != 2:
        raise SystemExit(f"unexpected output.weight shape {tuple(W.shape)}")
    if W.shape[1] != Nw.shape[0]:
        W = W.T.contiguous()
    return W, Nw, (eps if eps is not None else 1e-6), {"W_type": wtype, "norm_type": ntype,
                                                        "tied": tied, "V": int(W.shape[0]),
                                                        "d": int(W.shape[1]), "eps": eps}


# ----------------------------------------------------------------------------- lens
def load_lens(path):
    ck = torch.load(path, map_location="cpu", weights_only=True)
    if "J" not in ck:
        raise SystemExit(f"{path} is not a JacobianLens file (keys {sorted(ck)})")
    J = {int(k): v for k, v in ck["J"].items()}
    return J, sorted(J), int(ck["d_model"]), int(ck.get("n_prompts", -1))


def rmsnorm(x, w, eps):
    var = x.float().pow(2).mean(-1, keepdim=True)
    return (x.float() * torch.rsqrt(var + eps)) * w


@torch.no_grad()
def decode_layer(H, J_l, W, Nw, eps, dev, chunk=192, want_full=False):
    """H [T,d] f32 -> (top idx [T,256] int32, top p [T,256] f16, full_logits or None)"""
    T = H.shape[0]
    idx = np.zeros((T, TOPK), np.int32)
    prb = np.zeros((T, TOPK), np.float16)
    full = [] if want_full else None
    for s in range(0, T, chunk):
        h = H[s:s + chunk].to(dev)
        if J_l is not None:
            h = h @ J_l.T
        z = rmsnorm(h, Nw, eps) @ W.T                     # [c,V]
        p = torch.softmax(z, -1)
        v, i = p.topk(TOPK, -1)
        idx[s:s + chunk] = i.cpu().numpy().astype(np.int32)
        prb[s:s + chunk] = v.cpu().numpy().astype(np.float16)
        if want_full:
            full.append(z.cpu())
    return idx, prb, (torch.cat(full) if want_full else None)


def rank_of(z_row, tid):
    # 1-based rank of token tid in logits row
    return int((z_row > z_row[tid]).sum().item()) + 1


# ----------------------------------------------------------------------------- IO
def read_capture(cap):
    meta = {}
    for line in open(os.path.join(cap, "metadata.txt")):
        if "=" in line:
            k, v = line.rstrip("\n").split("=", 1)
            meta[k] = v
    def rd(p):  # token pieces can be partial UTF-8 byte sequences (CJK) -> never let decoding crash
        return open(p, encoding="utf-8", errors="replace").read()
    ptoks = json.loads(rd(os.path.join(cap, "prompt_tokens.json")))
    gpath = os.path.join(cap, "generated_tokens.json")
    gtoks = json.loads(rd(gpath)) if os.path.exists(gpath) else []
    tpath = os.path.join(cap, "generated_text.txt")
    text = rd(tpath) if os.path.exists(tpath) else ""
    return meta, ptoks, gtoks, text


def layer_path(cap, l):
    return os.path.join(cap, "layers", f"l_out-{l}.npy")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap", required=True, help="capture_activations output dir for ONE prompt")
    ap.add_argument("--gguf", nargs="+", required=True, help="GGUF file(s); pass all shards")
    ap.add_argument("--lens", required=True)
    ap.add_argument("--tokenizer", default=None, help="dir with tokenizer.json (for decoding words)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--layers", default=None, help="comma list of source layers (default: all in lens)")
    ap.add_argument("--s0", action="store_true")
    ap.add_argument("--rank-token", default=None)
    ap.add_argument("--pos", type=int, default=None, help="position for --rank-token (default last prompt pos)")
    ap.add_argument("--next-ranks", action="store_true")
    ap.add_argument("--logit-lens", action="store_true")
    ap.add_argument("--no-field", action="store_true", help="skip writing the top-256 field")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    t0 = time.time()
    dev = torch.device(args.device)
    os.makedirs(args.out, exist_ok=True)
    tag = args.tag or os.path.basename(os.path.normpath(args.cap))

    W, Nw, eps, winfo = load_unembed_from_gguf(args.gguf)
    W, Nw = W.to(dev), Nw.to(dev)
    J, src_layers, d_model, n_prompts = load_lens(args.lens)
    assert d_model == winfo["d"], (d_model, winfo["d"])
    layers = [int(x) for x in args.layers.split(",")] if args.layers else src_layers
    meta, ptoks, gtoks, text = read_capture(args.cap)
    n_p, n_g = len(ptoks), len(gtoks)
    tok = None
    if args.tokenizer:
        try:
            from tokenizers import Tokenizer
            tok = Tokenizer.from_file(os.path.join(args.tokenizer, "tokenizer.json"))
        except Exception as e:
            print("tokenizer load failed:", e, file=sys.stderr)
    dec = (lambda i: tok.decode([int(i)])) if tok else (lambda i: str(int(i)))
    all_ids = [t["token_id"] for t in ptoks] + [t["token_id"] for t in gtoks]

    report = {"model": args.gguf[0], "lens": args.lens, "lens_n_prompts": n_prompts, "unembed": winfo,
              "prompt": meta.get("prompt", ""), "n_prompt": n_p, "n_gen": n_g,
              "tokens": [t.get("piece", "") for t in gtoks], "text": text,
              "generated_ids": [t["token_id"] for t in gtoks], "layers": layers}

    # ---- S0: decoder identity on l_out-39 (target layer, no J)
    if args.s0:
        # llama.cpp slices the LAST layer to output positions only: l_out-39 (and result_output)
        # have n_g+1 rows = [last prompt pos, gen pos 0 .. gen pos n_g-1]; row r predicts generated token r.
        H39 = torch.from_numpy(np.load(layer_path(args.cap, 39)))
        _, _, z = decode_layer(H39, None, W, Nw, eps, dev, want_full=True)
        pred = z.argmax(-1).numpy()
        gen_ids = [t["token_id"] for t in gtoks]
        rows_full = (H39.shape[0] == n_p + n_g)
        first_row = (n_p - 1) if rows_full else 0
        agree = [int(pred[first_row + r]) == int(gen_ids[r]) for r in range(min(n_g, pred.shape[0] - first_row))]
        ro = os.path.join(args.cap, "layers", "result_output.npy")
        ro_agree = ro_vs_gen = None
        if os.path.exists(ro):
            R = np.load(ro)
            if R.shape[0] == pred.shape[0]:
                ro_agree = float((R.argmax(-1) == pred).mean())        # our unembed vs model's own logits, all rows
                ro_vs_gen = float(np.mean([int(R.argmax(-1)[first_row + r]) == int(gen_ids[r]) for r in range(len(agree))])) if agree else None
        report["s0"] = {"rows_l_out_39": int(H39.shape[0]), "n_checked": len(agree),
                        "argmax_matches_generated": (sum(agree) / len(agree)) if agree else None,
                        "argmax_matches_result_output_all_rows": ro_agree,
                        "result_output_matches_generated": ro_vs_gen,
                        "first_mismatches": [(r, dec(pred[first_row + r]), dec(gen_ids[r])) for r, a in enumerate(agree) if not a][:5]}
        print("S0", json.dumps(report["s0"], ensure_ascii=False))
        del z, H39

    # ---- per-layer decode
    T = n_p + n_g
    field_idx = np.zeros((len(layers), T, TOPK), np.int32) if not args.no_field else None
    field_prb = np.zeros((len(layers), T, TOPK), np.float16) if not args.no_field else None
    rank_tid = None
    if args.rank_token:
        if tok is None:
            raise SystemExit("--rank-token needs --tokenizer")
        enc = tok.encode(args.rank_token, add_special_tokens=False).ids
        if len(enc) != 1:
            print(f"WARNING: {args.rank_token!r} tokenizes to {enc}; using first id", file=sys.stderr)
        rank_tid = enc[0]
    pos = args.pos if args.pos is not None else n_p - 1
    ranks, ranks_ll, next_ranks, next_ranks_ll = {}, {}, {}, {}
    need_full = bool(args.rank_token or args.next_ranks)
    for li, l in enumerate(layers):
        H = torch.from_numpy(np.load(layer_path(args.cap, l)))
        assert H.shape[0] == T, (l, H.shape, T)
        idx, prb, z = decode_layer(H, J[l].to(dev, torch.float32), W, Nw, eps, dev, want_full=need_full)
        if field_idx is not None:
            field_idx[li], field_prb[li] = idx, prb
        if args.rank_token:
            ranks[l] = rank_of(z[pos], rank_tid)
        if args.next_ranks:
            rs = [rank_of(z[i], all_ids[i + 1]) for i in range(16, T - 1)]
            next_ranks[l] = float(np.median(rs)) if rs else None
        if args.logit_lens and need_full:
            _, _, zl = decode_layer(H, None, W, Nw, eps, dev, want_full=True)
            if args.rank_token:
                ranks_ll[l] = rank_of(zl[pos], rank_tid)
            if args.next_ranks:
                rs = [rank_of(zl[i], all_ids[i + 1]) for i in range(16, T - 1)]
                next_ranks_ll[l] = float(np.median(rs)) if rs else None
        top = [dec(i) for i in idx[pos][:8]]
        print(f"L{l:02d} pos{pos} top8: {top}" + (f"  rank[{args.rank_token!r}]={ranks[l]}" if args.rank_token else "")
              + (f"  ll={ranks_ll[l]}" if l in ranks_ll else ""), flush=True)
    if args.rank_token:
        report["rank_token"] = {"token": args.rank_token, "id": rank_tid, "pos": pos, "jlens": ranks, "logit_lens": ranks_ll or None}
        best = min(ranks, key=ranks.get)
        print(f"RANK {args.rank_token!r}: best L{best}={ranks[best]}; L24-36 = {[ranks.get(l) for l in range(24, 37, 2)]}")
    if args.next_ranks:
        report["next_token_median_rank"] = {"jlens": next_ranks, "logit_lens": next_ranks_ll or None}
        print("NEXT-RANKS jlens", {l: next_ranks[l] for l in (8, 16, 24, 32) if l in next_ranks},
              "logit_lens", {l: next_ranks_ll.get(l) for l in (8, 16, 24, 32)} if next_ranks_ll else "")
    if field_idx is not None:
        np.savez_compressed(os.path.join(args.out, f"{tag}_field.npz"), indices=field_idx, probs=field_prb,
                            layers=np.array(layers, np.int32))
    report["elapsed_s"] = round(time.time() - t0, 1)
    json.dump(report, open(os.path.join(args.out, f"{tag}.json"), "w"), ensure_ascii=False, indent=1)
    print(f"wrote {args.out}/{tag}.json  ({report['elapsed_s']}s)")


if __name__ == "__main__":
    main()
