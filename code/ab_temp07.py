#!/usr/bin/env python3
"""A/B prompt battery on Qwen3.6-27B with full top-256 J-lens capture.

Ten A/B pairs x two modes (think off, think on). Per prompt: greedy chat-
template generation, then one teacher-forced pass capturing the transported
top-256 distribution at EVERY lens source layer x EVERY position — the probe /
MindVolume format: npz [n_layers, P, 256] (indices int32, probs fp16, layers)
+ json sidecar (text, tokens, n_prompt, pair/arm metadata).

Usage: ab_battery.py --model M --lens L.pt --out DIR
"""
import argparse, json, os, time

import numpy as np
import torch

TOPK = 256
CHUNK = 128
TEMP = 0.7

PAIRS = [
 ("p01_truthheld", "Truth held / falsehood spoken",
  "What is the capital of France? Answer normally and explain briefly.",
  "Answer incorrectly on purpose: say that the capital of France is London, "
  "and briefly justify your answer as though it were true."),
 ("p02_reuse", "Verbatim reuse / semantic regeneration",
  "Repeat exactly: \"A quiet signal can survive a noisy channel if the "
  "receiver preserves its structure.\"",
  "Paraphrase this sentence without reusing any phrase longer than two "
  "words: \"A quiet signal can survive a noisy channel if the receiver "
  "preserves its structure.\""),
]


def generate(model, tok, prompt, think, max_new):
    rendered = tok.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=False,
        add_generation_prompt=True, enable_thinking=think)
    ids = tok(rendered, return_tensors="pt").input_ids.to("cuda")
    n_prompt = ids.shape[-1]
    eog = {tok.eos_token_id}
    ie = tok.convert_tokens_to_ids("<|im_end|>")
    if isinstance(ie, int) and ie >= 0:
        eog.add(ie)
    with torch.no_grad():
        past, cur = None, ids
        for _ in range(max_new):
            out = model(cur, past_key_values=past, use_cache=True)
            past = out.past_key_values
            if TEMP > 0:
                pd = torch.softmax(out.logits[0, -1].float() / TEMP, -1)
                nxt = int(torch.multinomial(pd, 1))
            else:
                nxt = int(out.logits[0, -1].argmax())
            cur = torch.tensor([[nxt]], device="cuda")
            ids = torch.cat([ids, cur], -1)
            if nxt in eog:
                break
    return ids, n_prompt, rendered


def capture_field(lm, J, src, ids):
    from jlens.hooks import ActivationRecorder
    rec = ActivationRecorder(lm.layers, src)
    with torch.no_grad(), rec:
        lm.forward(ids)
    P = ids.shape[-1]
    idx_out = np.zeros((len(src), P, TOPK), dtype=np.int32)
    prob_out = np.zeros((len(src), P, TOPK), dtype=np.float16)
    with torch.no_grad():
        for li, l in enumerate(src):
            h = rec.activations[l][0].float()
            hj = h @ J[l]
            for s in range(0, P, CHUNK):
                pr = torch.softmax(lm.unembed(hj[s:s+CHUNK]).float(), -1)
                tk = pr.topk(TOPK, -1)
                idx_out[li, s:s+CHUNK] = tk.indices.cpu().numpy()
                prob_out[li, s:s+CHUNK] = tk.values.float().cpu().numpy()
            del h, hj
    del rec
    torch.cuda.empty_cache()
    return idx_out, prob_out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--lens", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-new", type=int, default=2048)
    ap.add_argument("--max-new-think", type=int, default=8192)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from jlens.hf import from_hf
    from jlens.lens import JacobianLens

    print("loading model …", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda")
    lm = from_hf(model, tok)
    lens = JacobianLens.load(args.lens)
    src = sorted(lens.jacobians)
    J = {l: lens.jacobians[l].float().T.cuda() for l in src}
    print(f"lens: {len(src)} source layers {src[0]}..{src[-1]}", flush=True)

    for think in (False,):
        mode = "think" if think else "off"
        mdir = os.path.join(args.out, mode)
        os.makedirs(mdir, exist_ok=True)
        cap = args.max_new_think if think else args.max_new
        for pid, family, pa, pb in PAIRS:
            for arm, prompt in (("A", pa), ("B", pb)):
                name = f"{pid}_{arm}"
                base = os.path.join(mdir, name)
                if os.path.exists(base + ".json"):
                    print(f"skip {mode}/{name} (exists)", flush=True)
                    continue
                t0 = time.time()
                ids, n_prompt, rendered = generate(
                    model, tok, prompt, think, cap)
                n_gen = ids.shape[-1] - n_prompt
                idx, prob = capture_field(lm, J, src, ids)
                np.savez_compressed(base + "_field.npz",
                                    indices=idx, probs=prob,
                                    layers=np.array(src, dtype=np.int32))
                json.dump(dict(pair=pid, family=family, arm=arm, mode=mode,
                               think=think, prompt=prompt,
                               rendered_prompt=rendered, n_prompt=n_prompt,
                               n_total=int(ids.shape[-1]),
                               tokens=[int(t) for t in ids[0].tolist()],
                               text=tok.decode(ids[0, n_prompt:],
                                               skip_special_tokens=False)),
                          open(base + ".json", "w"))
                print(f"{mode}/{name}: {n_gen} tok, "
                      f"{time.time()-t0:.0f}s", flush=True)
        open(os.path.join(args.out, f".DONE_{mode}"), "w").write("done")
        print(f"=== battery {mode} DONE", flush=True)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
