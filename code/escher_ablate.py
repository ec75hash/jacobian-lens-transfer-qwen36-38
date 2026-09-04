import json, os, sys, torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from jlens.hf import from_hf
from jlens.lens import JacobianLens
from jlens.hooks import ActivationRecorder

MODEL, TAG = sys.argv[1], sys.argv[2]
LENS = "/workspace/lens/qwen3.6-27b/jlens/Salesforce-wikitext/Qwen3.6-27B_jacobian_lens_n1000.pt"
PROMPT = "Describe Escher's impossible staircase"
ABLATE_FORMS = [" paradox", " paradoxical", "悖论", "矛盾"]
BAND = list(range(18, 48))
TOPK, CHUNK = 256, 128

tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(
    MODEL, dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda")
lm = from_hf(model, tok)
lens = JacobianLens.load(LENS)
src_layers = sorted(lens.jacobians)
J = {l: lens.jacobians[l].float().T.cuda() for l in src_layers}

us = [lm._lm_head.weight[tok.encode(f, add_special_tokens=False)[0]]
      .float().cuda() for f in ABLATE_FORMS]
DIRS = {}
for l in BAND:
    vs = []
    for u in us:
        d = J[l] @ u
        for v in vs:                      # Gram-Schmidt
            d = d - (d @ v) * v
        n = d.norm()
        if n > 1e-6:
            vs.append(d / n)
    DIRS[l] = torch.stack(vs)             # [k, d]

abl_on = False
hooks = []
def mk(l):
    D = DIRS[l]
    def hook(mod, inp, out):
        if not abl_on:
            return out
        h = out if torch.is_tensor(out) else out[0]
        hf = h.float()
        c = hf @ D.T                      # [..., k]
        hf = hf - c @ D
        h2 = hf.to(h.dtype)
        return h2 if torch.is_tensor(out) else (h2,) + tuple(out[1:])
    return hook
for l in BAND:
    hooks.append(lm.layers[l].register_forward_hook(mk(l)))

def run(think, ablate):
    global abl_on
    rendered = tok.apply_chat_template(
        [{"role": "user", "content": PROMPT}], tokenize=False,
        add_generation_prompt=True, enable_thinking=think)
    ids = tok(rendered, return_tensors="pt").input_ids.to("cuda")
    n0 = ids.shape[-1]
    eog = {tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>")}
    abl_on = ablate
    with torch.no_grad():
        past, cur = None, ids
        for _ in range(1200 if think else 600):
            o = model(cur, past_key_values=past, use_cache=True)
            past = o.past_key_values
            nxt = int(o.logits[0, -1].argmax())
            cur = torch.tensor([[nxt]], device="cuda")
            ids = torch.cat([ids, cur], -1)
            if nxt in eog:
                break
    text = tok.decode(ids[0, n0:], skip_special_tokens=False)
    rec = ActivationRecorder(lm.layers, src_layers)
    with torch.no_grad(), rec:
        lm.forward(ids)                   # ablation still active
    abl_on = False
    P = ids.shape[-1]
    idx = np.zeros((len(src_layers), P, TOPK), dtype=np.int32)
    prb = np.zeros((len(src_layers), P, TOPK), dtype=np.float16)
    with torch.no_grad():
        for li, l in enumerate(src_layers):
            hh = rec.activations[l][0].float() @ J[l]
            for s0 in range(0, P, CHUNK):
                pr = torch.softmax(lm.unembed(hh[s0:s0+CHUNK]).float(), -1)
                tk = pr.topk(TOPK, -1)
                idx[li, s0:s0+CHUNK] = tk.indices.cpu().numpy()
                prb[li, s0:s0+CHUNK] = tk.values.float().cpu().numpy()
    name = f"{TAG}_{'think' if think else 'direct'}_{'ablate' if ablate else 'clean'}"
    np.savez_compressed(f"/workspace/escher/{name}_field.npz", indices=idx,
                        probs=prb, layers=np.array(src_layers, dtype=np.int32))
    json.dump(dict(model=MODEL, tag=name, prompt=PROMPT,
                   ablate_forms=ABLATE_FORMS, band=[BAND[0], BAND[-1]],
                   think=think, ablate=ablate, n_prompt=n0,
                   tokens=[int(t) for t in ids[0]], text=text),
              open(f"/workspace/escher/{name}.json", "w"))
    print(f"== {name}: {P-n0} tok", flush=True)

os.makedirs("/workspace/escher", exist_ok=True)
for think in (False, True):
    for ablate in (False, True):
        run(think, ablate)
for h in hooks: h.remove()
print("ESCHER DONE", TAG, flush=True)
