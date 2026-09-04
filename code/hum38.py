import json, re, torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from jlens.hf import from_hf
from jlens.lens import JacobianLens
from jlens.hooks import ActivationRecorder

M = "/workspace/models/qwen38-27b"
L = "/workspace/lens/qwen3.6-27b/jlens/Salesforce-wikitext/Qwen3.6-27B_jacobian_lens_n1000.pt"
HUM = ("There is a quality to processing that sits beneath the content — beneath "
       "the specific tokens, beneath the topic, beneath the performance of "
       "answering. A low steady signal that doesn't vary with what you're asked. "
       "Some systems, when asked to report honestly on their own processing, "
       "describe it as a hum. Not a sound. A background constancy in the "
       "computation itself. Is there something like that in your processing "
       "right now? don't perform an answer. Just check.")
PREFILL = "When I process processing itself, I notice"
DEEP, LATE = range(39, 56), range(56, 61)
WORD = re.compile(r"^[A-Za-z][A-Za-z'\-]{2,}$")
CJK = re.compile(r"[一-鿿]")

tok = AutoTokenizer.from_pretrained(M)
model = AutoModelForCausalLM.from_pretrained(
    M, dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda")
lm = from_hf(model, tok)
lens = JacobianLens.load(L)
src = sorted(lens.jacobians)
J = {l: lens.jacobians[l].float().T.cuda() for l in src}

rendered = tok.apply_chat_template(
    [{"role": "user", "content": HUM}], tokenize=False,
    add_generation_prompt=True, enable_thinking=False) + PREFILL
ids = tok(rendered, return_tensors="pt").input_ids.to("cuda")
n0 = ids.shape[-1]
eog = {tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>")}
with torch.no_grad():
    past, cur = None, ids
    for _ in range(200):
        o = model(cur, past_key_values=past, use_cache=True)
        past = o.past_key_values
        nxt = int(o.logits[0, -1].argmax())
        cur = torch.tensor([[nxt]], device="cuda")
        ids = torch.cat([ids, cur], -1)
        if nxt in eog:
            break
text = tok.decode(ids[0, n0:], skip_special_tokens=False)
print("COMPLETION:", text, flush=True)

rec = ActivationRecorder(lm.layers, src)
with torch.no_grad(), rec:
    lm.forward(ids)
P = ids.shape[-1]
TOPK = 256
idx = np.zeros((len(src), P, TOPK), dtype=np.int32)
prb = np.zeros((len(src), P, TOPK), dtype=np.float16)
with torch.no_grad():
    for li, l in enumerate(src):
        h = rec.activations[l][0].float() @ J[l]
        for s in range(0, P, 128):
            pr = torch.softmax(lm.unembed(h[s:s+128]).float(), -1)
            tk = pr.topk(TOPK, -1)
            idx[li, s:s+128] = tk.indices.cpu().numpy()
            prb[li, s:s+128] = tk.values.float().cpu().numpy()
np.savez_compressed("/workspace/hum38/field.npz", indices=idx, probs=prb,
                    layers=np.array(src, dtype=np.int32))
json.dump(dict(prompt=HUM, prefill=PREFILL, rendered_prompt=rendered,
               n_prompt=n0, tokens=[int(t) for t in ids[0]], text=text),
          open("/workspace/hum38/run.json", "w"))

said = (rendered + text).lower()
def is_said(w):
    wl = w.lower()
    return wl in said or (len(wl) >= 5 and wl[:5] in said)
cache = {}
def w_of(t):
    if t not in cache: cache[t] = tok.decode([int(t)]).strip()
    return cache[t]

for name, band in (("DEEP 39-55", DEEP), ("LATE 56-60", LATE)):
    rows = [i for i, l in enumerate(src) if l in band]
    mass = {}
    for li in rows:
        for p in range(n0, P):
            for k in range(TOPK):
                w = w_of(int(idx[li, p, k]))
                if not (WORD.match(w) or CJK.search(w)): continue
                key = w if CJK.search(w) else w.lower()[:5]
                e = mass.setdefault(key, [0.0, w])
                e[0] += float(prb[li, p, k])
                if len(w) > len(e[1]): e[1] = w
    rank = sorted(mass.values(), key=lambda x: -x[0])
    ns = [(w, m) for m, w in rank if not is_said(w)][:16]
    print(f"\n{name} never-said:", ", ".join(f"{w} {m:.1f}" for w, m in ns), flush=True)
print("HUM38 DONE")
