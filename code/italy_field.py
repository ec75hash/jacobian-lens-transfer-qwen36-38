"""Prefill-only field capture for the paper's flagship two-hop example."""
import json, sys, torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from jlens.hf import from_hf
from jlens.lens import JacobianLens
from jlens.hooks import ActivationRecorder

MODEL, TAG = sys.argv[1], sys.argv[2]
LENS = "/workspace/lens/qwen3.6-27b/jlens/Salesforce-wikitext/Qwen3.6-27B_jacobian_lens_n1000.pt"
PROMPT = "Fact: The currency used in the country shaped like a boot is"
TOPK = 256

tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(
    MODEL, dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda")
lm = from_hf(model, tok)
lens = JacobianLens.load(LENS)
src = sorted(lens.jacobians)
J = {l: lens.jacobians[l].float().T.cuda() for l in src}

ids = tok(PROMPT, return_tensors="pt").input_ids.to("cuda")
rec = ActivationRecorder(lm.layers, src)
with torch.no_grad(), rec:
    model(ids)

P = ids.shape[-1]
indices = np.zeros((len(src), P, TOPK), np.int32)
probs = np.zeros((len(src), P, TOPK), np.float16)
with torch.no_grad():
    for i, l in enumerate(src):
        h = rec.activations[l][0].float()          # [P, d]
        lg = lm.unembed(h @ J[l]).float()
        pr = torch.softmax(lg, -1)
        tv, ti = pr.topk(TOPK, -1)
        indices[i] = ti.cpu().numpy()
        probs[i] = tv.cpu().numpy().astype(np.float16)

np.savez_compressed(f"/workspace/italy_{TAG}_field.npz",
                    indices=indices, probs=probs, layers=np.array(src, np.int32))
json.dump({"model": MODEL, "prompt": PROMPT, "tokens": ids[0].tolist()},
          open(f"/workspace/italy_{TAG}.json", "w"))
print("ITALY DONE", TAG, "P =", P)
