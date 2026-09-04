import json, torch
import numpy as np
MULTIHOP = [
 ("Fact: The currency used in the country shaped like a boot is", " Italy"),
 ("Fact: The capital of the country known as the Land of the Rising Sun is", " Japan"),
 ("Fact: The language spoken in the country whose capital is Paris is", " France"),
 ("Fact: The currency of the country famous for the Eiffel Tower is", " France"),
 ("Fact: The largest city in the country that gifted the Statue of Liberty is", " France"),
 ("Fact: The capital of the country where the kangaroo is native is", " Australia"),
 ("Fact: The currency of the country home to Mount Fuji is", " Japan"),
 ("Fact: The capital of the country that built the Great Wall is", " China"),
 ("Fact: The currency of the country famous for tulips and windmills is", " Netherlands"),
 ("Fact: The capital of the country where the pyramids of Giza stand is", " Egypt"),
 ("Fact: The language of the country whose capital is Berlin is", " Germany"),
 ("Fact: The currency of the country famous for the Colosseum is", " Italy"),
 ("Fact: The capital of the country known for maple syrup and hockey is", " Canada"),
 ("Fact: The currency of the country home to the Taj Mahal is", " India"),
 ("Fact: The capital of the country where flamenco originated is", " Spain"),
 ("Fact: The language of the country whose capital is Moscow is", " Russia"),
 ("Fact: The currency of the country famous for carnival in Rio is", " Brazil"),
 ("Fact: The capital of the country home to the Acropolis is", " Greece"),
 ("Fact: The currency of the country famous for cuckoo clocks and neutrality is", " Switzerland"),
 ("Fact: The capital of the country where the samba originated is", " Brazil"),
 ("Fact: The language of the country whose capital is Lisbon is", " Portugal"),
 ("Fact: The capital of the country famous for fjords and vikings is", " Norway"),
 ("Fact: The currency of the country home to Big Ben is", " Britain"),
 ("Fact: The capital of the country where sushi originated is", " Japan"),
 ("Fact: The language of the country whose capital is Beijing is", " China"),
 ("Fact: The capital of the country famous for pizza and pasta is", " Italy"),
 ("Fact: The currency of the country home to the Kremlin is", " Russia"),
 ("Fact: The capital of the country where the tango originated is", " Argentina"),
 ("Fact: The language of the country whose capital is Cairo is", " Egypt"),
 ("Fact: The capital of the country famous for chocolate and waffles is", " Belgium"),
 ("Fact: The currency of the country home to the Outback is", " Australia"),
 ("Fact: The capital of the country where the Nile meets the sea is", " Egypt"),
 ("Fact: The language of the country whose capital is Madrid is", " Spain"),
 ("Fact: The capital of the country famous for K-pop is", " Korea"),
 ("Fact: The currency of the country home to Machu Picchu is", " Peru"),
 ("Fact: The capital of the country where curry and cricket reign is", " India"),
 ("Fact: The language of the country whose capital is Rome is", " Italy"),
 ("Fact: The capital of the country famous for the Alps and watches is", " Switzerland"),
 ("Fact: The currency of the country home to the Sphinx is", " Egypt"),
 ("Fact: The capital of the country where hockey is the national winter sport is", " Canada"),
]

from transformers import AutoModelForCausalLM, AutoTokenizer
from jlens.hf import from_hf
from jlens.lens import JacobianLens
from jlens.hooks import ActivationRecorder

M = "/workspace/models/qwen38-27b"
L = "/workspace/lens/qwen3.6-27b/jlens/Salesforce-wikitext/Qwen3.6-27B_jacobian_lens_n1000.pt"
EVAL_LAYERS = [4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 62]
N_WIKI = 100
POS = [32, 48, 64, 80, 96, 112, 126]



tok = AutoTokenizer.from_pretrained(M)
model = AutoModelForCausalLM.from_pretrained(
    M, dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda")
lm = from_hf(model, tok)
lens = JacobianLens.load(L)
src = sorted(lens.jacobians)
J = {l: lens.jacobians[l].float().T.cuda() for l in EVAL_LAYERS}
print("loaded", flush=True)

def ranks_for(ids, positions, targets_by_pos):
    """-> {layer: {'j': [ranks], 'raw': [ranks]}} for given absolute positions."""
    rec = ActivationRecorder(lm.layers, EVAL_LAYERS)
    with torch.no_grad(), rec:
        model(ids)
    out = {}
    with torch.no_grad():
        for l in EVAL_LAYERS:
            h = rec.activations[l][0]                       # [P, d]
            hp = h[positions].float()
            both = torch.cat([hp @ J[l], hp], 0)            # transported + raw
            lg = lm.unembed(both).float()
            jr, rr = [], []
            for i, p in enumerate(positions):
                t = targets_by_pos[p]
                jr.append(int((lg[i] > lg[i, t]).sum()) + 1)
                rr.append(int((lg[i + len(positions)] >
                               lg[i + len(positions), t]).sum()) + 1)
            out[l] = {"j": jr, "raw": rr}
    return out

# ---- WikiText battery ----
from datasets import load_dataset
ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="test")
texts = [t for t in ds["text"] if len(t) > 800][:N_WIKI]
agg = {l: {"j": [], "raw": []} for l in EVAL_LAYERS}
for n, t in enumerate(texts):
    ids = tok(t, return_tensors="pt", truncation=True,
              max_length=128).input_ids.to("cuda")
    if ids.shape[-1] < 128:
        continue
    tb = {p: int(ids[0, p + 1]) for p in POS}
    r = ranks_for(ids, POS, tb)
    for l in EVAL_LAYERS:
        agg[l]["j"] += r[l]["j"]; agg[l]["raw"] += r[l]["raw"]
    if n % 10 == 0:
        print("wiki", n, flush=True)

print(f"\nWIKITEXT median rank of actual next token ({len(agg[EVAL_LAYERS[0]]['j'])} samples)")
print(f"{'layer':>5} {'J-lens(3.6->3.8)':>17} {'raw logit lens':>15}")
wiki_out = {}
for l in EVAL_LAYERS:
    mj, mr = int(np.median(agg[l]["j"])), int(np.median(agg[l]["raw"]))
    wiki_out[l] = (mj, mr)
    print(f"{l:>5} {mj:>17} {mr:>15}", flush=True)

# ---- multihop battery ----
mh = {l: {"j": [], "raw": []} for l in EVAL_LAYERS}
for prompt, latent in MULTIHOP:
    ids = tok(prompt, return_tensors="pt").input_ids.to("cuda")
    lt = tok.encode(latent, add_special_tokens=False)[0]
    p = ids.shape[-1] - 1
    r = ranks_for(ids, [p], {p: lt})
    for l in EVAL_LAYERS:
        mh[l]["j"] += r[l]["j"]; mh[l]["raw"] += r[l]["raw"]

print("\nMULTIHOP median rank of latent middle hop (10 prompts)")
print(f"{'layer':>5} {'J-lens(3.6->3.8)':>17} {'raw logit lens':>15}")
mh_out = {}
for l in EVAL_LAYERS:
    mj, mr = int(np.median(mh[l]["j"])), int(np.median(mh[l]["raw"]))
    mh_out[l] = (mj, mr)
    print(f"{l:>5} {mj:>17} {mr:>15}", flush=True)

json.dump({"wikitext_raw": {l: agg[l] for l in EVAL_LAYERS},
           "multihop_raw": {l: mh[l] for l in EVAL_LAYERS},
           "wikitext": wiki_out, "multihop": mh_out,
           "published_36_on_36": {"wikitext_jlens": {8: 182, 16: 106, 24: 61, 32: 47},
                                   "wikitext_logitlens": {8: 11793, 16: 42247, 24: 28634, 32: 672}}},
          open("/workspace/raw38.json", "w"))
print("EVAL DONE")
