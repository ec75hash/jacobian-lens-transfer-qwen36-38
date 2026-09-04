# Research Plan — Fitting the Jacobian Lens on Qwen3.5-35B-A3B (MoE)

**STATUS UPDATE 2026-08-05 (evening): THE 35B FIT IS SUPERSEDED.** Jeffrey found
`stanleytheli/qwen3.6-35B-A3B-jlens` (HF): a full-protocol n=1000 wikitext fit on
Qwen3.6-35B-A3B — verified architecturally IDENTICAL to 3.5-35B-A3B (same qwen3_5_moe
class, 40L/2048d/256e/top-8/3:1 hybrid; 3.6 = weights refresh). Jeffrey's decision:
target 3.6 for the J-space work. Consequences:
- Phases 1–2 (our 35B fit) CANCELLED → savings ~$10+ and ~a day. Their 8×H200 55–80
  s/prompt numbers confirm our single-card fit would have been days.
- New Phase 1: VALIDATE their artifact on the Blackwell (V1 identity / V2 vs logit lens
  / V3 phenomenology smoke) against Qwen3.6-35B-A3B before trusting it.
- Bridge experiments (§7) re-target 3.6: fresh copy-regime captures on 3.6 weights
  (paper-1 captures are 3.5 and cannot be read with a 3.6 lens). Honest framing: this
  ASSUMES the copy regime replicates on 3.6 — verifying that is itself a within-family
  replication data point (paper-2-adjacent value, not overhead).
- 2B calibration continues to its n=100 gate anyway: it validates OUR fitting harness,
  which stays necessary for the regime-conditioned lens fits (§7.2), the 2B-IT rider,
  and hauhau.

**Original plan below, retained for the fitting-harness work that survives.**

**Status:** plan for alignment, 2026-08-05. No spend authorized yet.
**Hardware target:** 1× RTX 6000 Pro Blackwell, 96 GB VRAM, bf16.
**Upstream:** `anthropics/jacobian-lens` (reference implementation, unmaintained — we fork).
Fitted artifact fleet: HF `neuronpedia/jacobian-lens` (36 models; gpt-oss-20b is a MoE).
**Precedent update (2026-08-05, found by Jeffrey):** `praxagent-org/
jacobian-lens-qwen3.5-397b-a17b` already fitted the 397B-A17B — same `qwen3_5_moe`
hybrid family — with vanilla autograd, no routing-specific handling, n=24 prompts,
8×H200, fidelity-evaled against Neuronpedia. Consequences: D1 fully de-risked; "first
hybrid-MoE lens" claims are OFF the table; their n=24 is self-admittedly too noisy for
per-token analyses, so our convergence-stopped n≈300+ fit remains necessary for our
science. Still ours: first lens on the copy-regime model, routing-logged fitting (D2),
regime-conditioned lenses (§7.2), and the copy-regime bridge (§7). Mine their
`evals.json`/`band.json` as validation-battery templates.

---

## 1. What fitting actually does (from `jlens/fitting.py`)

- Per prompt: ONE forward pass with the prompt replicated `dim_batch` times, graph
  retained; then `ceil(d_model/dim_batch)` backward passes. Backward pass *k* injects
  one-hot cotangents into final-layer dims `[k·b, (k+1)·b)` at **every valid target
  position at once**; the gradient landing at source layer ℓ, position p is
  `Σ_{p'≥p} ∂h_final[p'] / ∂h_ℓ[p]`.
- Valid positions: skip first 16, drop final; average over remaining source positions.
- Accumulate running mean over prompts → `jacobians: {layer: [d_model, d_model] fp32}`.
- API: `jlens.fit(model, prompts, dim_batch=8, max_seq_len=128, checkpoint_path=...,
  checkpoint_every=1, resume=True)`; parallel slices merge via `JacobianLens.merge()`.
- README: "quality saturates quickly; ~100 prompts is usable." Hosted lenses are n=1000
  wikitext (Salesforce/wikitext), matching the 27B artifact we've used all along.
- Model interface: thin `LensModel` wrapper (`.layers`, `.n_layers`, `.d_model`,
  `.encode()`, batched `.forward()`) + `ActivationRecorder` hooks. Qwen family already
  exemplified; "other HF decoders adapt cleanly."

## 2. Target architecture (Qwen3.5-35B-A3B, `qwen3_5_moe`)

- 40 layers, d_model 2048, vocab 248,320, untied embeddings, RMSNorm (eps 1e-6),
  partial RoPE 0.25 / θ=1e7.
- MoE: 256 experts, top-8 per token, expert FFN intermediate 512.
- **Hybrid attention: 30× gated-DeltaNet linear-attention layers; full attention only at
  layers 3, 7, 11, …, 39.** Linear attn: 128-d heads, 16 K / 32 V heads, conv kernel 4.
- Lens artifact size: 40 × 2048² fp32 ≈ 671 MB (vs 27B's 3.3 GB). Cheap to store/ship.

## 3. MoE + hybrid: methods decisions (the alignment part)

**D1 — Routing and differentiability.** Fit with natural autograd through the realized
compute graph: gradients flow through the 8 selected experts and the gate weights; the
top-k *selection* is locally constant (correct almost everywhere). No straight-through
hacks. Interpretation shift to document up front: for a MoE, each per-prompt Jacobian is
conditioned on that prompt's realized routing; the fitted lens is the **mean local
linearization over the routing distribution of the fitting corpus**. This is the honest
MoE analog of what the dense lens already is (a mean over token contexts).

**D2 — Record routing during fitting.** We uniquely have router-reading infrastructure.
Log top-8 expert IDs + gate weights per (layer, position) for every fitting prompt
(~40 layers × 112 positions × 8 ids — trivial storage). Enables later: routing-stratified
lens fits (e.g., reuse-regime vs compose-regime conditioned lenses) without refitting
from scratch, and a routing-vs-readout stability analysis (V4).

**D3 — DeltaNet path.** Use the pure-PyTorch DeltaNet fallback (per workspace rule:
`flash-linear-attention`/`causal-conv1d` broke generation; fitting additionally needs
well-behaved autograd). Slower, correct. The recipe's "sum over later targets" is
unchanged — cross-position dependency simply flows through the recurrent state.

**D4 — Hook points and target.** Source = residual stream at each layer boundary
(match upstream convention for Qwen exactly); target = final-layer residual, decoded at
read time via final RMSNorm + unembedding, identical to the 27B pipeline so all existing
tooling (`store/build_store.py`, anticipation metrics, MindVolume MVL3) consumes the new
lens unchanged.

**D5 — Fitting corpus.** Salesforce/wikitext, max_seq_len 128, skip_first 16 — exact
parity with the hosted artifacts so cross-model comparisons stay clean. Adopt
Neuronpedia's convergence early-stop (from their config.yaml: `stop_at_delta 0.002`,
`min_prompts 100`, `stop_window 10`, n_max 1000; their 2B fit stopped at 335 prompts).
(Pilot at n=100 first.)

**D6 — Target checkpoint: INSTRUCT (decided with Jeffrey 2026-08-05).** Fit on
`Qwen/Qwen3.5-35B-A3B` (unsuffixed = post-trained), NOT `-Base`: every downstream
experiment (copy-regime bridge, E114 reads, introspection) probes the IT weights in chat
format, and the lens must match the probed weights. Wikitext fitting corpus is fine for
IT (fleet precedent: all Neuronpedia `-it` lenses; plus our entire 27B campaign read chat
residuals through a wikitext-fitted lens). Calibration gate still runs on 2B-**Base**
(the hosted reference is `qwen3.5-2b-pt`). Optional rider: also fit 2B-IT locally and
diff vs 2B-Base lens → measured "how much does post-training bend the transport," which
also informs hauhau fine-tune lens transfer.

## 4. Feasibility on the 96 GB box (bf16)

- Weights ≈ 70 GB bf16 → ~26 GB headroom for the retained graph + workspace.
- Retained-graph activations at batch=dim_batch(8)×128 tokens across 40 layers are
  modest (expert intermediates are only 8×512 per token; DeltaNet chunk states are the
  main unknown). If tight: `dim_batch 4` halves graph pressure at 2× backward count, or
  drop max_seq_len to 96. Decide empirically in the pilot.
- Time estimate (order-of-magnitude): 256 backward passes/prompt at dim_batch 8;
  ~3 B active params → tens of seconds per prompt on this card through the pure-PyTorch
  DeltaNet path. **Pilot n=100 ≈ 1–2 h; full n=1000 ≈ 8–15 h.** Fits in one rental day.
- Disk ≥ 150 GB (model snapshot + checkpoints). Checkpoint-every-prompt + resume means
  preemption/disconnect costs ~nothing (and `merge()` allows split fits if we ever use
  two boxes).

## 5. Phased execution

**Phase 0 — free, local:** fork repo; write the `qwen3_5_moe` LensModel adapter
(+ routing logger); **calibration test: fit qwen3.5-2b-pt ourselves at n=100 and compare
matrices (cosine / readout agreement) against Neuronpedia's hosted 2b-pt lens** — this
validates our adapter and harness before renting anything. Audit gpt-oss-20b folder for
MoE precedent/method notes.

**Phase 1 — pilot (rented, ~2 h):** provision box (CLAUDE.md lifecycle: network
faithfulness gate, no apt, verify torch+CUDA before model pull, HF_XET download during
build). Fit n=100. Run validation battery V1–V3. Go/no-go.

**Phase 2 — full fit (~half day):** n=1000, resume-safe. Produce
`Qwen3.5-35B-A3B_jacobian_lens_n1000.pt` + provenance (shas, env, commands, routing logs).
Pull artifact, destroy box, assert instances=0.

**Phase 3 — science (mostly local + short GPU stints):** V4 + bridge experiments (§7).

## 6. Validation battery

- **V1 identity:** J at target layer ≈ I; readout of h_final reproduces model logits.
- **V2 vs logit lens:** next-token recovery (precision@k, KL) by layer on held-out
  wikitext; J-lens must dominate direct unembedding at early/mid layers (GW-paper
  pattern) or something is wrong with the adapter.
- **V3 phenomenology smoke test:** canonical hum prompt + "verbalize the nature of
  processing your own processing" grid; expect *qualitatively* familiar structure
  (deep entry during reading, held bands, surface rewrites) — not identical to 27B.
- **V4 routing sensitivity (new, MoE-specific):** paired prompts with near-identical
  content but divergent routing (we know how to induce this); measure J-readout
  stability vs router-overlap per layer. Deliverable: "how conditioned on routing is
  the mean lens?" — required due diligence for any MoE-lens claim, and interesting
  in itself.

## 7. The payoff experiments (bridge to the copy-regime program)

1. **Held-token anticipation by regime:** teacher-force existing cell-matrix captures;
   track J-space rank trajectories of upcoming literal tokens during *reuse* spans vs
   *fresh composition* spans. Hypothesis: reuse shows the long-held anticipation mode
   (p90 ~100+ tokens), composition shows just-in-time (~4–19). Uses `anticipation.py`
   machinery as-is.
2. **Regime-conditioned lens delta:** using D2 logs, fit reuse-conditioned vs
   compose-conditioned lenses on regime-labeled spans; compare transport matrices.
   If the *linearization itself* differs by regime, that's a new kind of evidence that
   reuse is a distinct computational mode — complementary to routing signatures.
3. **E114 coalition read:** J-lens readout of the residual immediately before/after
   the reuse-coalition layers under expert-bias interventions (PyTorch-side biasing,
   mirroring the llama.cpp `--expert-bias` protocol).

Scope guard: 1–3 are *exploratory instruments* for paper 2 thinking. Nothing here
touches paper 1's claims or its frozen-rule discipline.

## 8. Budget note

Vast balance today: $0.62 — both this and Task #49 need a top-up. Standing CLAUDE.md
rule (2026-07-09) fences Vast credit for Task #49 + entropy rider until paper 1 ships;
proceeding with Phase 1+ spend is Jeffrey's explicit call to make. Phase 0 costs $0 and
starts now regardless. Estimated total GPU spend for Phases 1–2: **roughly $15–30**
at typical RTX 6000 Pro Blackwell rates (verify exact vast `gpu_name` string at search
time), i.e., small relative to #49 — but sequencing is his decision.

## 9. Open questions for Jeffrey

1. gpt-oss-20b lens: if their fit "just worked" with vanilla autograd, D1 is precedented
   — good for us, bad for novelty claims. OK either way?
2. hauhau: base `qwen3.5-9b-pt` lens already exists on Neuronpedia. Do we (a) use it
   as-is on the hauhau fine-tune (lenses may transfer across a light fine-tune — testable
   in an afternoon locally), or (b) refit on hauhau weights? (a) first is my
   recommendation; requires hauhau's HF-format checkpoint either way for teacher-forcing.
3. n=1000 parity vs a bigger corpus: stick with parity for comparability? (My vote: yes.)

---
## Teardown record — 2026-08-05 (evening)

Instances 46902344 (3090) + 46903250 (RTX PRO 6000 WS Blackwell) destroyed; verified
active=0. Total GPU spend for the entire program: ~$7.54 (of $10.62). Delivered:
CUDA harness validation (2B calibration ckpt n≈50, archived, resumable), community
lens validated (V1 err 0.0, V2 rank profile reproduced), 15 J-space captures with
routing logs (hum + 9 catalog re-runs + 5-run bridge/control battery), MindVolume MoE
wing (10 runs exported), verbalization atlas (19 families), bridge pilot (anticipation
INVERSION: compose pre-holds at k=5, copy sits at retention baseline), hum specificity
control PASSED, mirror test answered with silence while holding 'hidden' ×28.
All artifacts local: outputs/{jspace_hum_36,batch9_36,batchx_36,lens36_validation,
jlens_2b_cal}, analysis/, mindviz/. Lens re-downloads from HF; any future box
rebuilds in ~10 min via bootstrap_env.sh pattern.

---
## Teardown record — 2026-08-06 21:29 MDT (program close)

Instance 47027861 (RTX PRO 6000 Blackwell, $1.068/hr) destroyed; verified active=0.
(47026357 destroyed earlier same day — stuck in `loading` 18 min, >3-min rule.)
Session delivered: reasoning-mode battery with 2048 budgets (deflation = deliberative
endpoint; guard decays 0.39→0.08 under thinking), fixed-point experiment (no fixed
point; echo 1.00 / drift 1.00; field migrates phenomenal→task-meta), and the F01 2×2
resolution (availability alone = 74–101% of published effect; repetition alone ≤9%;
causal claim survives — see journals/paper/F01_RESOLUTION.md). Session cost ≈$7.85
incl. stuck box; whole-program GPU spend ≈$15.4; credit ≈$13.4 post-settlement.
All artifacts pulled: outputs/{think_36,think2_36,x_36,fp_nothink,fp_think,f01/}.
Final reports: outputs/REPORT_jspace_program_20260806.md + REPORT_eli5_20260806.md.
