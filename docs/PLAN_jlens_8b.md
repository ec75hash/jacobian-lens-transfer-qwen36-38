# PLAN — Port the lens-chat rig to Qwen3-8B (drafted 2026-08-10, demo day)

Goal: the full instrument stack (temporal jstate + stability, while_reading,
salience baseline, sham/re-ask arms, web tool) on a small dense model where
turns run in seconds — making the queued experiments cheap — plus a second
model family point for every finding from the 35B nights.

## Assets already in hand
- Lens: `andyx10/jacobian-lens-qwen3-8b` (Qwen/Qwen3-8B instruct, layers
  0–34, d=4096, reference-implementation fit). UNAUDITED (0 downloads,
  n_prompts unknown) — our calibration sweep doubles as its audit; refit
  overnight on a cheap card if it proves thin.
- Model: Qwen/Qwen3-8B, 16 GB bf16. Dense, standard attention — no DeltaNet
  fallback, no (expected) cached-prefill bug.
- Code: everything above `JDict` is dimension-agnostic; `word_of`, vocab,
  im_end lookups already dynamic.

## Phase 0 — today (no work, just decisions)
- Demo runs on the 35B box (47385234). Credit top-up when convenient.
- End of day: pull chat_logs; DECIDE box fate. Cheapest 8B path is
  REPURPOSING this box (env already built; 96 GB fits 8B with luxury, or
  both models if we accept tightness). Alternative: $0.25–0.40/hr sidecar.
  Standing rule: no teardown without Jeffrey's word.

## Phase 1 — the port (~one evening)
1. Downloads: model + lens (~20 min total at demo-box speeds).
2. Loader shim: `qwen3_8b_lens.pt` stores `{J: layer->tensor, ...}`; adapt
   to `JacobianLens.from_pretrained` expectations (few lines).
3. **Calibration sweep** (the real step): recovery-defect + per-layer l1
   across all 35 lens layers using the existing jstate machinery.
   Deliverables: empirical deep band, late zone, JS_LAYERS pick, frozen
   per-layer l1 table, and a verdict on the lens's fit quality.
   Per-model calibration is mandatory (35B journal rule: nothing transfers
   across architectures uninspected).
4. Config port: parametrize DEEP/LATE (currently lens_tool module
   constants) + JS_LAYERS/JS_L1 via server args, so one codebase serves
   both models. Provisional bands if sweep is deferred: deep 18:30,
   late 30:33 (proportional from 35B).
5. Smoke suite via scripted client: plain turn, consult_lens + jstate,
   while_reading, last_turn (covers field), web_search, /reask, sham.
6. Perf pass: measure tok/s; TEST multi-token injection on dense attention
   — if the DeltaNet shape bug is absent, retire the single-step loop for
   this model (order-of-magnitude faster tool turns). Keep single-step as
   fallback flag.

## Phase 2 — science that was waiting for fast turns
Priority order:
1. **Arms A–D** (no-tool / real / shuffled / other-run payloads): the
   flagship controlled question — does the model benefit from information
   about ITSELF specifically? Sham infra exists; needs the shuffled +
   cross-run donor variants and an N-per-arm harness.
2. Temporal/entropy replication across many turns (burst-vs-plateau,
   regional mass-entropy coupling, syntax-parking) — distributions, not
   anecdotes.
3. Recursive self-reading convergence (S0→S1→S2 distances, J-CoT r_t).
4. Cross-model comparison, 8B vs 35B, fresh calibration each side: does
   stance-outranks-topic stability replicate? Does the denial cluster?
5. Optional, clearly-labeled steering arm (carrier/residual injection) —
   never blended with introspection runs.
Note: 8B-instruct vs J-CoT's 8B-Base — same architecture, different
post-training; comparisons to their numbers stay caveated.

## Phase 3 — the Fable audit (promised 8-09, protocol TBD)
Adapt the games to a model without lens access: evidence-vs-question
coupling probes, sham "measurements" of Fable presented as real,
report-vs-access calibration scoring, Qwen-with-lens as the instrumented
comparison. Needs its own short protocol doc before running.

## Hygiene carried forward
- Turn lock stays (the cross-session leak from demo day is a finding AND
  a fixed bug — write it up eventually).
- Per-model frozen calibrations; contaminated-turn flags respected;
  pull-before-teardown always.
