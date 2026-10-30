**Research account:** this copy is published at [ec75hash/jacobian-lens-transfer-qwen36-38](https://github.com/ec75hash/jacobian-lens-transfer-qwen36-38). The [original repository](https://github.com/jeffreywilliamportfolio/jacobian-lens-transfer-qwen36-38) and its history remain available. [Archived paper v1.1](https://zenodo.org/records/22133736) · [Copy provenance](PUBLICATION_PROVENANCE.md).

# Jacobian lens transfer: Qwen3.6-27B to Qwen3.8-27B

Paper source, figures, and evaluation code for the preprint

**Survival of the Fitted: Cross-Checkpoint Transfer of a Jacobian Lens from Qwen3.6-27B to Qwen3.8-27B**
Jeffrey W. Shorthill ([ORCID 0009-0004-3954-2752](https://orcid.org/0009-0004-3954-2752)), independent researcher.
Version 1.1, 27 August 2026. Preprint, not peer reviewed. CC BY 4.0.

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22133735.svg)](https://doi.org/10.5281/zenodo.22133735)

- Concept DOI (all versions): [10.5281/zenodo.22133735](https://doi.org/10.5281/zenodo.22133735)
- Version 1.1 record: [10.5281/zenodo.22133736](https://doi.org/10.5281/zenodo.22133736)
- Data (captures, rank tables, logs): [huggingface.co/datasets/ec75hash/jacobian-lens-transfer-qwen36-38](https://huggingface.co/datasets/ec75hash/jacobian-lens-transfer-qwen36-38)

## What the paper reports

A Jacobian lens fitted on Qwen3.6-27B (the published Neuronpedia artifact, 1,000 WikiText-103 prompts) was applied without refitting to Qwen3.8-27B, a successor checkpoint released 113 days later with matching architecture and tokenizer. Two evaluations were run under one protocol on both checkpoints: recovering an unstated country in 40 two-hop prompts, and ranking the teacher-forced next token at 700 positions in WikiText-103 passages. The latent-entity readout survived the transfer with a depth-dependent cost: median target rank rose from 4 to 17 at layer 48, while at layers 24 to 36 the successor had lower ranks (38 against 121 at layer 24). The next-token readout showed paired rank inflation of 1.2 to 1.3 times at mid-network layers and about 2 times by layer 48; the raw logit-lens readout shifted by a similar 1.2 to 1.3 times at mid-network layers, so the lens-specific cost is concentrated near layer 48. Projecting out lens-derived directions for *paradox* at layers 18 to 47 removed the word from every ablated generation on both checkpoints, with reasoning mode enabled and disabled. An exploratory follow-up applied a second fitted lens (Qwen3.6-35B-A3B) unchanged to Qwen3.5-35B-A3B Instruct and to a HauhauCS Qwen3.6-35B-A3B derivative on the same 40 prompts. The paper concludes that a fitted lens can remain informative across a nearby, architecture-compatible version change without preserving its original calibration, and that transfer error should be measured before readout differences are attributed to the model.

## Layout

| Path | Contents |
|---|---|
| `paper/lens_transfer_v1.1.tex` | LaTeX source of the published v1.1 (byte-identical to the Zenodo source zip) |
| `paper/refs_revised.bib` | Bibliography (the tex calls `\bibliography{refs_revised}`) |
| `paper/lens_transfer_v1.1.pdf` | The published PDF (byte-identical to the Zenodo v1.1 PDF) |
| `paper/figs/fig_concept.pdf`, `fig_board.pdf`, `fig_ranks.pdf` | The three figures the tex includes (Figures 1 to 3) |
| `code/` | The evaluation scripts that produced the 27B results, as run |
| `code/figures/` | Scripts that built `fig_concept.pdf` and `fig_board.pdf` |
| `code/exploratory_35b/` | Driver, decoding code, capture example, build script, and prompt files for the exploratory 35B follow-up (paper Sections 2.4 and 3.3, Table 2) |
| `docs/` | Two planning notes from the surrounding Jacobian-lens program (see below) |
| `CITATION.cff`, `LICENSE` | Citation metadata; CC BY 4.0 |

## Rebuilding the PDF

The preamble uses `fontenc`/`lmodern` (no `fontspec`), so the engine is `pdflatex`; references use `natbib` and `bibtex`.

```sh
cd paper
pdflatex lens_transfer_v1.1
bibtex   lens_transfer_v1.1
pdflatex lens_transfer_v1.1
pdflatex lens_transfer_v1.1
```

or `cd paper && latexmk -pdf lens_transfer_v1.1`. The tex sets `\graphicspath{{figs/}}`, so figures resolve from `paper/figs/`. Checked 2026-09-03 with TeX Live 2025: builds without warnings; the rebuilt PDF has 8 pages and its extracted text is identical to the Zenodo v1.1 PDF. Build products are git-ignored.

## Code

All scripts are the as-run copies from the 2026-08-14 (27B) and 2026-08-15 (35B) sessions and are byte-identical to the copies in the HF dataset. Paths inside them point at the Vast.ai working tree (`/workspace/...`) and were not rewritten; adjust them to run elsewhere.

### 27B study (`code/`)

Requirements: the `jlens` reference implementation ([anthropics/jacobian-lens](https://github.com/anthropics/jacobian-lens)), `transformers`, `torch`, `datasets`; models in bf16 on one 96 GB GPU.

| Script | Role in the paper |
|---|---|
| `eval36_big.py`, `eval38_big.py` | Per-layer median target rank, Jacobian and raw readouts, on the home model (3.6) and the successor (3.8): 40 two-hop prompts at the final position plus 100 WikiText-103 passages at 7 positions. Behind Table 1 and Figure 3. The two files differ only in the model path and output name. |
| `raw36.py`, `raw38.py` | Same evaluation, but dumps the full per-prompt and per-position rank lists used for the paired sign tests and bootstrap intervals. |
| `mh_items.py` | The 40-prompt two-hop set with first-token targets. |
| `escher_ablate.py` | Single-concept intervention: Gram-Schmidt-orthogonalized pullbacks of the *paradox* token forms projected out at layers 18 to 47 during generation and capture, crossed over model, clean/ablated, and reasoning mode (8 cells). |
| `ab_temp07.py` | Temperature-0.7 sampling check on the home model for the two behavioral prompt pairs, with full-field capture. |
| `hum38.py` | Exploratory held-content example: fixed self-description prompt with forced prefix on Qwen3.8-27B, full-field capture, never-said word masses at layers 39 to 55 and 56 to 60. |
| `italy_field.py` | Prefill-only full-field capture of the flagship two-hop prompt on both models; input to `fig_board.pdf`. |

### Figures (`code/figures/`)

| Script | Output |
|---|---|
| `build_concept.py` | `paper/figs/fig_concept.pdf` (Figure 1). Needs matplotlib and a CJK font for the Chinese token. |
| `build_board.py` | `paper/figs/fig_board.pdf` (Figure 2) from the `italy_{36,38}_field.npz` captures in the HF dataset (`figure_capture/`). Needs the Qwen3.8-27B tokenizer. |

The script that drew `fig_ranks.pdf` (Figure 3) was not preserved. The figure itself is included, and the numbers it plots are the per-layer medians in `eval36_big.json` and `eval38_big.json` in the HF dataset.

### Exploratory 35B follow-up (`code/exploratory_35b/`)

Runs through llama.cpp GGUF captures rather than the `jlens` PyTorch path.

| File | Role |
|---|---|
| `s1s2_batch.py` | Evaluation driver: latent-entity rank per layer on the 40 two-hop prompts (S1) and next-token rank on one fixed paragraph (S2), Jacobian versus raw readout. Behind Table 2. |
| `lens_decode.py` | Readout decoding from GGUF: `h @ J[l].T`, then the target model's own output norm and unembedding read from the same GGUF. |
| `capture_activations.cpp` | Patched llama.cpp example that dumps per-layer residuals during prefill. |
| `build_llama.sh` | Builds llama.cpp at the pinned commit `1772701f9` with the example above (CUDA, sm_120). |
| `mh40.json`, `mh40.tsv` | The same 40 two-hop prompts in the driver's input formats. |
| `s2.tsv` | The fixed paragraph for the 212-position surface-token check. |

Results, run logs, and the provenance note (model, lens, and binary hash prefixes) are in the HF dataset under `exploratory_35b/`.

## Docs

`docs/PLAN_jlens_35b_a3b.md` and `docs/PLAN_jlens_8b.md` are planning notes from the surrounding Jacobian-lens program (fitting a lens on Qwen3.5-35B-A3B; porting the lens-chat rig to Qwen3-8B). They predate this study and are included for context on where the lens-transfer question came from; they are not the paper's protocol.

## Not included

Get these from the HF dataset [ec75hash/jacobian-lens-transfer-qwen36-38](https://huggingface.co/datasets/ec75hash/jacobian-lens-transfer-qwen36-38) or the model repositories named in the paper.

- Raw captures and full-field `.npz` files: `escher_ablation/`, `temperature_check/`, `figure_capture/`, and `hum38_field.npz` (11.1 MB, also over this repository's 10 MB file limit).
- Result tables: `eval36_big.json`, `eval38_big.json`, `raw36.json`, `raw38.json`, and the 35B `s1s2_*.json` files.
- 35B run logs and `exploratory_35b/PROVENANCE.md`.
- Model weights: Qwen/Qwen3.6-27B, Qwen/Qwen3.8-27B, Qwen/Qwen3.5-35B-A3B, and the HauhauCS Qwen3.6-35B-A3B derivative.
- Lens artifacts (`.pt`): `neuronpedia/jacobian-lens` (`qwen3.6-27b/.../Qwen3.6-27B_jacobian_lens_n1000.pt`) and `stanleytheli/qwen3.6-35B-A3B-jlens` (`lens.pt`).
- The generator for `fig_ranks.pdf` (not preserved; see above).

## License

CC BY 4.0 (see `LICENSE`) for the paper text, figures, and code in this repository.

## Citation

See `CITATION.cff`. Please cite the version DOI 10.5281/zenodo.22133736 (v1.1) or the concept DOI 10.5281/zenodo.22133735 (all versions).
