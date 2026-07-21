# OCR Capability Roadmap: Reconciling the Aspirational Pipeline with the Research Prototype

## Context

`ondc-intelligence` (the production FastAPI service) has one live capability, `product.extract`, which sends full product images straight to Gemini for JSON extraction. Its capability registry already anticipates a `document.ocr` capability (mentioned only in a docstring, not actually registered) but nothing has been built for it.

Separately, a much more mature-than-expected research codebase exists at `OCR Prototype/`: a 5-tier cost-ordered OCR escalation pipeline (classical OCR → local VLM → paid cloud vision tiers) plus deterministic barcode decoding and a heuristic/ML structuring stage, built specifically against the ONDC grocery product schema. There's also a prior design artifact ("DigiCatalog OCR Pipeline") describing an aspirational 7-stage architecture — preprocessing, classical-OCR-first field resolution, patch-level (not full-photo) LLM escalation for only the hardest fields, hard spend ceilings, and human-review/quarantine paths.

The user's request was to fully understand both codebases and produce a roadmap reconciling the aspiration with the actual code — explicitly **not** an integration plan yet, and **not** code changes. Two decisions were made during clarification:
1. **Deliverable is a roadmap/architecture document**, not an ondc-intelligence integration plan (that decision — new `document.ocr` capability vs. replacing `product.extract`'s engine vs. hybrid — is deliberately deferred to a future planning round).
2. **Cloud-only for production**: the prototype's local-VLM tier (`mlx_vlm`, Apple-Silicon-only) is dropped from any production path; it stays a dev/eval-only tool.

This plan's "execution" is to write the roadmap below into a durable markdown file in the repo (rather than only living in this conversation), so it can be referenced in the future planning round.

## The Roadmap

### 1. Stage-by-stage reconciliation (aspiration vs. prototype)

| # | Target stage | Status | Where |
|---|---|---|---|
| 0 | Preprocessing (format/integrity, orientation+deskew, denoise/contrast, resolution clamp, product-region crop, `quality_score`) | **Built** | `OCR Prototype/preprocessing.py` |
| 0b | Background removal (optional) | **Built**, optional | `OCR Prototype/bgremove.py` |
| 1 | Classical OCR / rules resolve most fields free | **Built, quality-capped** | Tier 0 cache (`ocr/cache.py`) → Tier 1 RapidOCR (`ocr/rapid.py`) → heuristic (`stage2/heuristic.py`) → barcode (`barcode.py`). Stage-1 field recall 42%, Stage-2 structured F1 33%. |
| 2 | Vision-LLM sees only a small cropped **patch**, only for unresolved fields | **Missing** — every tier gets the whole image | `ocr/router.py` tiers 2–4 |
| 3 | Hard per-record/global spend ceilings before paid calls | **Missing** | No budget module exists; tiers are gated only on credential presence |
| 4 | Cost-ordered escalation routing | **Built, strong** | `ocr/router.py` — the soundest piece of the prototype |
| 5 | Structuring OCR/VLM output into the target schema | **Built, weak** | `stage2/heuristic.py` (reliable, deterministic) + `stage2/layoutlm.py` (weak, "useless until fine-tuned" per its own docstring, F1 as low as 2% on Mrp despite a trained checkpoint existing) |
| 6 | Human-review / quarantine / reject paths | **Missing** | No such routing concept exists today |

**Takeaway**: the hard research question (does tiered OCR work at all) is substantially derisked — Stages 0, 1, 4, 5 exist and are benchmarked. What's missing is production discipline: patch-level escalation, spend ceilings, and review/quarantine routing.

### 2. Cloud-only tier ladder — recommendation

Local VLM is dropped (per user decision), leaving:

```
Tier 0: SHA-256 checksum cache        (free, unchanged)
Tier 1: RapidOCR                      (free, unchanged, runs in any Linux container)
Tier 2 (top): a single paid cloud vision tier
```

**Recommendation: consolidate on Gemini as the single paid vision tier; do not add Anthropic/Google Vision credentials to production.**

Reasoning:
- `ondc-intelligence` already standardizes on Gemini (`GEMINI_API_KEY` is the only LLM credential wired into `app/config.py`; `GeminiClient` in `app/agents/gemini.py` is mature and retry-tuned). Each additional provider means new secrets, a new `/v1/health` failure mode, and a new billing relationship — real ongoing cost, not just one-time integration cost.
- The eval doc's model comparisons (Qwen3.5, Gemma-4, PaddleOCR-VLM, Chandra) were all run via a **local** mlx backend — a different serving path than production Gemini. They say nothing yet about how cloud Gemini specifically compares to Claude/Google Vision on this dataset. That gap should be closed with a same-serving-path bake-off in Phase B, not assumed away.
- Google Cloud Vision's real strength is dense `DOCUMENT_TEXT_DETECTION` at very low fixed cost — a different value proposition than a general vision-LLM call. If it earns a place at all, it's better framed as a "Tier 1.5" cheap dense-OCR fallback between RapidOCR and the Gemini escalation tier, not as a peer to Gemini/Claude. Worth A/B testing in Phase B, not assuming.
- Keep `ocr/claude_vision.py` and `ocr/google_vision.py` in the prototype's **offline eval harness** as quality-ceiling reference points even after dropping them from the production-eligible ladder — cheap to keep as a benchmark, expensive to keep as live credentials.
- This is a starting recommendation, not a permanent one: if Phase B's bake-off shows Gemini meaningfully underperforms on this task, that's the trigger to revisit.

### 3. "Patch, not photo" — is it premature?

Today every tier gets the whole preprocessed image, not a per-field crop. Building true patch-level escalation requires: per-field region hypotheses (RapidOCR already returns line-level bounding boxes — the missing piece is turning "field X unresolved" into "candidate box for field X"), a field→region policy (some fields like MRP/FSSAI have strong positional priors, others like Brand/Product Name are diffuse), a crop-and-re-route step in `router.py`, patch-scoped prompts, and a whole-image fallback for diffuse/no-signal cases.

**Recommendation: defer this. It's premature while Stage-2 structuring is at 33% F1** — the pipeline doesn't yet reliably know *which* fields are missing/wrong, so precision-cropping on top of an untrustworthy trigger risks wasted engineering. The stronger near-term lever is closing the 42%→33% drop from OCR recall to structured F1 (i.e., fixing the heuristic/parsing logic, which needs no new infrastructure). Patch-level escalation should land in a later phase (after Phase B), once the "field unresolved" signal itself is trustworthy. One cheap thing worth doing now as groundwork: start **recording** (not yet acting on) missing-field + bounding-box data during Phase B's eval runs, so the eventual patch-crop design is evidence-based.

### 4. Missing scaffolding: spend ceilings and human-review/quarantine

Neither exists today. Proposed conceptual placement only (not implementation):
- **Spend ceilings**: a check-and-reserve gate immediately before any paid-tier call in `router.py` — extending the existing credential-presence gate (`ocr/google_vision.py`/`claude_vision.py`'s `available()` pattern) to also check remaining per-record and global budget, refusing the call (not auditing after the fact) when exceeded.
- **Human-review/quarantine/reject**: every record should exit the pipeline in one of (at least) three states — resolved, needs-human-review (fields unresolved/low-confidence after exhausting eligible tiers, or after a spend-ceiling refusal), or quarantined/rejected (image structurally unusable). This matters because `ceiling.py`'s finding that only 58% of a 200-product sample even had a back-panel photo means many "failures" are input-availability limits, not OCR failures — the outcome state needs a reason code to distinguish these, or a human-review queue fills with unresolvable cases.

### 5. The honest quality tradeoff — don't blindly replace `product.extract`

The eval doc is unambiguous: **direct Image→JSON (one strong model call) currently beats OCR→JSON tiered pipelines on F1** (best Image→JSON F1 70.2 vs. best OCR→JSON F1 68.6) — which is exactly what `product.extract` already does today. The tiered pipeline has a proven *cost* story, not yet a proven *quality* one, and even the cost/quality numbers used Gemma/Qwen locally, not Gemini.

**Do not propose replacing or routing production traffic through the tiered pipeline until it's validated on a shared eval harness against a pre-defined quality bar.** Concretely, before any Phase C integration decision:
1. Reconcile or cross-walk the two eval harnesses/ground truths (`OCR Prototype`'s `Ground Truth.csv`/`evaluate_ocr.py`/`eval_stage2.py` vs. `ondc-intelligence`'s `evals/evals_image2schema.py`/`evals/gt_match.py`) so numbers are comparable.
2. Re-run the Image→JSON vs. OCR→JSON comparison with **Gemini** in both roles (not local Qwen/Gemma), since that's the provider actually in production.
3. Define the quality bar in advance (e.g., "tiered pipeline's F1 within N points of `product.extract`'s production F1, at under M% of per-record cost") — a decided gate, not a post-hoc judgment call.

### 6. Phased roadmap (outcome-gated, not calendar-dated)

**Phase A — Cleanup, reproducibility, cloud-only consolidation** (prototype-only, no ondc-intelligence changes)
Exit criteria: `requirements.txt` reproducibly installs everything actually used (rapidocr, anthropic, google-cloud-vision, torch/transformers if LayoutLM kept, pyzbar's native zbar dependency documented); local-VLM path clearly marked dev/eval-only and excluded from the production tier list; `router.py`'s default ladder reflects §2's recommendation; hygiene items (§7) done; repo is git-safe if ever committed.

**Phase B — Prove Stage-2 structuring reliability** (still prototype-only)
Exit criteria: a written quality bar (macro F1 or per-critical-field recall target) is met on the existing eval set, expressed relative to the achievable ceiling (58% back-panel availability), not against 100%; the Gemini-as-paid-tier bake-off from §2 is run and documented; a decision is made on LayoutLM (invest to make it measurably help, or explicitly shelve it — no ambiguous half-used state).

**Phase C — Decide and design the `ondc-intelligence` integration shape** (deferred decision point, not pre-decided here)
To be planned separately once Phase B's evidence exists. Open questions flagged, not answered: new `document.ocr` capability vs. replacing `product.extract`'s engine vs. hybrid fallback; whether `CapabilityStatus.BETA` (already defined in `app/models/enums.py`, unused today) is the rollout vehicle; how the tiered pipeline's dependencies fold into the production Dockerfile; where spend-ceiling/review-routing concepts (§4) concretely wire into `app/core/errors.py`/`PipelineStage`; whether patch-level escalation (§3) is built now or deferred further.

**Phase D — Implementation.** Out of scope entirely; its own plan once Phase C's shape is decided.

### 7. Near-term hygiene (worth doing regardless of bigger decisions)

- Delete stray 0-byte files `OCR Prototype/--model` and `OCR Prototype/--port` (artifacts of a misinvoked shell command).
- Resolve `OCR Prototype/image with barcode` (scratch note) — delete or relocate/rename clearly.
- Decide the fate of `OCR Prototype Backup/` — confirmed to predate Stage-2 entirely; delete or document why it's kept.
- Fix `requirements.txt` (Phase A exit criterion, repeated here for visibility) and document `pyzbar`'s native `zbar` system dependency separately, since it can't be captured in `requirements.txt`.
- If this ever goes under git: verify `.gitignore` actually excludes `Dataset/` (20k+ images), `ocr_cache.sqlite`, `barcode_cache.sqlite`, `bgmask_cache.sqlite`, and `venv/` — don't assume, check.
- Consider whether `layoutlmv3_finetuned/` (a model checkpoint) belongs in git history at all vs. a model registry/artifact store.

### Critical files referenced

- `OCR Prototype/ocr/router.py` — escalation ladder; center of Phase A tier-list surgery and Phase C's future spend-ceiling/patch hooks.
- `OCR Prototype/stage2/heuristic.py` + `stage2/pipeline.py` — Phase B's structuring-reliability work; currently the weakest link relative to Stage-1 recall.
- `OCR Prototype/requirements.txt` — Phase A reproducibility fix.
- `ondc-intelligence/app/core/registry.py` + `app/agents/img2schema_agent.py` — the production baseline any Phase C decision must be validated against.
- `OCR Prototype/Evaluation/` (`evaluate_ocr.py`, `eval_stage2.py`, `ceiling.py`) + `ondc-intelligence/evals/gt_match.py` — the two eval harnesses Phase B needs to reconcile.

## Execution (what happens if this plan is approved)

Write this roadmap to a new markdown file — proposed location: `OCR Prototype/ROADMAP.md` (co-located with the prototype it describes, since Phases A/B are prototype-only work and Phase C explicitly punts on the ondc-intelligence question). No other files are touched; no code changes. This is purely documenting the discussion so it survives past this conversation.

## Verification

Read back the written file to confirm it renders correctly as markdown and matches this content.
