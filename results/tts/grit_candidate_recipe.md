# GRIT 3B — Proposed TTS Candidate Recipe

Derived from per-candidate correctness analysis on `TTS_Hard.json` (77 baseline-failure questions across VQA, Counting, OCR).

| C# | Image | Text | Rationale |
|---|---|---|---|
| C1 | original | original | baseline reference |
| C2 | original | hardcoded_paraphrase | one text variant |
| C3 | grayscale | hardcoded_paraphrase | best image aug + text diversity |
| C4 | edge_enhance | original | keep |
| C5 | grayscale | original | best performer |
| C6 | jpeg_recompress | original | keep |
| C7 | brightness_contrast | original | keep |
| C8 | rotation | original | keep |
| C9 | edge_enhance | model_paraphrase | keep |

## Change from original recipe

| Slot | Before | After | Reason |
|---|---|---|---|
| C3 | original + model_paraphrase | grayscale + hardcoded_paraphrase | weakest candidate (5.2%) replaced with strongest image aug paired with text variant |

## Per-candidate contribution on GRIT (baseline-failure questions)

| C# | Image | Text | VQA | Counting | OCR | Total |
|---|---|---|---|---|---|---|
| C1 | original | original | 0/29 (0%) | 0/30 (0%) | 0/18 (0%) | 0/77 (0%) |
| C2 | original | hardcoded_paraphrase | 0/29 (0%) | 4/30 (13%) | 3/18 (17%) | 7/77 (9%) |
| C3 | original | model_paraphrase | 0/29 (0%) | 1/30 (3%) | 3/18 (17%) | 4/77 (5%) |
| C4 | edge_enhance | original | 1/29 (3%) | 3/30 (10%) | 3/18 (17%) | 7/77 (9%) |
| C5 | grayscale | original | 3/29 (10%) | 4/30 (13%) | 5/18 (28%) | 12/77 (16%) |
| C6 | jpeg_recompress | original | 1/29 (3%) | 3/30 (10%) | 5/18 (28%) | 9/77 (12%) |
| C7 | brightness_contrast | original | 2/29 (7%) | 3/30 (10%) | 6/18 (33%) | 11/77 (14%) |
| C8 | rotation | original | 3/29 (10%) | 5/30 (17%) | 1/18 (6%) | 9/77 (12%) |
| C9 | edge_enhance | model_paraphrase | 3/29 (10%) | 3/30 (10%) | 2/18 (11%) | 8/77 (10%) |
| **Total** | | | **13/261 (5%)** | **26/270 (10%)** | **28/162 (17%)** | **67/693 (10%)** |

> C1 is 0% by construction — all questions in TTS_Hard had baseline_correct=False, and C1 is the baseline call.
