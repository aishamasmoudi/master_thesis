# Plot Checklist — Master's Thesis Presentation

Every plot/figure referenced in `PRESENTATION PLAN.md` and `Master Thesis Presentation - DRAFT.pptx`,
one line per slide that needs one. Check off what you already have; anything unchecked needs to be run
before the talk. Diagrams that don't need data (architecture box diagrams, the "Where We Are" status
slide) are built directly in the .pptx and aren't listed here.

## PLOTS I NEED TO RUN MODELS FOR
- [ ] MSE: DINOv2 and V-JEPA2 on original dataset (60 frames, 15 epochs, 4 fps), evaluated on original test set
- [ ] MSE: original models vs. enriched-trained model (60 frames, 15 epochs, 4 fps), evaluated on enriched test set 
- [ ] MSE: original models vs. enriched-trained model (60 frames, 15 epochs, 4 fps), evaluated on original test set 
- [ ] MSE: same comparison, enriched training set subsampled to match original size (confound control -> remove ~2000 videos)
- [ ] MSE — DINOv2-small, with vs. without an added positional embedding in the pooler
- [ ] Benchmarks
- [ ] Everything DBSCAN-related (because of wrong cluster threshold)

## TO TRAIN
- [ ] DINOv2 on original dataset (60 frames, 15 epochs, 4 fps)
- [ ] V-JEPA2 on original dataset (60 frames, 15 epochs, 4 fps) 



---

## Context

- [x] *(no plot needed — architecture diagram only, already in the deck)*

## Part 1 — Enriching the dataset

**1.2 — Validating the dynamicity metric** (now split across 6 dedicated slides: "Validating the
Dynamicity Metric", "Sanity Checks", "Hyperparameter Optimization", "Hyperparameter Optimization:
Results", "Method Validation", "Selection of Cluster-Count Threshold...")

- [ ] Show examples of dynamic and low-dynamic videos — slide "What is a dynamic video?"
- [ ] **optional** Sanity-check plot: cluster count vs. truncation duration, for the 3 test videos (should be
      non-decreasing) — slide "Sanity Checks"
- [ ] ROC-AUC for choice of parameters — slide "Hyperparameter Optimization: Results"
- [ ] Violin plot for choice of parameters — slide "Hyperparameter Optimization: Results"
- [ ] Held-out generalization: same plot(s) on the 39 new dynamic (object-permanence) videos vs. the
      original 32 random videos — slide "Method Validation"
- [ ] Cluster-count histogram across the full 12,000-video dataset (used to justify threshold = 4) —
      slide "Selection of Cluster-Count Threshold..."
- [ ] Duration distribution per cluster count (the set of histograms with 5th-percentile markers) — same
      slide

**1.4 — Results** (slides "Does Enrichment Improve Alignment?" / "Controlling for Dataset Size")

- [ ] MSE: original model vs. enriched-trained model, evaluated on enriched test set
- [ ] MSE: same comparison, enriched training set subsampled to match original size (confound control)

## Part 2 — Optimization & Benchmarking

**2.1 — Optimization attempts** (slide "What We Tried to Improve Training")

- [ ] *(table of results, not a plot — confirm you have the actual numbers for each row, especially the
      frame-dropping consistency-check results across DINOv2/V-JEPA2/ResNet-50)*

**2.2 — Benchmarking** (slides "Benchmarking Results (I)" / "(II)" / "Frame-Based vs. Video-Based")

- [ ] Performance vs. number of parameters (DINOv2 small/base/large/giant) — **you have all 4 numbers,
      buildable now**
- [ ] Performance vs. publication/release date (all benchmarked encoders)
- [ ] Robust vs. normal (DINOv2 vs. DINOv2-robust, matched by size) — **you have all 6 numbers, buildable
      now**
- [ ] CNN vs. Transformer, frame-based (ConvNeXt vs. DINOv2/DINOv3/ViT)
- [ ] Frame-based vs. video-based (DINOv2 vs. V-JEPA2) — headline comparison, feeds into Part 3

## Part 3 — What does the model actually use?

**3.1 — Prediction error vs. video duration** (slide "Does the Model Use Sensory History?")

- [ ] MSE vs. duration, main test set — DINOv2 & V-JEPA2, each under full-clip / shuffled / last-frame, plus
      human last-frame baseline and split-half noise floor
      *(already built: `evaluation_plots_clean.ipynb`, "Plot" cell after the main bootstrap — re-run with
      current CSVs before using)*
- [ ] **(new)** Same MSE-vs-duration plot, on the SensoryHistory benchmark (same fixed set of target videos,
      each truncated to 0.25/0.5/1.0/2.0/4.0s) — the video-identity-controlled companion to the plot above,
      added to `PROJECT STORY.md` Part 2 Experiment 1 to address the duration/video-identity confound in the
      main test set *(not yet run — needs benchmark inference on the SensoryHistory condition)*

**3.2 — Low-dynamicity restriction** (slide "Controlling for Coverage Loss")

- [ ] Same plot, restricted to low-dynamicity videos (0-2 clusters)
      *(already built: `evaluation_plots_clean.ipynb`, "Plot: low-dynamicity subset only" cell)*

**3.3 — Architecture explanation** (slide "Why Is the Model Invariant to Shuffling?")

- [ ] *(no plot — box diagram only, already in the deck)*

**3.4 — Positional embedding follow-up** (slide "Next: Does Positional Embedding Matter?")

- [ ] MSE — DINOv2-small, with vs. without an added positional embedding in the pooler *(not yet run)*

**3.5 — Benchmark suite** (slide "Benchmark Suite: Object Permanence")

- [ ] MSE / predicted report rate on the object-permanence (appear → disappear → reappear) benchmark set,
      both models *(not yet run — new benchmark inference, per your note)*
- [ ] *(optional, only if you'll have them)* same for the other curated benchmark sets already in
      `load_trialTypes.py` (EventSegmentation, TemporalDecay/Retention, SensoryHistory duration sweep) —
      don't add slides for these unless you'll actually have the plots

**3.6 — Visualization** (slide "Visualization: Report Rate Through Time")

- [ ] Report rate vs. time-revealed, for the same example clip used in 3.5 — **only the categories
      relevant to that clip's narrative**, not all 12 *(not yet run)*

**3.7 — What's next (H2/H3)**

- [ ] *(no plot — designed-but-not-run experiments, text only)*

---

## Quick tally

| Status | Count |
|---|---|
| Already built (rerun with current CSVs before use) | 2 |
| Buildable now from numbers you already have | 2 |
| Not yet run | rest |

The two "already built" ones (3.1, 3.2) live in `evaluation_plots_clean.ipynb` — worth a fresh run to
confirm the CSVs they point at are current before the talk, since paths in that notebook's config cell
have been edited a few times this session.
