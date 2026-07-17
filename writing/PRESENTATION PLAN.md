# Presentation Plan — Dynamic Object Perception: Human-Model Alignment

Built from: MAIN WRITING, PART 1 - Dynamicity, PART 1 - ENCODERS, PART 1 - Further optimization,
PART 1 - embedding, PART 2, Models to run.pdf, plus this session's code work
(notebook analyses, VideoMAE segment implementation), calibrated with Aicha.

Legend used throughout: **[HAVE]** = data/plot already exists · **[PENDING]** = planned, not yet run ·
**[CUT]** = don't present.

---

## 0. Context (1-2 slides)

**Story beat:** why this project, framed as a testable question, not just "we built some models."

- Motivation: dynamic object perception — how the brain continuously updates object representations
  as a scene evolves. Computational models let us test this at scale with falsifiable predictions.
- The task: ~12,000 short egocentric clips (Ego4D-derived, 0.1–15s), each with human reports of which
  of 12 object categories (Cup, Knife, Chair, Person, Car, Bike, Dog, Cat, Table, Book, Plant, Bed) were
  perceived. Soft labels in [0,1] = fraction of raters who reported each category.
- The model paradigm (**[HAVE]**, one clean diagram is enough): frozen pretrained encoder (DINOv2 /
  V-JEPA2 / others) → trainable attentive pooler (self-attention refine + single learned query
  cross-attention pool) → trainable linear classifier → 12 predicted report rates. Only the pooler +
  classifier are trained; this is a *probing* study, not fine-tuning — the point is to ask whether the
  frozen representation already contains what's needed.
- Metric: MSE between predicted and human full-clip report rates on held-out test data.
- The three-part structure of the thesis (mirrors MAIN WRITING's own framing): (1) improve alignment via
  data, (2) understand what drives the model's predictions, (3) [future work, not this talk] design new
  human experiments from what's learned.

**Don't include:** the full Part 3 (new behavioral experiments) — MAIN WRITING frames this as future work
beyond what you've done. Mention it as a closing "where this goes next," not a body section.

---

## Part 1 — Enriching the dataset with dynamic videos

### 1.1 Motivation slide

- The training set is skewed toward static, short clips: of 11,366 train videos, 6,840 (~60%) are
  "static" (≤1 DBSCAN cluster) (**[HAVE]**, from PART 1 - Further optimization). Static videos are
  heavily skewed toward *short* durations too (median 1.9s).
- The hypothesis: a short static clip is close to a single-image recognition task — no real "sensory
  history" for the model to integrate over. If we want to actually test whether the model does temporal
  reasoning, we need more of the dataset to genuinely require it.
- Framing line for the slide: *"If most of our training data doesn't require temporal integration to
  solve, we can't expect — or cleanly measure — a model that does it."*

### 1.2 How: the DBSCAN dynamicity pipeline

This is a methodology slide — keep it to the validated pipeline, not the exploratory dead ends.

**What DBSCAN is and how it produces a cluster count** (worth a short sub-slide or a couple of build
lines — the audience needs this to trust the metric):

- DBSCAN (Density-Based Spatial Clustering of Applications with Noise) is a *density-based* clustering
  algorithm: it groups points that are packed closely together, and labels points that don't belong to
  any dense region as noise, rather than forcing every point into a cluster.
- Two parameters: **epsilon (eps)** — two points count as "neighbors" if they're within eps of each
  other in embedding space; **min_samples** — the minimum number of neighbors a point needs to anchor a
  cluster. Larger eps → easier to merge points into clusters (fewer, bigger clusters). Larger min_samples
  → stricter, harder to form a cluster.
- Applied here: for one clip, sample frames every 200ms, embed each frame's CLS token with a pretrained
  vision encoder (DINOv2/DINOv3), then run DBSCAN once *per clip* over that clip's own set of frame
  embeddings. The output is simply how many distinct dense groups of similar frames DBSCAN finds.
- The intuition this cashes out: in a static clip, most frames look similar to each other, so they
  collapse into one (or zero, if too few/noisy) cluster. In a dynamic clip, frames spread across visually/
  semantically distinct states as the scene changes, so they split into more clusters. Cluster count is
  therefore a proxy for how much a clip's content changes over time — no manual labeling required.

**Validation, in order** (**[HAVE]** — this is your strongest, most complete methodological result):

  1. Curated testbed: 32 highly-dynamic benchmark videos vs. 32 near-static random videos.
  2. Hyperparameter search over (eps, min_samples); best pair (25, 2) separates the two groups with
     ROC-AUC ≈ 0.95, Cohen's d ≈ 2, p < 0.05 (Welch + Mann-Whitney).
  3. Held-out generalization check: same hyperparameters applied to 39 new dynamic (object-permanence)
     videos — separation holds.
  4. Sanity check: truncating a video at increasing durations never *decreases* its cluster count
     (checked on 3 videos) — cluster count behaves monotonically as expected.
  5. Threshold selection: ran DBSCAN on the full 12,000-video dataset, examined the cluster-count
     histogram and the duration distribution per cluster count, settled on **threshold = 4 clusters** to
     call a clip "dynamic" (this is a different, coarser threshold than the "low-dynamicity = 0-2
     clusters" split used later in Part 3 — worth being explicit on the slide that these are two separate
     cutoffs used for two separate purposes: *generation* threshold vs. *analysis* filter).
- **[CUT / mention only if asked]:** the early exploratory phase in PART 1 - embedding (cosine similarity,
  L2 distance, second-difference "acceleration" metrics on 2 example videos, and the whole composite
  z-score / supervised-proxy-classifier / diversity-aware-selection discussion). None of that machinery
  is what you actually shipped — DBSCAN cluster count alone, with a simple threshold, is. Including the
  abandoned exploration would dilute a genuinely clean, validated result. One sentence acknowledging you
  explored richer scoring first and found the simple threshold sufficient is enough, if you want it.

### 1.3 Data generation

- Generated 2,000 dynamic clips for train + 200 for test (n_clusters ≥ 4, minimum 3.2s duration — the
  5th percentile duration for 4-cluster clips, chosen to make dynamic clips easier/faster to find), with
  a 10-try tolerance per source video before moving on. (**[HAVE]**)
- Collected human report data on the new clips — confirmed done. (**[HAVE]**)

### 1.4 Results — the actual payoff slide

- **[PENDING — scheduled to run]**: MSE of original model vs. model retrained on enriched dataset
  (12,000+2,000 train / 602+200 test), evaluated on the enriched test set. This is the one number this
  whole section is building to — make sure the slide title states the comparison explicitly ("does adding
  dynamic clips improve alignment on dynamic clips specifically") so the placeholder slide still
  communicates the intended finding shape while the run is in progress.
- **[PENDING, and flag verbally as a planned control, not yet run]**: the same comparison but subsampling
  the enriched training set down to the original size, to remove "just more data" as a confound. Say this
  explicitly in the talk — it's a good sign of rigor to name the confound and the control even before you
  have the number.

---

## Part 2 — Optimization & Benchmarking

### 2.1 Optimization attempts (brief — mostly negative/null results, don't over-invest slides here)

One slide, table format, is enough: method → hypothesis → what happened → why (in your reduced-scale
pilot, not necessarily generalizable).

| Method | Result | Why (your own reasoning, PART 1 - Further optimization) |
|---|---|---|
| Tubing (spatial mask on pooler attention, V-JEPA-style) | Hurt performance | You're masking the *pooler's* attention, not the encoder's input — the encoder still sees everything, so masking only throws away information rather than forcing better representations the way it does during V-JEPA's masked self-supervised pretraining. Aggressive ratios (V-JEPA/VideoMAE use 90%) don't transfer to a regression task with few frames. |
| Random frame dropping | Two-stage story, worth telling in full: an early version dropped from *all* T positions including already-padded ones, which produced catastrophic results (padding budget could eat the whole drop ratio). Fixing it to only drop from real, non-padded frames gave similar-to-improved performance. Later, once DINOv2, V-JEPA2, and ResNet-50 were each grid-search-optimized and frame dropping was added on top as a consistency check, it consistently helped DINOv2 and V-JEPA2 but was catastrophic for ResNet-50 — dropped from the final recipe for simplicity/consistency across encoders, given time constraints. | Good story for the talk: shows an augmentation bug caught and fixed, then a legitimate cross-encoder inconsistency that motivated *not* using it, rather than a simple "it didn't work." |
| Removing static videos | Untested at full scale / inconclusive in pilot | Static clips still carry valid label signal; removing ~20% of a small dataset may cost more than it fixes. Open question at full dataset scale. |
| Rare-category upsampling | Two methods tried (vanilla KDE-based vs. repeat-factor sampling); repeat-factor is the principled one adopted, but continuous labels break the original discrete formulation and needed adapting | — |
| Curriculum learning (easy→hard staging by duration/dynamicity) | **[CUT — planned, not executed]** | Don't present as a finding; it's a documented idea, not a result. Mention only if asked "what else did you consider." |

**Framing line:** *"Several standard regularization/augmentation tricks from the video pretraining
literature didn't transfer cleanly to our setting — likely because we're probing a frozen backbone with
very few frames, not pretraining a model from scratch with abundant redundant frames. This shaped our
choice to keep the baseline recipe simple."*

### 2.2 Benchmarking encoders

**Purpose of this section, state it explicitly on the section title slide:** *"The goal here is to
identify the best-performing frame-based model and the best-performing video-based model. Those two
become the models used for every subsequent (Part 3) analysis."* This gives the audience a reason to care
about the comparison beyond "here are some numbers" — it's the model-selection step, not just a survey.

- Methodology note (say this once, applies to every model in the table): for each encoder, we ran a grid
  search over learning rate and weight decay (lr ∈ {1e-3, 1e-4}, wd ∈ {0.01, 0.8}) and report the best
  configuration's test MSE.
- **[HAVE]** Finished models with real test MSE (`Models to run.pdf`): DINOv2 small (0.0055),
  base (0.0052), large (0.0050), giant (0.0051); V-JEPA2 (0.0055); DINOv3 (0.0048, best so far); DINOv2
  robust small/base/large (0.0058 / 0.0053 / 0.0051); ConvNeXt (0.0074, worst so far).
- Plots to build (per PART 1 - ENCODERS §6 — only show what you'll actually have data for by presentation
  time):
  - **Performance vs. number of parameters** — DINOv2 small/base/large/giant is your cleanest scaling
    curve; you have all four numbers already.
  - **Performance vs. publication/release date of the encoder** — same spirit as the parameter-scaling
    plot, orthogonal axis. You already have release years for every candidate encoder (PART 1 - ENCODERS
    §"Priority List"), so this is buildable as soon as enough models across different years have finished.
  - **Robustified vs. normal** — DINOv2 vs. DINOv2-robust, matched by size (small/base/large). You have
    all six numbers already — this plot is buildable now.
  - **CNN vs. Transformer, frame-based** — ConvNeXt (CNN) vs. DINOv2/DINOv3/ViT (Transformer). Partial
    data now (ConvNeXt + DINOv2 + DINOv3 finished; ViT finished per the PDF too — check, it may already
    be usable).
  - **Frame-based vs. video-based** — DINOv2 vs. V-JEPA2 is your headline comparison here and feeds
    directly into Part 3 (these are the two models Part 3 actually uses); you already have both.
  - **[PENDING]** VideoMAE (segment-concatenation fix implemented this session, not yet retrained/run),
    X3D (needs the same segment-concatenation treatment — flagged in the PDF, not yet done), SlowFast (not
    implemented), VideoMamba (not run). Don't promise these plots for the talk unless they'll genuinely be
    ready — better to show a handful of solid comparisons than several with obvious gaps.
- **[CUT]** The long "encoders to add" wishlist (adversarially robust CLIP/FARE/TeCoA, Hiera, TimeSformer,
  InternVideo2, VGG-16, SlowFast, MViT, Video Swin) — this is a roadmap document, not a result. If you want
  one "future directions" slide, name 2-3 of these as next steps, not all of them.

---

## Part 3 — Subsequent analyses: what does the model actually use?

This is your most developed, most interesting section — treat it as the intellectual core of the talk.
Frame it explicitly as **Hypothesis 1: does the model use temporal/sensory history at all**, matching your
own H1/H2/H3 structure in PART 2.docx (only H1 has been run — H2 "motion as attention cue" and H3 "object
permanence" are designed but not executed; mention them as "designed, next" if you want to show the full
arc of the hypothesis-testing framework, but don't claim results for them). Both models used throughout
this section are the winners from Part 2's benchmarking (DINOv2 as best frame-based, V-JEPA2 as best
video-based).

### 3.1 Experiment 1 — Prediction error vs. video duration **[HAVE]**

- Plot: MSE vs. human full-clip reports, for DINOv2 and V-JEPA2 each under 3 conditions (full clip solid,
  shuffled dotted, last-frame dashed), plus human last-frame baseline and the split-half noise floor,
  as a function of video duration.
- Finding: at short durations all curves converge (sanity check — little content, so one frame ≈ full
  clip). As duration grows, last-frame predictors — both the human baseline *and* the models' last-frame
  condition — diverge upward from the full-clip curves. A single frame becomes a progressively worse
  summary of a longer clip, for humans and models alike.
- Finding: the shuffled curves track the ordered curves closely for both models, across the full duration
  range — shuffling frame order barely moves either model's error relative to its own ordered baseline.

### 3.2 Experiment 2 — Same, restricted to low-dynamicity videos (0-2 clusters) **[HAVE]**

- This is your confound-control experiment: restricting to visually static clips minimizes the risk that
  a "last frame" is missing content that appeared earlier (the coverage-loss confound).
- Finding: the same qualitative pattern holds, but the gap at long durations shrinks — some, not all, of
  the full-clip advantage was coverage loss, but a real residual gap remains. This is genuine evidence of
  temporal integration surviving a confound control.

### 3.3 The architecture explanation — why the shuffle-invariance is (partly) provable, not just observed

This is worth its own slide, separate from the empirical shuffle-gap plot, because it upgrades "we
measured a gap near zero" into an actual mechanistic account for at least one of the two models.

- DINOv2 processes each frame of a clip completely independently — no cross-frame communication inside
  the encoder at all.
- The attentive pooler (shared architecture across every model in this project) has no positional
  embedding of any kind, for either model.
- Given those two facts together, DINOv2 + pooler is **mathematically permutation-invariant to frame
  order** — not just an empirical tendency, a guaranteed property of the architecture, for any set of
  trained weights.
- We did not build the architecture to include positional embeddings anywhere — this was a deliberate
  simplicity choice at the time, not something the shuffling analysis was originally designed to probe.
  But the shuffling results (3.1) suggested this absence might matter beyond just explaining DINOv2's
  invariance — prompting a natural follow-up question: would giving the model access to temporal position
  actually change anything?

### 3.4 Follow-up — does adding a positional embedding change anything? **[PENDING]**

- **[PENDING]** Motivated directly by 3.1/3.3: since the current pooler has no positional embedding
  anywhere, we're testing whether adding one changes performance, starting with DINOv2-small as a fast,
  low-cost first test before committing to a larger sweep.
- Framing for the slide: *"We didn't build our architecture to include positional embeddings, but the
  shuffling results raised the question of whether that absence matters. So, especially after the
  shuffling analysis, we're running a first test — adding a positional embedding to the pooler and
  retraining a small model (DINOv2-small) — to see whether it has any measurable impact."*
- This is a good place to end Part 3 on an open, forward-looking note rather than a closed conclusion —
  consistent with the rest of the talk's honesty about what's resolved vs. still in progress.

### 3.5 Benchmark suite: curated diagnostic video sets **[PENDING]**

Distinct from Part 2's encoder benchmarking (which compares *many* encoders to pick the winners) — this
is the two *winning* models (DINOv2, V-JEPA2) run on specialized, curated video sets designed to probe a
specific behavior, rather than the general held-out test set. Matches the "benchmark set" MAIN WRITING
already names as one of the three held-out evaluation sets (test / visualization / benchmark), and the
existing `load_trialTypes.py` benchmark infrastructure (SensoryHistory duration sweep, EventSegmentation,
TemporalDecay/Retention, ObjectPermanence).

- **[PENDING]** Headline example: an object-permanence benchmark set — clips where an object appears,
  disappears, then reappears. This is a direct, complementary probe of H3 (does the model maintain an
  object representation through occlusion) — evidence here doesn't require the full truncation-based H3
  experiment design from PART 2.docx to be run first; it's a faster, earlier read on the same question.
  Worth explicitly connecting the two on the slide/in the talk: *"a first look at H3, before the fuller
  designed experiment."*
- Mention the other benchmark categories only if you'll actually have plots for them by presentation time
  (EventSegmentation, TemporalDecay, SensoryHistory duration sweep) — otherwise name them as available
  infrastructure for future benchmarks, not results.

### 3.6 Visualization: report rate through time **[PENDING]**

The other of the three held-out sets MAIN WRITING names (test / visualization / benchmark) — per-category
predicted (and, where available, human) report rate plotted as a function of how much of a specific
example clip has been revealed, illustrating a trajectory rather than a single aggregate MSE number.

- **[PENDING]** Only plot curves for categories that are actually relevant to that clip's narrative (e.g.
  present from the start, or the one that appears/disappears/reappears) — not all 12 categories on every
  clip. Categories irrelevant to the clip just add flat, uninformative lines near zero and dilute the
  point of the plot, which is to show a clean trajectory for the object(s) the clip is actually about.
- Natural pairing with 3.5: a visualization plot for the *same* clip used in the object-permanence
  benchmark example would let the audience see both the aggregate benchmark number and the qualitative
  trajectory behind it on adjacent slides.

### 3.7 What's designed but not run (H2, H3)

- One slide, framed as "next" rather than "results": H2 (does the model use motion as a segmentation/
  attention cue — reverse playback, optical-flow-based moving/static region masking) and H3 (object
  permanence — does the model's prediction for a disappeared object persist, decay, or drop). Both have
  fully designed experiments in PART 2.docx with clear predictions and confound discussion; the
  truncation-based H3 design specifically hasn't been run yet, though 3.5 gives an earlier, complementary
  read on the same question. Good closing material to show the hypothesis-testing framework generalizes
  beyond H1.

---

## Suggested slide order (condensed)

1. Motivation + task setup
2. Model architecture (one diagram: frozen encoder → pooler → classifier)
3. Part 1 title card
4. Why enrich (static-clip skew)
5. What DBSCAN does + dynamicity pipeline + validation (AUC/Cohen's d)
6. Data generation numbers
7. **[PENDING]** Original vs. enriched-trained model, on enriched test set
8. **[PENDING]** Same, with training-set-size control
9. Part 2 title card — state the goal: pick the best frame-based and video-based model for Part 3
10. Optimization attempts table (brief, "what didn't work and why," incl. the frame-dropping bug-then-fix
    story)
11. Benchmarking: params, publication date, robust-vs-normal, CNN-vs-Transformer, frame-vs-video
12. Part 3 title card
13. Prediction error vs. duration (full/last-frame/shuffled, both models + human + noise floor)
14. Same, low-dynamicity restricted — the confound control
15. Architecture slide: why the shuffle-invariance is (partly) provable
16. **[PENDING]** Positional embedding follow-up (DINOv2-small)
17. **[PENDING]** Benchmark suite: object-permanence (appear/disappear/reappear) — early read on H3
18. **[PENDING]** Visualization: report rate through time, relevant categories only, same example clip as 17
19. What's next: H2/H3 designed experiments, training-set-size control, VideoMAE/X3D benchmarking
20. Close: back to the three-part thesis structure, Part 3 (new behavioral experiments) as future work

---

## Open items to calibrate with you

- Decide how much of the H2/H3 "designed but not run" material you want in the talk vs. held for Q&A.
- Decide whether curriculum learning and the early embedding-based dynamicity exploration (composite
  scores, supervised proxy classifier) are worth a single "other things we considered" slide, or cut
  entirely — I've marked them as cuts above but this is a judgment call about talk length.
- Decide exact timing/status to state for the positional-embedding follow-up (3.4) by presentation day —
  right now it's framed as "planned, about to run" rather than "has results."
