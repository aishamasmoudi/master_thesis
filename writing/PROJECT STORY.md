# Project Story

Outline + Detailed Master Thesis Story.
Sections marked **Plots/figures needed** list what's required to make each part's point; items marked
*(pending)* don't exist yet.

---

## Background

In studying brain function, neural recordings and human behavioral experiments provide key evidence about
underlying mechanisms. Computational modeling offers a complementary approach: it can test hypotheses at
scale and generate precise, falsifiable predictions. In this project, we want to answer a specific scientific 
question: **can we use ANNs to build a computational model of dynamic object perception** — the mechanisms by 
which our brains continuously construct and update object representations that let us perceive, plan, and act in
synchrony with a dynamic world?

Specifically, we investigate:
1. Whether it is possible to learn a spatiotemporal transformation from a large-scale human behavioral
   dataset, using various pretrained vision-encoding models (both video- and image-based).
2. How successful models can be leveraged to study the mechanisms of how humans perceive dynamic visual
   scenes, via in-silico experiments.

## Initial Context:

### Experimental Setup

One image-based (DINOv2) and one video-based model (V-JEPA2) were trained on 12,000 egocentric clips sampled from Ego4D, a
large-scale egocentric video dataset, at a random start time and clip duration spanning 200ms to 15s.
Beforehand, large-scale human behavioral data were collected on those same clips: on each trial, a
participant watches a clip, then reports which objects they recognized from a fixed list of 12 object
categories. The resulting data is represented as a 12-dimensional vector indicating the presence of each
category, with values ranging continuously from 0 to 1 (the fraction of participants who reported that
category). Models are trained to match human responses, then evaluated on a held-out test set. We quantify
human-model alignment by comparing predicted vs. reported object vectors using MSE.

### Model Architecture

Each model follows the same recipe: a **frozen, pretrained encoder** extracts a sequence of token
embeddings from a clip; a **trainable attentive pooler** — adapted from V-JEPA2's own attentive-pooler
architecture — condenses that sequence into a single vector per clip via self-attention and a learned
cross-attention query; and a **trainable linear classifier** produces the final prediction. Only the
pooler and classifier are trained — the encoder stays frozen throughout.

## Project Overview

The project has two stages:

1. **Improve previous human-model alignment**, by (a) enriching the dataset with more dynamic clips,
   (b) training and benchmarking many different image- and video-based models, (c) trying out different
   data augmentation techniques.
2. **Understand the behavior of the best-performing models**, which can later be leveraged to design new
   in-silico experiments — in particular, to understand how, and how much, the model relies on *sensory
   history* — giving us plausible mechanisms for how humans perceive dynamic visual scenes.

---

## Part 1.1 — Enrich the Dataset

We aim to enrich the dataset by adding more dynamic clips. This is motivated by prior knowledge that many
clips in the original dataset are not very dynamic — nothing meaningfully changes, and there aren't many
distinct scenes within a clip. Since our core question is whether a model can transform visual input over
time into a representation that stays useful as things move and change (i.e. genuine temporal
integration), a dataset with too few dynamic clips is underpowered for the specific question we're trying
to answer: a model doing real temporal integration and a model that only recognizes objects in one frame
would behave identically on a clip where nothing changes, so such clips carry no power to distinguish the
two hypotheses.

Our approach is to build a **"dynamicity" metric** that meaningfully assesses how dynamic a clip is, and
that we can then use to sample additional dynamic clips from the original Ego4D pool — enriching our
12,000-clip dataset. The metric we use is the number of clusters DBSCAN assigns to a clip's frame
embeddings (more distinct visual/semantic states over time → more clusters). We first validate that this
method is an efficient, reliable way to judge dynamicity, then apply it to generate more dynamic clips. As
part of validating the method, we also use it to confirm our initial intuition: most clips in the original
dataset are, in fact, not very dynamic. We then retrain our model on the enriched dataset and test whether
alignment improves.

**Plots/figures needed:**
- Example dynamic vs. low-dynamic (manually chosen) video clips, to build intuition before the statistics
- **OPTIONAL**: DBSCAN hyperparameter optimization: plot showing ROC-AUC, p-value, and Cohen's d when separating low- and
  high-dynamic clips after running DBSCAN with several different pairs of hyperparameters (eps and min_samples)
- Distribution comparison (boxplot/violin/histogram): DBSCAN cluster count, dynamic vs. low-dynamic
  clips, for the chosen optimal hyperparameters
- **OPTIONAL**: Sanity check: cluster count vs. truncation duration for a few example clips (should never decrease)
- Held-out generalization: same distribution comparison, on a held-out set of dynamic clips
- Cluster-count histogram across the full 12,000-clip dataset — this is both the basis for the threshold
  choice (4 clusters) *and* the direct confirmation of "most clips are not very dynamic"
- Example low cluster number vs high cluster number video clips to build intuition on the dynamicity metric
- **OPTIONAL**: Duration distribution per cluster count, used to set the minimum-duration floor for generated clips
- **(pending)** MSE: original model vs. model retrained on the enriched dataset, evaluated on the enriched
  test set — the actual payoff of this section
- **(pending)** Same comparison, with the enriched training set subsampled to match the original size, to
  rule out "just more data" as a confound

## Part 1.2 — Benchmark Encoders

The goal is to identify the best-performing frame-based and video-based encoders. We train several image-
and video-based models varying in architecture, size, and training objective (adversarially robust vs. standard, etc.), 
then benchmark them against each other to choose the two models — one frame-based, one video-based — carried forward into Part 2.

All benchmarked encoders use the training recipe settled on in Part 1.3 below.

**Plots/figures needed (all pending):**
- Performance vs. number of parameters (DINOv2 small/base/large/giant)
- Performance vs. publication date, across all benchmarked image then video encoders
- Robust vs. standard training (e.g. DINOv2 vs. DINOv2-robust, matched by size)
- Architecture family comparison, frame-based (CNN vs. Transformer)
- Frame-based vs. video-based

## Part 1.3 — Data Augmentation Experiments

Before settling on a training recipe for the encoder benchmark in Part 1.2, we tried two data
augmentation techniques borrowed from the video self-supervised-learning literature (V-JEPA, VideoMAE):
**tube masking** and **random frame dropping**. Both are applied to the pooler's attention mask, not to
the pixels — the frozen encoder always sees every frame in full; only the *pooler* is told which tokens to
ignore. This matters for interpreting the findings below: unlike in V-JEPA's own self-supervised
pretraining, where masking is applied to the encoder's input and forces it to learn richer representations
to fill in the gaps, our masking never touches the frozen encoder — it can only ever throw information
away from the trainable head, not force better representations out of it.

**Tube masking.** Following V-JEPA's recipe, we sample two masks per clip — one short-range (several small
spatial blocks, e.g. 8) and one long-range (a couple of large blocks, e.g. 2) — and mask the *same* spatial
patch positions across every frame of the clip (hence "tube": the masked region extends through the full
temporal dimension like a cylinder), together covering a `tube_mask_ratio` fraction of each frame. We swept
`tube_mask_ratio` starting from conservative values (V-JEPA/VideoMAE's own recipe uses a 90% ratio, judged
too aggressive here since that number was tuned for a masked-reconstruction pretraining objective, not a
regression task with comparatively few frames). **Finding: tube masking hurt performance.** Since the
encoder still sees the full frame regardless, masking the pooler's attention only removes information the
pooler could have used to make its prediction — with every token potentially informative for regression,
this is pure information loss rather than a useful regularizer.

**Random frame dropping.** A random fraction of *frames* per clip is marked
invalid in the attention mask, same mechanism as tube masking but along the temporal axis. We random
drop real, non-padded frames. This technique yields performance comparable to, or better than, the unaugmneted baseline.

As a later consistency check, once DINOv2, V-JEPA2, and ResNet-50 were each independently
hyperparameter-optimized, frame dropping was added on top of each one's best configuration to see
whether the augmentation helped reliably across encoder types. It consistently helped
DINOv2 and V-JEPA2 — but was catastrophic for ResNet-50.

**Decision: frame dropping was not adopted into the final training recipe**, despite helping two of the
three encoders tested, because of that ResNet-50 result — rather than maintain a per-encoder-conditional
augmentation policy, we kept the recipe simple and consistent across all benchmarked encoders, given time
constraints.

**Plots/figures needed:**
- **Pending**: Test MSE vs. `tube_mask_ratio`, across the swept values
- Test MSE vs. `frame_drop_ratio`, across the swept values (post-fix)
- Per-encoder consistency check: test MSE with vs. without frame dropping, for DINOv2 / V-JEPA2 / ResNet-50
  side by side — this is the plot that actually motivates the "not adopted" decision

---

## Part 2 — What Does the Model Actually Use?

Main question: **how does the model use temporal / sensory history?**

### Experiment 1 — Prediction error vs. video duration and dynamicity

We compare MSE between human full-clip reports and three references: the model's full-clip prediction, the
model's last-frame-only prediction, and a human last-frame baseline (participants who saw only the final
frame). We also plot the split-half noise floor (human-to-human reliability) as a reference ceiling. All of
this is shown as a function of both video duration, then of DBSCAN cluster count.

**Finding**: a single frame becomes a progressively worse summary of a longer clip, for humans and models
alike — full-clip performance stays comparatively stable while last-frame performance degrades as content
accumulates. An important confound: this could reflect either genuine temporal integration, or simply
*coverage loss* — a last frame necessarily misses whatever appeared earlier in the clip, regardless of
whether the model does any real temporal reasoning.

**Plots/figures needed:**
- MSE vs. video duration — full clip, last frame, human last-frame baseline, noise floor
- **Needs rerunning — was built with the wrong cluster threshold**: MSE vs. DBSCAN cluster count — same
  four references. *Note: either the coarse 2-bin split (low dynamicity = 0-3 clusters, i.e. below the
  dynamic threshold, vs. dynamic = 4+ — the currently-built version uses 0-2 vs. 3+, a stale boundary from
  before the threshold was confirmed as 4, which leaves 3-cluster clips in neither bin), or the original,
  finer-grained experiment design (0, 1, 2, ... 7+).*

### Experiment 2 — Confound control: restricting to low-dynamicity clips

Same visualization as Experiment 1, restricted to low-dynamicity clips only (using the same DBSCAN metric),
which minimizes the risk that a missing last frame is hiding content shown earlier.

**Finding**: the same qualitative pattern holds, but the gap at longer durations shrinks. Some of the
full-clip advantage was coverage loss — but a real gap remains, giving genuine evidence of temporal
integration beyond what coverage loss alone would predict.

**Plots/figures needed:**
- **Needs rerunning — same stale-threshold issue as Experiment 1**: same MSE-vs-duration plot, restricted
  to low-dynamicity clips (0-3, not 0-2)

### Experiment 3 — Is the model sensitive to sequence order?

We test whether shuffling frame order (holding the same set of frames fixed, only permuting their
sequence) changes model predictions — this isolates order-sensitivity from the coverage-loss confound
entirely, since shuffling never removes content.

*What makes the two encoders different here*: DINOv2 processes each frame of a clip completely
independently — there is no cross-frame communication inside the encoder at all, so a priori we would not
expect it to be order-sensitive. V-JEPA2 processes an entire clip together, through a shared
spatio-temporal transformer that uses rotary position embeddings tied to each token's exact
(time, height, width) location — so a priori we would expect it to be *capable* of order sensitivity, since
its own internal computation already depends on absolute temporal position. Separately from the encoder:
our attentive pooler (identical for both models) has no positional embedding of its own — a deliberate
simplicity choice at the time it was built. So this experiment is really testing the *overall* system
(encoder + pooler) for order sensitivity, not the pooler in isolation.

**Finding**: shuffle invariance in *both* models. For DINOv2 this is architecturally unsurprising, given the
above. For V-JEPA2 it's a more informative result: since its backbone has access to positional information,
we'd expect shuffling to hurt performance — and increasingly so for longer clips, where there's more
content to reorder. That's not what we observe: V-JEPA2 appears to integrate information across frames
(unlike a model doing nothing with time at all) without being sensitive to the order in which frames
arrive.

This result made us examine the fact that our current architecture has no positional embedding anywhere in
the trainable head, but the shuffling results suggest this absence could be one reason the model behaves 
less human-like than it otherwise might (i.e. not leveraging sequence information a human plausibly does). 
Follow-up question: would explicitly giving the model access to temporal position actually change anything?

**Plots/figures needed:**
- MSE gap (shuffled − ordered) vs. video duration, both models, with a zero reference line — the most
  direct visualization of "how much does shuffling change things"
- *(supporting, optional)* the raw MSE-vs-duration plot with the shuffled condition overlaid on the
  ordered/last-frame curves from Experiment 1, showing the shuffled curve tracking the ordered one closely

### Experiment 4 — Adding a positional embedding *(pending)*

Direct follow-up to Experiment 3: add a positional embedding to the pooler and retrain a small model
(DINOv2-small, as a fast first test) to see whether giving the model explicit access to temporal position
changes its predictions or its alignment with human reports.
*Note: We should probably run DINOv2-small with positional embedding without the bug fix (applying attention mask
to the cross attention step). Because we need it to compare it to what I already have.*

**Plots/figures needed:**
- **(pending)** MSE — DINOv2-small, with vs. without the added positional embedding

### Experiment 5 — Curated diagnostic video sets *(pending)*

We evaluate the two winning models on benchmark sets specifically designed to probe particular behaviors,
beyond the general held-out test set: a SensoryHistory duration sweep, EventSegmentation, TemporalDecay/
Retention, and ObjectPermanence (clips where an object appears, disappears, then reappears — a direct,
complementary probe of whether the model maintains an object representation through occlusion).

**Plots/figures needed:**
- **(pending)** MSE / predicted report rate on each curated benchmark set, both models
- **(pending)** Visualization: predicted (and human) report rate over time for a specific
  example clip (e.g. from the ObjectPermanence set) — showing only the categories relevant to that clip's
  narrative, not all 12, to keep the trajectory readable

---

## Future Work

- **H2 — motion as a segmentation/attention cue**: does the model use motion to direct attention to
  regions of interest, rather than for identity? Tested via reverse playback (motion magnitude/structure
  preserved, direction flipped) and optical-flow-based masking of moving vs. static regions.
- **Mechanistic/architectural components**: which components are actually necessary for the model's
  predictive power? (ablation studies — temporal module vs. none, attention vs. mean pooling, etc.)
- **New in-silico → human experiments**: from these findings, what new behavioral experiments can we
  design to test the resulting hypotheses directly on humans, to understand the underlying mechanisms of
  dynamic scene perception?

## Conclusion

*(to be written once Part 1.1's enrichment results and Part 2's pending experiments are in — the shape of
the closing argument depends on those results, particularly whether enrichment improves alignment and
whether the positional-embedding test changes the shuffle-invariance finding.)*