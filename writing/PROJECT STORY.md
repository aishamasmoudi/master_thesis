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
2. Whether, through controlled in-silico manipulations of the input (video duration, frame order, curated
   diagnostic clips), we can identify what temporal information a trained model actually relies on to
   reproduce human report rates — and what that reveals about plausible mechanisms behind human dynamic
   object perception.

## Literature Review

**ANN-behavior alignment as a validated methodology.** Comparing model predictions to human/primate behavior
is an established way to test mechanistic hypotheses about the ventral visual stream. Simple, learned linear
readouts of inferior-temporal (IT) population activity accurately predict human object-recognition behavior
across a wide range of recognition tasks (Majaj, Hong, Solomon & DiCarlo, 2015), and this "compare model
predictions to large-scale human/primate behavioral data" paradigm has become the standard for evaluating
whether an ANN is a good model of biological vision. But it also has a known limit: even state-of-the-art
deep convolutional networks, while capturing coarse *object-level* confusion patterns, fail to predict
recognition behavior at the level of *individual images* — a failure not explained by simple low-level image
attributes (Rajalingham et al., 2018). This motivates looking beyond standard feedforward, single-image
architectures and objectives. Even the classic account of *how* the ventral stream itself builds
invariant object representations leans on time: DiCarlo, Zoccolan & Rust (2012) propose that a largely
feedforward cascade of computations achieves this, shaped in part by *temporal contiguity* in natural visual
experience — the idea that images of the same object tend to arrive in nearby moments in time, which the
visual system can exploit as a learning signal. Time, in other words, already matters for object
recognition even before we consider clips that unfold dynamically in front of a viewer.

**Real-world vision, though, rarely arrives as an isolated snapshot.** Humans can extract the gist of a scene
from a single image flashed for as little as 13ms (Potter, Wyble, Hagmann & McCourt, 2014) — evidence that a
single feedforward pass can support fast, coarse recognition. But natural viewing is a *sequence* of such
snapshots, not one flash, and the sequence itself carries information a single frame does not. Comparing
feedforward and recurrent neural network models against human recognition of rapidly-presented image
sequences, Sörensen, Bohté, de Jong, Slagter & Scholte (2023) found that only models with lateral recurrence
— i.e., that integrate information across frames rather than treating each one independently — matched
human trial-by-trial dynamic recognition performance, and that adding a neural-adaptation mechanism improved
this further. Separately, the human visual system appears to actively reshape how it represents a moving
scene over time: it "straightens" the trajectory that a natural video traces through pixel space, making it
straighter in its internal representation — a transformation hypothesized to support predicting what comes
next (Hénaff, Goris & Simoncelli, 2019). And a recent large-scale MEG study shows this isn't just a passive
representational property but something that actively drives behavior: how long a person's gaze lingers on
a part of a scene is best explained not by how much *visual processing* the current view demands, but by
ongoing *memory encoding* of what's already been seen — evidence that the brain treats "how much of this
scene have I already absorbed" as a variable that shapes ongoing behavior in real time (Sulewski, Amme,
Hebart, König & Kietzmann, 2025). This is close to the specific idea this thesis targets: that a person's
representation of a scene accumulates continuously and depends on what has come before, not just on the
instantaneous input, i.e. on the scene's *sensory history*. Category identity itself also modulates how
readily something reaches awareness at all — the attentional-blink deficit differs systematically across
object categories according to their mid- and high-level visual features (Lindh, Sligte, Assecondi, Shapiro
& Charest, 2019), a reminder that per-category effects in our own report-rate data may partly reflect
category-level detectability rather than dynamics alone.

**Self-supervised video models now offer a candidate mechanism for how this could work computationally.**
Pretraining a model to predict withheld or future content *in a learned latent space*, rather than in pixel
space, on internet-scale natural video, yields representations that transfer to motion understanding, action
anticipation, and even zero-shot robotic planning (Assran et al., 2025 — V-JEPA 2). The same latent-space
predictive objective, trained purely on natural video with no built-in physical priors, gives rise to
intuitive-physics understanding — including something resembling object permanence — when tested with
violation-of-expectation paradigms (Garrido et al., 2025). Critically, this connects back to biology: primate
neural responses during mental simulation of dynamic scenes are best predicted by models trained to forecast
the *future latent state* of pretrained video foundation models — better than pixel-level predictors, and
better than models without a dynamic-scene training objective (Nayebi, Rajalingham, Jazayeri & Yang, 2023).
Together, these results are the direct motivation for comparing a video-native, future-predictive
architecture against a purely spatial, frame-based one in this thesis, and for treating alignment to human
*behavior* — not classification accuracy — as the metric that matters.

**Where this leaves us.** Taken together, this literature establishes that ANN-behavior alignment is a
validated methodology (Majaj et al., 2015; Rajalingham et al., 2018), and that when perception unfolds over
time, both human behavior (Sörensen et al., 2023; Hénaff et al., 2019; Sulewski et al., 2025) and the most
successful video-based models (Assran et al., 2025; Garrido et al., 2025; Nayebi et al., 2023) appear to
depend on integrating information across time rather than processing only the current instant. That's the
gap the two questions above are aimed at: whether the same kind of alignment can be achieved for the
specific case we study — humans reporting recognized objects from short egocentric clips (question 1) —
and, if it can, what probing the resulting model's behavior reveals about how it's using time, and what
that in turn suggests about the mechanisms of human dynamic perception (question 2).

**A note on coverage.** The literature review above is built around the papers my supervisor gave me. It
covers the ANN-behavior alignment methodology, evidence for temporal/sequential processing in human dynamic
vision, and self-supervised video models as a candidate mechanism — the three pillars this thesis rests on.
It does *not* include anything specific to (a) the exact behavioral paradigm used here — continuous
recognition-style report rates collected as a function of viewing duration — or (b) egocentric video
specifically, as opposed to third-person natural video/scenes. Neither gap is necessarily a problem — the
paradigm and stimulus choice are methodological decisions this thesis makes and justifies on its own terms,
not literature it needs to review — but if a reader pushes on either point, those are the two places where
an additional citation or two would be worth having in reserve.

## Initial Context:

### Experimental Setup

We train one image-based and one video-based pretrained encoder — each paired with the trainable head
described below — on 12,000 egocentric clips sampled from Ego4D, a large-scale egocentric video dataset, at
a random start time and clip duration spanning 200ms to 15s. Beforehand, large-scale human behavioral data
were collected on those same clips: on each trial, a participant watches a clip, then reports which objects
they recognized from a fixed list of 12 object categories. The resulting data is represented as a
12-dimensional vector indicating the presence of each category, with values ranging continuously from 0 to
1 (the fraction of participants who reported that category). Models are trained to match human responses,
then evaluated on a held-out test set. We quantify human-model alignment by comparing predicted vs. reported
object vectors using MSE. (Part 1.3 below covers *which* specific encoders were selected to fill the
image-based and video-based role, and how.)

### Model Architecture

Each model follows the same recipe: a **frozen, pretrained encoder** extracts a sequence of token
embeddings from a clip; a **trainable attentive pooler** — adapted from V-JEPA2's own attentive-pooler
architecture — condenses that sequence into a single vector per clip via self-attention and a learned
cross-attention query; and a **trainable linear classifier** produces the final prediction. Only the
pooler and classifier are trained — the encoder stays frozen throughout.

## Project Overview

The project has two stages:

1. **Improve previous human-model alignment**, by (a) enriching the dataset with more dynamic clips,
   (b) trying out different data augmentation techniques, (c) training and benchmarking many different
   image- and video-based models.
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

## Part 1.2 — Data Augmentation Experiments

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

## Part 1.3 — Benchmark Encoders

The goal is to identify the best-performing frame-based and video-based encoders. We train several image-
and video-based models varying in architecture, size, and training objective (adversarially robust vs. standard, etc.), 
then benchmark them against each other to choose the two models — one frame-based, one video-based — carried forward into Part 2.

All benchmarked encoders use the training recipe settled on in Part 1.2 above.

**Plots/figures needed (all pending):**
- Performance vs. number of parameters (DINOv2 small/base/large/giant)
- Performance vs. publication date, across all benchmarked image then video encoders
- Robust vs. standard training (e.g. DINOv2 vs. DINOv2-robust, matched by size)
- Architecture family comparison, frame-based (CNN vs. Transformer)
- Frame-based vs. video-based

---

## Part 2 — What Does the Model Actually Use?

### Setting the stage: humans exhibit sensory history

Before turning to the model, it's worth stating plainly what is already established on the human side of
this dataset. Human report rates are collected on full clips, but we also collected a last-frame-only
baseline — a separate group of participants who saw only the single final frame of each clip instead of the
whole thing. This baseline diverges further from the full-clip ground truth as clip duration grows: a
single final frame becomes a progressively worse summary of what was actually shown as clips get longer.
In other words, human report rates depend on the accumulated dynamic content of a clip, not just its final
instant — humans behaviorally exhibit sensory history. This isn't a hypothesis we test in Part 2; it's the
standalone behavioral fact that motivates it. Given that humans do this, the question for the rest of this
section is whether, and through which specific mechanism, our trained models do the same.

Main question: **through which mechanism, if any, does the model use temporal / sensory history?**

### Possible Hypotheses

We consider three (non-exclusive) hypotheses for what mechanism, if any, underlies a model's use of sensory
history:

**H1 — The model doesn't use temporal information at all.** The model could be solving the task by
recognizing objects within individual frames, with no real integration over time — treating a clip as an
unordered set of images rather than a sequence. This is plausible for humans too: people can extract scene
meaning from a single ~13-100ms flash (Potter et al., 2014), so human reports could similarly be dominated
by a few informative frames rather than genuine accumulation. This is the baseline hypothesis every other
one depends on — if temporal information doesn't matter at all, there is no spatiotemporal transformation
left to explain. **Tested by**: Experiments 1-3 below (duration/dynamicity error curves, order-shuffling).

**H2 — The model uses motion as a segmentation/attention cue, not for identity.** Rather than building
genuine temporal object representations, the model might use motion only to direct attention toward regions
of interest — moving things "pop out" and get recognized from appearance alone, with time playing a *where*
role rather than a *what* role. This is plausible for humans because motion is processed by the dorsal
stream and captures attention pre-attentively, directing where the ventral stream looks before it determines
what it's looking at. **Tested by**: reverse-playback and moving-vs-static-region masking (Future Work —
not yet run).

**H3 — The model maintains object representations through occlusion.** Objects frequently leave the frame
or become occluded in egocentric video; a model with genuine temporal integration should keep predicting an
object's presence for a while after it disappears, rather than re-evaluating presence from only the most
recent frames. This is plausible for humans because object permanence is a basic property of human
perception, and predictive-coding accounts propose the brain actively maintains object predictions through
occlusion rather than treating each moment as independent. **Tested by**: the ObjectPermanence curated
benchmark set (Experiment 5 — not yet run).

These aren't mutually exclusive: a model could, for instance, fail to track objects through occlusion
(against H3) while still showing some order-sensitivity (for H1) if it uses motion cues in a coarse,
non-persistent way (some H2).

### Experiment 1 — Prediction error vs. video duration and dynamicity

We compare MSE between human full-clip reports and three references: the model's full-clip prediction, the
model's last-frame-only prediction, and a human last-frame baseline (participants who saw only the final
frame). We also plot the split-half noise floor (human-to-human reliability) as a reference ceiling. All of
this is shown as a function of both video duration, then of DBSCAN cluster count.

**Confound**: in the main test set, different clips appear at different durations — video identity and
duration are entangled, so a duration-dependent trend could partly reflect which specific clips happen to be
long vs. short, rather than duration itself.

**Controlled companion — SensoryHistory benchmark.** To isolate the duration effect from video identity, we
repeat this analysis on the SensoryHistory benchmark set: a fixed pool of target videos, each truncated to
the same 5 durations (0.25s, 0.5s, 1.0s, 2.0s, 4.0s), so every duration bin contains the *same* underlying
clips, just cut shorter or longer. Any duration-dependent trend observed here can't be attributed to which
clips happen to populate each bin, since the clips are identical across bins — it isolates the effect of
duration itself.

**Finding**: on the main test set, a single frame becomes a progressively worse summary of a longer clip,
for humans and models alike — full-clip performance stays comparatively stable while last-frame performance
degrades as content accumulates. This could reflect either genuine temporal integration, or simply
*coverage loss* — a last frame necessarily misses whatever appeared earlier in the clip, regardless of
whether the model does any real temporal reasoning. *(SensoryHistory-benchmark result: pending.)*

**Plots/figures needed:**
- MSE vs. video duration, main test set — full clip, last frame, human last-frame baseline, noise floor
- **Needs rerunning — was built with the wrong cluster threshold**: MSE vs. DBSCAN cluster count — same
  four references. *Note: either the coarse 2-bin split (low dynamicity = 0-3 clusters, i.e. below the
  dynamic threshold, vs. dynamic = 4+ — the currently-built version uses 0-2 vs. 3+, a stale boundary from
  before the threshold was confirmed as 4, which leaves 3-cluster clips in neither bin), or the original,
  finer-grained experiment design (0, 1, 2, ... 7+).*
- **(pending)** Same MSE-vs-duration plot, on the SensoryHistory benchmark (same clips at 0.25/0.5/1.0/2.0/4.0s)
  — the video-identity-controlled companion to the main-test-set plot above

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