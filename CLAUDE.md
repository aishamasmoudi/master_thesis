# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

This is a master's thesis codebase studying **human "sensory history" / continuous recognition memory**
(the "SensoryHistory" experiment, on Ego4D video clips) and comparing human behavioral reports against
representations extracted from frozen neural video/image encoders. The core question across the
experiments is whether/how temporal information (single frame vs. many frames, frame order, playback
speed, event structure, video dynamicity) affects how well a model's learned representation predicts human
category-report rates on short video clips, and how that compares to how much the *same* manipulations
affect human report rates themselves.

There is no build system, package manifest, or test suite in this repo — it is a collection of standalone
research scripts run on a SLURM/GPU cluster (paths reference `/braintree/...` and `/orcd/...` HPC mounts)
plus two analysis notebooks. One import points at something that lives outside this repo checkout:
- `VideoMamba` (imported by `videoBased_encoders.py`) — expected at
  `~/Ego4D/model_optimization/VideoMamba/videomamba/video_sm/models` on the training machine, added to
  `sys.path` at runtime. In this checkout it's present as a broken git submodule entry (`git ls-tree` shows
  it as a `160000` gitlink) with **no `.gitmodules` and an empty working directory** — it was never
  actually populated. Only the `videomamba_m` encoder needs it; nothing else in the repo is affected.
  `preprocess_data.py` (imported by `data_loaders.py`) *used to* be external but is now present in this
  checkout — don't assume it's missing.

Do not try to "fix" the VideoMamba import — it's a cluster-side dependency, not a bug in this checkout.

## Conceptual background: the frozen-encoder + attentive-probing pipeline

Every model in this project follows the same recipe: **freeze a pretrained vision/video backbone, and
train only a lightweight attention-based pooling head + linear classifier on top of it** to regress human
category-report rates. This is "attentive probing" — a fancier cousin of linear probing, needed because the
backbone's output is a *sequence* of tokens, not a single fixed-size vector.

- **Why frozen?** `requires_grad=False` is set on `model_SH.encoder` in both training scripts. Only
  `self.pooler` and `self.classifier` are ever trained (`trainable = [p for p in model_SH.parameters() if
  p.requires_grad]`). This is deliberate: the experiment measures whether a *pretrained* representation
  already contains what's needed to predict human reports, not whether you can fine-tune a giant model to
  fit this small dataset. A big/expressive trainable head would confound "the representation was already
  good" with "the head did the heavy lifting," so the head is kept minimal on purpose.

- **ViT vocabulary, with real numbers from this codebase:**
  - **patch_size**: an image gets chopped into non-overlapping squares (patches), each treated as one
    transformer token — exactly like words in a sentence. DINOv2 uses `patch_size=14` on a `224×224`
    frame → a `16×16` grid → **256 patches**. Mechanically this is a single strided convolution
    (`kernel_size=stride=patch_size`), mathematically equivalent to flattening each patch's pixels and
    running them through a linear layer.
  - **hidden_size**: how many numbers describe each token after encoding (e.g. 1024 for DINOv2-large).
  - **added_tokens**: extra tokens not tied to any patch, e.g. a CLS token (DINOv2: `added_tokens=1`, so a
    frame really produces 257 tokens, not 256 — always account for this or mask/reshape math goes off by
    one).
  - **tubelet_size**: video-native models (V-JEPA2) patchify *space and time together* — a tubelet is a 3D
    block spanning `tubelet_size` consecutive frames (V-JEPA2: `tubelet_size=2`). This roughly quarters
    attention compute (sequence length halves, cost is quadratic in sequence length) and lets local motion
    get encoded directly into a token's content instead of needing attention to infer it post-hoc from two
    separate per-frame tokens. `num_patches`/`tubelets` bookkeeping (`tubelets = frames_per_clip //
    tubelet_size`) has to be exact, or the padding mask gets reshaped at the wrong boundaries.
  - **"Tokenizing a clip" differs by backbone**: the same 60-frame clip becomes ~15,420 independent
    per-frame tokens under DINOv2, vs. 30 tubelet-tokens (each spanning 2 frames + spatial patch) under
    V-JEPA2. This is *why* `encoder_config` exists — every downstream reshape needs to know the specific
    tokenization scheme in use.

- **Why a custom `VJEPA2AttentivePoolerMasked` instead of HF's `VJEPA2ForVideoClassification`?** HF ships a
  ready-made encoder+pooler+classifier class with exactly this shape, and it's imported (but never
  instantiated — dead import) in all four `*_encoders.py`/`eval_model_*.py` files. Two reasons it can't be
  used directly: (1) its pooler (`VJEPA2AttentivePooler`) has **no attention-mask parameter at all** — it
  can't ignore padding tokens, which this dataset needs constantly since clips range ~0.1–15s and get
  forced into a fixed `frames_per_clip` window; (2) it hardcodes its own `VJEPA2Model` backbone internally
  rather than accepting a pre-loaded encoder, so it can't be reused across the dozen unrelated backbones
  (DINOv2, ResNet, ConvNeXt, X3D, VideoMAE, ...) this project compares. `VJEPA2AttentivePoolerMasked`
  threads a padding mask through both its self-attention layers *and* its final cross-attention pooling
  step, and the wrapper classes (`FrameEncoder_.../VideoEncoder_...ForHumanSensoryHistoryReports`) take the
  encoder as a constructor argument so the same pooling recipe works behind any backbone.

- **The one real difference between frame-based and video-based**: DINOv2 processes each frame of a clip
  as an *independent* image (no cross-frame attention inside DINOv2 at all — frame 5 never sees frame 40).
  V-JEPA2 processes the whole clip in one call, with its own internal spatio-temporal self-attention using
  **rotary position embeddings** tied to each token's `(time, height, width)` grid slot (`VJEPA2RopeAttention`
  in `transformers.models.vjepa2.modeling_vjepa2`, confirmed by reading the actual library source — the
  pooler's own attention classes, `VJEPA2PoolerSelfAttention`/`VJEPA2PoolerCrossAttention`, do **not** use
  RoPE or any positional signal). This is the entire reason frame-order shuffling can only ever affect
  V-JEPA2: DINOv2 + the (positionless) pooler is mathematically permutation-invariant to frame order by
  construction — no retraining or code change can alter that, since the invariance is architectural, not a
  property of learned weights. If frame-order sensitivity is ever wanted for the frame-based branch, it
  would require adding an explicit learned temporal positional embedding before pooling — nothing in the
  codebase currently does this.

- **Pooler mechanics** (`VJEPA2AttentivePoolerMasked.forward`, identical in all four files): input is the
  full per-clip token sequence (`hidden_state`, shape `(batch, N, hidden_size)`). A few self-attention
  layers let tokens refine each other (respecting the mask) — still `N` tokens after this, nothing pooled
  yet. Then a single **learned query** (`self.query_tokens`, one shared trainable vector reused for every
  clip) cross-attends over all `N` tokens and produces exactly one output vector — this is the actual
  pooling step, a trainable, content-weighted alternative to averaging. The variable name `hidden_state`
  gets reassigned partway through the function (many-token sequence → one pooled vector) — easy to misread,
  worth mentally renaming to `pooled_state` after the cross-attention call. Finally
  `self.classifier = nn.Linear(hidden_size, 12)` maps the pooled vector to 12 category scores, trained via
  MSE against human report rates — kept deliberately linear for the same "don't let the head do the work"
  reason the whole pipeline is frozen.

- **"Full clip" / "last frame" / "shuffled" are all the same trained checkpoint, evaluated differently at
  inference time** — not three separately-trained models. `eval_model_*.py` loads one fixed
  `checkpoint_path` regardless of the `one_frame`/`shuffle_frames` flags, and only ever runs
  `model.eval()` + `torch.no_grad()` inference. "Last frame" for V-JEPA2 specifically means "last 1
  tubelet" (2 frames), not literally 1 frame — the 3D patch embedding conv structurally cannot process
  fewer than `tubelet_size` frames (HF's `VJEPA2Embeddings.forward` duplicates frames if given fewer than
  that). The tubelet-shuffle experiment permutes whole tubelets, not individual frames, both because the
  encoder has no way to represent anything finer than a tubelet, and because frame-level shuffling would
  split apart genuinely-adjacent frame pairs into physically-impossible fake motion — a confound on top of
  the intended "does long-range order matter" test.

## Running the code

There are no test/lint/build commands (no `pytest`, no `pyproject.toml`/`requirements.txt`). Scripts are
run directly with `python <script>.py`, generally on a multi-GPU SLURM node. Notable entry points:

- `python frameBased_encoders.py --base_dir <path> --encoder <name> [--approach baseline] [--batch_size 10]`
  Trains a frame-based encoder (e.g. `dino_v2*`, `resnet-50*`, `ViT`, `convnext*`, `siglip2`, `alexnet`)
  via `torch.multiprocessing.spawn` + DDP across all visible GPUs (`torch.cuda.device_count()`).
- `python videoBased_encoders.py --base_dir <path> --encoder <name> [--approach baseline] [--batch_size 10]`
  Same, but for video-native encoders (`vjepa2`, `x3d_s/m/l`, `videomae_b/l`, `videomamba_m`).
- `eval_model_frame_based.py` / `eval_model_video_based.py` are **not** parameterized via argparse/`main()`
  despite importing `argparse` — they are top-to-bottom scripts with hardcoded config near the bottom
  (`hypothesis`, `model_name`, `model_ID`, `checkpoint_path`, `base_dir`, `one_frame`, `shuffle_frames`,
  `low_dynamicity_only`). Edit those variables in place before running, or copy the file per experiment.
  They load a trained checkpoint and dump per-video/per-category predictions to a CSV in
  `{exp_results_dir}/{hypothesis}_{model_name}-{experiment_id}_preds_{frames}Frames_AttentionalPooling.csv`.
- `reliability.py` / `data_loaders.py` / `load_trialTypes.py` are library modules imported by analysis
  code (and by the notebooks), not standalone entry points.
- `evaluation_plots.ipynb` — the original analysis notebook, organized by hypothesis (e.g. "Hypothesis 1:
  Temporal Information — Freeze Frame & Single Frame"). Superseded for the duration/dynamicity/MSE-gap
  analyses by `evaluation_plots_clean.ipynb` (see below) — prefer extending the clean notebook over this
  one for new prediction-error-style plots.
- `evaluation_plots_clean.ipynb` — see "Evaluation notebook" section below.

Because training/eval assumes cluster paths (`/braintree/data2/active/users/aicha/Ego4D_data`,
`/orcd/data/dicarlo/001/om/lynnka`, `/home/aicha/orcd/pool/Ego4D_data`) and GPU hardware, most of this
code cannot be exercised locally — treat correctness review as static/code-reading rather than
run-and-check unless you're actually on the cluster.

## Architecture

### Data pipeline (human side)
`data_loaders.py` loads human behavioral trial data (button-press "final_choice" reports per video,
per category) from CSV/pickle caches on the cluster, falling back to `preprocess_data.py` to build the
cache from raw export data if it's missing or `recompute=True` is passed. `final_choice` is stored as a
stringified bool array and parsed back with `utils.string_to_bool_array`; `categories` is a stringified
list parsed with `ast.literal_eval`. `load_trialTypes.py` builds the *trial definitions* (video IDs, clip
start/duration, S3 URLs) for the various experiment conditions: main SH videos, and the benchmark suites
(SensoryHistory duration sweep, EventSegmentation, TemporalDecay/Retention, ObjectPermanence, Visualization
sweep, RSVP, clip-sequence conditions). `preprocess_data.py`'s `assign_clusterInfo` merges precomputed
DBSCAN cluster counts (`n_clusters`, `n_noise_points`, `cluster_per_frame` — a per-video "how much does
this clip's content visually change over time" metric, from `clusterAnnotations_*.csv`) onto trial and
report-rate dataframes via a `stimulus_video_url` lookup — cheap, since clustering was already run offline.
`reliability.py` computes split-half reliability of human report rates (correlation/MSE/R²): for a given
grouping variable (duration bin, or any other column), it randomly splits participants into two halves,
compares their aggregate report rates, and fits a reliability-vs-sample-size curve
(`compute_reliability_scaling` → `apply_reliability_prediction`, saturating-exponential curve fit) to
extrapolate reliability at a chosen sample size. This split-half MSE is the "noise floor" used in the
evaluation notebook — the error even a perfect model couldn't beat, since the human "ground truth" itself
is a noisy finite-sample estimate. `compute_reliability`/`compute_reliability_scaling` are generic w.r.t.
`grouping_variable` — nothing about them is duration-specific, so grouping by a different column (e.g.
dynamicity) is a small additive change, not a rewrite. `load_reliability_durations` caches its result to a
**hardcoded filename** (`Reliability_{level}_duration.csv`) — any new grouping variant needs its own cache
filename or it'll silently read/overwrite the wrong cached results.

### Model pipeline (encoder side)
`frameBased_encoders.py` and `videoBased_encoders.py` are near-parallel implementations (frame-based
processes each frame independently through an image encoder; video-based feeds full clips through a
native video encoder) that share the same overall shape:

1. `get_encoder(model_name)` — a big dispatch table (`model_registry`) that loads a HF processor/model (or
   torchvision/hub model) per encoder name and returns an `encoder_config` dict describing patch size,
   added tokens, hidden size, and spatial grid — needed downstream to build attention masks (see
   "Conceptual background" above for what these mean).
2. `CustomVideoDataset` (`torch.utils.data.Dataset`, decodes clips with `torchcodec.decoders.VideoDecoder`)
   + `generate_segments` / `collate_fn_all_segments` — sample fixed-length segments per video (always the
   clip's *final* `frames_per_clip * sampling` frames, since `max_segments_per_video=1` and segments are
   built backward from the end of the video), pad short clips with constant gray frames at the *start*
   (real content always at the end), build a mask marking real vs. padding, and apply optional data
   augmentation (see below).
3. A "human-report" head — `FrameEncoder_ForHumanSensoryHistoryReports` /
   `VideoEncoder_ForHumanSensoryHistoryReports` — wraps the frozen backbone and pools per-frame/per-token
   embeddings with `VJEPA2AttentivePoolerMasked` into a per-category prediction vector.
4. `train_ddp(rank, world_size, ...)` / `eval_ddp(...)` — DDP training/eval loops spawned via
   `torch.multiprocessing.spawn`, one process per GPU, gathering predictions across ranks in
   `get_predictions`. Every epoch overwrites `checkpoint_final.pt` with the latest state — there is no
   "keep best epoch by validation MSE" logic. The per-epoch validation loop does not call `model.eval()`
   before running (only `torch.no_grad()`, which doesn't disable dropout) — so per-epoch validation MSE
   includes dropout noise that the final test-set evaluation (which does call `.eval()`) doesn't have.
5. `main()` sweeps a small hyperparameter grid (`learning_rates x weight_decays x frame counts`) per
   `--approach`/`--encoder`, checkpointing under a per-experiment `result_dir` and skipping/resuming based
   on a `status.json` epoch marker. `--approach` maps to a specific augmentation: `tubing` (spatial
   masking), `frame_dropping`/`frame_dropping_2` (whole-frame masking), `speed` (frame-rate jitter),
   `remove_static` (drop near-static short training clips), `upsample_rare_categories` (KDE-weighted
   oversampling for Cat/Dog/Bike), `vanilla_model` (disable even default `AutoAugment`/crop/flip
   augmentation), `baseline` (default recipe).

`eval_model_frame_based.py` / `eval_model_video_based.py` duplicate the model/dataset classes above (they
are not imported from the `*_encoders.py` files) for one-off evaluation of a specific trained checkpoint
against a specific hypothesis condition (frame shuffling, single-frame vs full clip, low-dynamicity-only
subset, etc.). **This duplication is not load-bearing** — all four files' top-level code (outside
`def`/`class`) is side-effect-free besides `model_registry` dicts and setting `CUDA_VISIBLE_DEVICES`, and
`main()`/`train_ddp`/`eval_ddp` sit behind `if __name__ == "__main__":` guards, so `eval_model_*.py` could
safely `from frameBased_encoders import ...` / `from videoBased_encoders import ...` instead of re-pasting
~300 lines. It hasn't been refactored yet. **When editing model/dataset/pooler logic, the fix must be
applied in all four files by hand** — see "Known issues" below for a concrete case where forgetting this
caused real confusion.

`data_augmentation.py` holds the video-side augmentations applied via the mask (not the pixels): tube
masking (`apply_tube_mask(s)`, contiguous rectangular spatial blocks zeroed across all frames, short-range
+ long-range) and frame dropping (`apply_frame_drop`, `apply_frame_drop_2` — the `_2` variant only drops
from frames not already zeroed by padding). These operate on the `(num_segments, T*patches_per_frame)`
attention mask, leaving `clips` pixel data untouched, so the attention pooler ignores masked positions.

### Encoder registry
Frame-based encoders (`frameBased_encoders.py`): DINOv2 (small/base/large/giant, incl. "robust" variants),
DINOv3, ResNet-50 (incl. adversarially-trained `eps*` checkpoints), AlexNet, ViT, ViT-large, I-JEPA,
SigLIP2, ConvNeXt/ConvNeXt-V2 (multiple sizes).
Video-based encoders (`videoBased_encoders.py`): V-JEPA2, X3D (s/m/l), VideoMAE (base/large), VideoMamba
(middle, via external `VideoMamba` repo + HF hub checkpoint download — currently broken, see "Project
overview").

## Evaluation notebook (`evaluation_plots_clean.ipynb`)

A from-scratch, deliberately minimal replacement for the "Prediction Error vs. Video Duration" section of
`evaluation_plots.ipynb`, built up incrementally. Structure (each section: markdown header + code):

- **Setup/config**: `categories` (12 labels), `duration_bins = [0.1, 0.5, 1, 1.5, 2, 3, 4, 6, 8, 15]`
  (uneven `pd.cut` edges — finer near the short end where more clips exist, capped at 15s to match
  `frames_per_clip=60 @ 4fps`), `model_configs` (one entry per predictor: DINOv2/V-JEPA2 × full-clip/
  last-frame/shuffled, each pointing at a prediction CSV).
- **`test_df`**: human full-clip report rates (`load_humanReportRates('test')`) — this is the ground truth
  every predictor gets compared against. Confusingly, the ground-truth column is literally named
  `'prediction'` (name carried over from the reliability code's split-half terminology) — every model's
  own prediction lives in its own `{name}_prediction` column instead.
  `frame_prediction` = human report rates from participants who only saw the clip's **final frame as a
  still image** — the behavioral analog to the models' last-frame condition, used to calibrate how much of
  the models' full-clip-vs-last-frame gap is "expected" (even human judgment degrades without video
  context) vs. model-specific.
  `pred_cols = [cfg['col'] for cfg in model_configs] + ['frame_prediction']` — must include
  `frame_prediction` explicitly, or the human curve silently has no bootstrap data and never renders.
- **Noise floor**: `load_reliability_durations(...)`, filtered to `metric=='MSE', type=='prediction',
  category=='all', Sample size==32` — see `reliability.py` above. Plotted via
  `ax.plot(range(len(nf_mean)), ...)`, which aligns to the duration-bin x-axis **positionally**, not by an
  explicit join — worth double-checking the ordering matches if the noise floor ever looks off.
- **Bootstrap (`df_boots`)**: for 100 repetitions, resample `test_df` rows with replacement *within each
  duration bin*, compute each predictor's MSE against ground truth per bin. Plotted MSE = mean across
  repetitions; error bars = 95% percentile interval (`errorbar=('pi', 95)`) across those repetitions — this
  is bootstrap uncertainty from having a finite set of test videos, not a parametric CI. Crucially, within
  one repetition `b`, *every* predictor's MSE is computed from the same resampled row set — so any two
  predictors' MSE columns for matching `b` are already a valid **paired** comparison (used below for the
  MSE-gap plots, by pivoting `df_boots` rather than re-bootstrapping).
- **Main plot**: color = model (DINOv2/V-JEPA2/Human), linestyle = input condition (solid=full clip,
  dashed=last frame, dotted=shuffled).
- **Low-dynamicity subset**: same pipeline, `test_df` rebuilt with `n_clusters.between(0, 2)` filtered in
  *before* melting — mirrors the notebook-D bug fix below, deliberately not reusing the unfiltered
  `test_df`.
- **Dynamicity plot**: same pipeline, grouped by `dynamicity_bins = pd.cut(n_clusters, bins=[-0.5, 2.5,
  np.inf], labels=['Low dynamicity (0-2 clusters)', 'High dynamicity (3+ clusters)'])` instead of duration
  — 2 x-axis categories, uses `markersize=6` (visible dots) instead of `0` since a 2-point line needs
  markers to actually show data points. No noise floor plotted for this one yet (would need a
  dynamicity-grouped variant of `load_reliability_durations` — architecturally easy, not yet built, see
  "Running the code" note on `reliability.py`'s generic `grouping_variable`).
- **MSE-gap plots** (duration and dynamicity versions): y-axis = `MSE(shuffled) − MSE(full clip)`, one line
  per model, `axhline(0)` reference. Built by **pivoting the existing `df_boots`/`df_boots_dynamicity`**
  (`pivot_table(index=[bin_col, 'b'], columns='predictor', values='mse')`, subtract shuffled minus full
  column-wise) rather than re-bootstrapping — this reuses the paired-resample property above for a
  mathematically correct paired difference at zero extra cost.

## Known issues

### Fixed this session
1. **`VJEPA2AttentivePoolerMasked.forward` never masked its final cross-attention step** (all four files:
   `frameBased_encoders.py`, `videoBased_encoders.py`, `eval_model_frame_based.py`,
   `eval_model_video_based.py`). The self-attention refinement layers correctly received `attention_mask`,
   but `self.cross_attention_layer(queries, hidden_state)` — the call that actually produces the pooled
   vector — never passed it through, even though `VJEPA2PoolerCrossAttentionLayer.forward` (confirmed from
   the actual `transformers` source) accepts and correctly uses it. Net effect: padding tokens leaked
   unmasked into every prediction for any clip needing padding (most of them). Fixed by adding
   `attention_mask=attention_mask` to that call in all four files. **Any checkpoint trained before this fix
   is now train/inference-mismatched** — the pooler's weights were optimized under the old, unmasked
   behavior, so old checkpoints need retraining (cheap: only the pooler+classifier head is trainable, the
   backbone stays frozen either way) before their predictions are trustworthy again.
2. **Video-based tubelet-shuffle experiment scrambled padding-tubelet position, not just real-content
   order** (`eval_model_video_based.py`, `CustomVideoDataset.__getitem__`, the `shuffle_frames` branch).
   `torch.randperm(tubelets)` permuted *all* tubelets including padding ones, which are always contiguous
   at the start otherwise (both in training and in the "ordered" eval). Since the V-JEPA2 backbone itself
   receives no padding mask (only the pooler does), this conflated "does frame order matter" with "does
   moving padding out of its familiar position matter" — a confound that scales with how much padding a
   clip has, i.e. with duration. Fixed to only permute the tubelets flagged real by the mask, leaving
   padding tubelets fixed in place (mirrors the existing real/padding distinction already used in
   `apply_frame_drop_2`, `data_augmentation.py`).
3. **`evaluation_plots.ipynb`'s `low_dynamicity_filter` silently applied to the shuffle-gap analysis**
   (Experiment 3), not just the plots it was meant for (Experiments 1/2). The headline "MSE gap
   (shuffled − ordered)" plot was built from the `test_df` filtered earlier in the notebook, while a later
   diagnostic section reloaded an unfiltered `test_df_full` — two different populations compared as if
   they were the same. Not touched in the old notebook; avoided in `evaluation_plots_clean.ipynb` by never
   introducing a shared mutable filter flag in the first place.

### Flagged, not fixed
- `eval_model_frame_based.py`/`eval_model_video_based.py` duplicate ~300 lines of class/function
  definitions from `frameBased_encoders.py`/`videoBased_encoders.py` instead of importing them (see
  "Architecture" above) — this is what made bug #1 need four separate edits, and directly caused a real
  debugging detour: a "DINOv2 shuffled" CSV generated with the patched pooler code was compared against an
  "ordered" baseline CSV generated by the pre-patch code, producing a large, duration-dependent-looking
  effect that was actually just the two files being out of sync, not a real shuffle effect.
- `train_ddp`'s per-epoch validation loop doesn't call `model.eval()` (only `torch.no_grad()`), so
  per-epoch validation MSE includes active dropout noise. Final test-set evaluation is unaffected (it does
  call `.eval()`).
- No best-checkpoint-by-validation-MSE selection — `checkpoint_final.pt` always holds whatever epoch ran
  last, not the best-performing one.
- `VJEPA2ForVideoClassification` is imported in all four `*_encoders.py`/`eval_model_*.py` files but never
  instantiated — dead import, likely left over from early prototyping before the masked pooler was written.

## Result artifacts
`result_csvs/` holds example prediction/reliability CSVs consumed by the evaluation notebooks. Filenames
encode the experiment config, e.g.
`{encoder}-{encoder}_{frames}frames_{epochs}epochs_lr_{lr}_wd_{wd}_{approach}_fps_{fps}_preds_{frames}Frames_AttentionalPooling.csv`.
`Reliability_all_clusters*.csv` holds the human split-half reliability curves from `reliability.py`.
