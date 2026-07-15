# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

This is a master's thesis codebase studying **human "sensory history" / continuous recognition memory**
(the "SensoryHistory" experiment, on Ego4D video clips) and comparing human behavioral reports against
representations extracted from frozen or fine-tuned neural video/image encoders. The core question across
the experiments is whether/how temporal information (single frame vs. many frames, frame order, playback
speed, event structure) affects how well a model's learned representation predicts human category-report
rates on short video clips.

There is no build system, package manifest, or test suite in this repo — it is a collection of standalone
research scripts run on a SLURM/GPU cluster (paths reference `/braintree/...` and `/orcd/...` HPC mounts)
plus one analysis notebook. Several imports point at files that live outside this repo on the cluster:
- `preprocess_data` (imported by `data_loaders.py`) — not present in this repo.
- `VideoMamba` (imported by `videoBased_encoders.py`) — expected at
  `~/Ego4D/model_optimization/VideoMamba/videomamba/video_sm/models` on the training machine, added to
  `sys.path` at runtime.

Do not try to "fix" missing imports for these — they are intentionally external to this checkout.

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
  code (and by the notebook), not standalone entry points.
- `evaluation_plots.ipynb` consumes the CSVs under `result_csvs/` (and equivalent files produced by the
  eval scripts) to generate the thesis figures, organized by hypothesis (e.g. "Hypothesis 1: Temporal
  Information — Freeze Frame & Single Frame").

Because training/eval assumes cluster paths (`/braintree/data2/active/users/aicha/Ego4D_data`,
`/orcd/data/dicarlo/001/om/lynnka`, `/home/aicha/orcd/pool/Ego4D_data`) and GPU hardware, most of this
code cannot be exercised locally — treat correctness review as static/code-reading rather than
run-and-check unless you're actually on the cluster.

## Architecture

### Data pipeline (human side)
`data_loaders.py` loads human behavioral trial data (button-press "final_choice" reports per video,
per category) from CSV/pickle caches on the cluster, falling back to `preprocess_data` (external) to
build the cache from raw export data if it's missing or `recompute=True` is passed. `final_choice` is
stored as a stringified bool array and parsed back with `utils.string_to_bool_array`; `categories` is a
stringified list parsed with `ast.literal_eval`. `load_trialTypes.py` builds the *trial definitions*
(video IDs, clip start/duration, S3 URLs) for the various experiment conditions: main SH videos, and the
benchmark suites (SensoryHistory duration sweep, EventSegmentation, TemporalDecay/Retention,
ObjectPermanence, Visualization sweep, RSVP, clip-sequence conditions). `reliability.py` computes
split-half reliability of human report rates (correlation/MSE/R²) at the video or aggregate level, with
bootstrap resampling over `File_ID` and category-level breakdowns, and fits reliability-vs-sample-size
curves.

### Model pipeline (encoder side)
`frameBased_encoders.py` and `videoBased_encoders.py` are near-parallel implementations (frame-based
processes each frame independently through an image encoder; video-based feeds full clips through a
native video encoder) that share the same overall shape:

1. `get_encoder(model_name)` — a big dispatch table (`model_registry`) that loads a HF processor/model (or
   torchvision/hub model) per encoder name and returns an `encoder_config` dict describing patch size,
   added tokens (e.g. CLS), hidden size, and spatial grid — needed downstream to build attention masks.
2. `CustomVideoDataset` (`torch.utils.data.Dataset`, decodes clips with `torchcodec.decoders.VideoDecoder`)
   + `generate_segments` / `collate_fn_all_segments` — sample fixed-length segments per video, apply
   optional data augmentation (see below), and produce a padded clip/mask/label batch.
3. A "human-report" head — `FrameEncoder_ForHumanSensoryHistoryReports` /
   `VideoEncoder_ForHumanSensoryHistoryReports` — wraps the frozen/finetuned backbone and pools
   per-frame/per-token embeddings with a `VJEPA2AttentivePoolerMasked` (a masked variant of HF's
   `VJEPA2AttentivePooler`) into a per-category prediction vector, trained to regress human report rates.
4. `train_ddp(rank, world_size, ...)` / `eval_ddp(...)` — DDP training/eval loops spawned via
   `torch.multiprocessing.spawn`, one process per GPU, gathering predictions across ranks in
   `get_predictions`.
5. `main()` sweeps a small hyperparameter grid (`learning_rates x weight_decays x frame counts`) per
   `--approach`/`--encoder`, checkpointing under a per-experiment `result_dir` and skipping/resuming based
   on a `status.json` epoch marker.

`eval_model_frame_based.py` / `eval_model_video_based.py` duplicate the model/dataset classes above (they
are not imported from the `*_encoders.py` files) for one-off evaluation of a specific trained checkpoint
against a specific hypothesis condition (frame shuffling, single-frame vs full clip, low-dynamicity-only
subset, etc.) — when editing model/dataset logic, check whether the same fix is needed in both the
`_encoders.py` file and the corresponding `eval_model_*.py` file.

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
(middle, via external `VideoMamba` repo + HF hub checkpoint download).

### Result artifacts
`result_csvs/` holds example prediction/reliability CSVs consumed by `evaluation_plots.ipynb`. Filenames
encode the experiment config, e.g.
`{encoder}-{encoder}_{frames}frames_{epochs}epochs_lr_{lr}_wd_{wd}_{approach}_fps_{fps}_preds_{frames}Frames_AttentionalPooling.csv`.
`Reliability_all_clusters*.csv` holds the human split-half reliability curves from `reliability.py`.
