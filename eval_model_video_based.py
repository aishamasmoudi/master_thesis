import os
import argparse
from platform import processor

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
import torch
import json
from pathlib import Path
from typing import Callable, Optional, Union
from torchcodec.decoders import VideoDecoder
from transformers import AutoVideoProcessor, AutoModel, AutoConfig
from transformers import VJEPA2PreTrainedModel, VJEPA2ForVideoClassification, VJEPA2VideoProcessor, VJEPA2Config
from transformers import AutoImageProcessor, ResNetForImageClassification, ViTForImageClassification, IJepaForImageClassification, AutoModelForImageClassification, ConvNextForImageClassification, ConvNextV2ForImageClassification
import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download
import tarfile
import pathlib
from torch.utils.data import Dataset, DataLoader
from torchcodec.samplers import clips_at_random_indices, clips_at_regular_indices
from torchvision.transforms import v2
from functools import partial
from scipy.stats import gaussian_kde
from transformers.modeling_outputs import ImageClassifierOutput
from torch import nn
import math
# Import distributed training modules
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from sklearn.random_projection import SparseRandomProjection

from transformers.models.vjepa2.modeling_vjepa2 import VJEPA2AttentivePooler, VJEPA2PoolerCrossAttentionLayer, VJEPA2PoolerSelfAttentionLayer
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.multiprocessing as mp
import time
import datetime
from transformers import VJEPA2Model
from transformers import VideoMAEImageProcessor, VideoMAEModel
from transformers import TimesformerModel

class VJEPA2AttentivePoolerMasked(nn.Module): # neural network module
    """
    Attentive Pooler:
    - Takes a sequence of embeddings (input = many vectors, one per frame/token)
    - Optionally builds an attention mask so padded/invalid positions are ignored
    - Runs several self-attention alyers over the sequence
    - Creates one learned query token per example
    - Uses cross-attention from that query to the sequence
    - Produces one pooled vector per example (output = one signle vector that represents the entire input)
    """

    def __init__(self, config: VJEPA2Config):
        super().__init__()
        # query_tokens: trainable tensor of shape (placeholder batch, one query token, embedding dimension),
        # later used to extract the most relevant information into one output vector (through cross-attention)
        self.query_tokens = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        # cross_attention_layers: this layer will take the learned query token as the query,
        # and the processed hidden_state sequence as the keys and values
        self.cross_attention_layer = VJEPA2PoolerCrossAttentionLayer(config)
        # self_attention_layers: list of self-attention layers
        # these process (global, context-aware representation) the input sequence before the final pooling query is applied
        self.self_attention_layers = nn.ModuleList(
            [VJEPA2PoolerSelfAttentionLayer(config) for _ in range(config.num_pooler_layers)]
        )

    def forward(self, hidden_state: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,) -> torch.Tensor:

        # below, each token/frame representation in hidden_state can interact with the others through self-attention
        # so each position gets updated using information from other positions in the sequence, while repsecting the mask
        if attention_mask is not None:
            # Step 1: Expand dimensions
            attention_mask = attention_mask[:, None, None, :]
            # Step 2: Convert to additive form
            attention_mask = (1.0 - attention_mask) * torch.finfo(hidden_state.dtype).min * 0.0005

        for layer in self.self_attention_layers:
            hidden_state = layer(hidden_state, attention_mask=attention_mask)[0] # each self-attention layer processes the whole sequence

        # below, the model learns a pooling function that can focus on the most relevant sequence elements (rather than avg/max pooling)
        # because the pooling operation itself is trainable
        queries = self.query_tokens.repeat(hidden_state.shape[0], 1, 1)
        hidden_state = self.cross_attention_layer(queries, hidden_state)[0]
        return hidden_state.squeeze(1)

from data_augmentation import apply_tube_masks, apply_frame_drop, apply_frame_drop_2

class VideoEncoder_ForHumanSensoryHistoryReports(VJEPA2PreTrainedModel):
    def __init__(self, encoder, config: VJEPA2Config, encoder_config: dict):
        super().__init__(config)

        self.encoder_config = encoder_config
        self.model_name = self.encoder_config["model_name"]
        self.num_labels = config.num_labels
        self.encoder = encoder
        # Classifier head
        self.pooler = VJEPA2AttentivePoolerMasked(config)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels, bias=True)
        # Initialize weights and apply final processing
        self.post_init()

        self.loss_type = 'ForVideoClassification'
        self.config.problem_type = 'multi_label_classification'
        self.loss_function_manual = nn.MSELoss()#nn.BCEWithLogitsLoss(weight=config.weight)

    def forward(
            self,
            pixel_values_videos: torch.Tensor,
            labels: Optional[torch.Tensor] = None,
            context_masks: Optional[torch.Tensor] = None,
            output_attentions: Optional[bool] = False,
            output_hidden_states: Optional[bool] = False,
    ) -> Union[tuple, ImageClassifierOutput]:
        """
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the image classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """

        if self.model_name == "vjepa2":
            outputs = self.encoder(pixel_values_videos=pixel_values_videos, output_hidden_states=True)
            last_hidden_states = outputs.last_hidden_state
        elif self.model_name.startswith("videomae"):
            outputs = self.encoder(pixel_values=pixel_values_videos)
            last_hidden_states = outputs.last_hidden_state
        elif self.model_name.startswith("videomamba"):
            # VideoMambaFeatureExtractor.forward returns the raw patch-token
            # tensor [B, L, C] directly (no HF-style output object, no CLS)
            last_hidden_states = self.encoder(pixel_values_videos)
        elif self.model_name.startswith('x3d'):
            # X3DFeatureExtractor.forward handles permute, blocks, and reshape internally
            x = self.encoder(pixel_values_videos)  # → [B, T'*H'*W', C]
            B, N, C = x.shape
            
            # Recover T', H', W' for mask expansion
            spatial = self.encoder_config["feature_map_size"]
            T = N // (spatial * spatial)
            H = W = spatial

            context_masks = context_masks.unsqueeze(-1)  
            context_masks = context_masks.expand(-1, -1, spatial * spatial)
            context_masks = context_masks.reshape(B, N)
                
            last_hidden_states = x

        pooler_output = self.pooler(last_hidden_states, attention_mask=context_masks)
        logits = self.classifier(pooler_output)

        #logits = torch.sigmoid(logits)

        loss = None
        if labels is not None:
            loss = self.loss_function_manual(logits, labels)
            #loss = self.loss_function(pooled_logits=logits, labels=labels.unsqueeze(0), config=self.config)


        return ImageClassifierOutput(
            loss=loss,
            logits=logits
            #hidden_states=outputs.hidden_states,
            #attentions=outputs.attentions,
        )


def generate_segments(video_frames, frames_per_clip, sampling=1, frame_jitter=0):
    """
    Generates lists of frame indices (i.e. "timestamps" to create small video clips or segments) by slicing the video, with optional jittering
    - video_frames is the total number of frames in the video (e.g. video is 10 s, 30 FPS = 300 frames)
    - frames_per_clip is the max length of each segment
    """
    segments = []
    start = video_frames
    while start > 0:
        end = max(start - frames_per_clip * sampling, 0)
        segments.append(np.sort(np.arange(start, end, -sampling)) - 1)
        start = end

        # jittering = adding noise to values (small swaps in frame ordering)
        if frame_jitter > 0:
            jitter_frames = np.random.randint(-frame_jitter, frame_jitter+1, len(segments[-1]) )
            segments[-1] += jitter_frames
            segments[-1] = np.clip(segments[-1], 0, video_frames-1) # ensure that it fall inside the video for the last and first frame.

    return segments


def collate_fn_all_segments(samples, transforms=None, tube_mask_ratio=0.0, frame_drop_ratio=0.0, frame_drop_ratio_2=0.0):
    """
    Simplified collate function - decoding is already done in workers
    """
    clips_list, labels, ids, masks = [], [], [], []

    for clips, lbl, vid_id, mask in samples:
        # Apply tube masking
        if tube_mask_ratio > 0.0:
            clips, mask = apply_tube_masks(clips, mask, tube_mask_ratio=tube_mask_ratio)
        # Apply frame dropping via attention mask (pooler ignores dropped frames entirely)
        if frame_drop_ratio > 0.0:
            clips, mask = apply_frame_drop(clips, mask, frame_drop_ratio=frame_drop_ratio)
        if frame_drop_ratio_2 > 0.0:
            print("Applying Frame Dropping")
            clips, mask = apply_frame_drop_2(clips, mask, frame_drop_ratio=frame_drop_ratio_2)

        # Apply transforms if provided
        if transforms is not None:
            clips = transforms(clips)

        clips_list.append(clips)
        labels.append(torch.tensor(lbl, dtype=torch.float32))
        masks.append(mask)
        ids.append(vid_id)

    # Stack labels
    labels = torch.stack(labels, dim=0)

    return clips_list, labels, ids, masks

def get_predictions(
        loader: DataLoader,
        model: nn.Module,
        processor: VJEPA2VideoProcessor,
        device: torch.device,
        rank: int = 0,
        max_features= None,
        random_state: int = 42,
        interpolate_pos_encoding: bool = False,
        decay_rate=None
) -> tuple:
    """Compute predictions and gather them from all GPUs."""
    model.eval()
    all_activations = []
    all_ids = []
    all_labels = []
    if max_features is not None:
        proj = SparseRandomProjection(n_components=max_features, random_state=random_state)

    #MSE, total = 0, 0
    with torch.no_grad():
        for step, (vids, labels, ids, masks) in enumerate(loader):
            if step % 5 == 0:
                print(f"[get_predictions] step {step}", flush=True)
            labels = labels.to(device)
            # vids = torch.stack(
            #     vids).squeeze()  # (batch, frames, channel, height, width)
            masks = torch.cat(masks, dim=0).to(device)
            inputs = torch.cat(vids, dim=0).to(device)
            #inputs = [processor(vid.squeeze(), return_tensors="pt").to(device) for vid in vids]
            outputs = model(pixel_values_videos=inputs, context_masks=masks)

            activations = outputs.logits

            all_activations.append(activations)
            all_ids.append(ids)
            all_labels.append(labels)


    # Concatenate all local predictions
    if all_activations:
        local_activations = torch.cat(all_activations, dim=0)
        local_labels = torch.cat(all_labels, dim=0)
        # IDs might be strings/ints, so handle them separately
        if isinstance(all_ids[0], torch.Tensor):
            local_ids = torch.cat(all_ids, dim=0)
        else:
            # If IDs are lists or other types, flatten them
            local_ids = [item for sublist in all_ids for item in sublist]

    else:
        # Handle empty case
        local_activations = torch.empty(0, device=device)
        local_labels = torch.empty(0, device=device)
        local_ids = []

    # Gather results from all GPUs
    # this is necessary because each GPU sees part of the dataset
    # if we don't gather them, each process only has partial results
    if dist.is_initialized():

        world_size = dist.get_world_size()

        # Gather logits
        gathered_activations = [torch.empty_like(local_activations) for _ in range(world_size)]
        dist.all_gather(gathered_activations, local_activations)
        all_activations_combined = torch.cat(gathered_activations, dim=0)

        # Gather labels
        gathered_labels = [torch.empty_like(local_labels) for _ in range(world_size)]
        dist.all_gather(gathered_labels, local_labels)
        all_labels_combined = torch.cat(gathered_labels, dim=0)

        # Gather IDs (more complex since they might not be tensors)
        if isinstance(local_ids, torch.Tensor):
            gathered_ids = [torch.empty_like(local_ids) for _ in range(world_size)]
            dist.all_gather(gathered_ids, local_ids)
            all_ids_combined = torch.cat(gathered_ids, dim=0)
        else:
            # Use all_gather_object for non-tensor data
            gathered_ids = [None for _ in range(world_size)]
            dist.all_gather_object(gathered_ids, local_ids)
            all_ids_combined = [item for sublist in gathered_ids for item in sublist]

        return all_activations_combined, all_ids_combined, all_labels_combined
    else:
        return local_activations, local_ids, local_labels

class CustomVideoDataset(Dataset):
    """
    Modified Dataset that performs video decoding in __getitem__ (worker process)
    """

    def __init__(self, video_file_paths, labels, frames_per_clip, sampling=1,
                 frame_jitter=0, max_segments=8, tubelet_size=1, patch_size=16,
                 added_tokens=1, fill_value=128, addFolderToID=None,
                 decoder_seek_mode = "exact", device=None, num_ffmpeg_threads=None,
                 encoder_config=None):
        self.video_file_paths = video_file_paths
        self.labels = labels
        self.frames_per_clip = frames_per_clip
        self.sampling = sampling
        self.frame_jitter = frame_jitter
        self.max_segments = max_segments
        self.tubelet_size = tubelet_size
        self.patch_size = patch_size
        self.added_tokens = added_tokens
        self.fill_value = fill_value
        self.addFolderToID = addFolderToID
        self.encoder_config = encoder_config


        self.decoder_seek_mode = decoder_seek_mode
        self.decoder_device = device
        self.decoder_num_ffmpeg_threads =num_ffmpeg_threads

    def __len__(self):
        return len(self.video_file_paths)

    def __getitem__(self, idx):
        video_path = self.video_file_paths[idx]
        label = self.labels[idx]
        video_id = self._get_video_id(video_path)

        # Decode video in worker process
        num_ffmpeg_threads = (
            int(self.decoder_num_ffmpeg_threads) if self.decoder_num_ffmpeg_threads else 0
        )
        #with set_cuda_backend("beta"):
        decoder = VideoDecoder(video_path, seek_mode=self.decoder_seek_mode, device=self.decoder_device, num_ffmpeg_threads=num_ffmpeg_threads)

        num_frames = decoder.metadata.num_frames

        segments = generate_segments(
            num_frames,
            self.frames_per_clip,
            self.sampling,
            self.frame_jitter
        )
        segments = segments[:self.max_segments]

        # Decode all segments in worker
        clips = []
        masks = []
        patches = self.encoder_config["num_patches"]
        tubelets = self.frames_per_clip // self.tubelet_size

        for seg in segments[::-1]:
            video_frames = decoder.get_frames_at(indices=seg).data
            n_real_original = video_frames.shape[0]

            if len(seg) < self.tubelet_size:
                video_frames = torch.cat([video_frames, video_frames], dim=0)
            if self.encoder_config["model_name"].startswith("x3d"):
                n_real = n_real_original
                n_padded = self.frames_per_clip - n_real
                mask = torch.zeros(self.frames_per_clip)
                mask[n_padded:] = 1.0
            else:
                mask_idx = torch.arange(
                    math.ceil((self.frames_per_clip - video_frames.shape[0]) / self.tubelet_size) * patches,
                    tubelets * patches
                )
                mask = torch.zeros((tubelets * patches,))
                mask[mask_idx] = 1

            # Pad if necessary
            if video_frames.shape[0] < self.frames_per_clip:
                missing_frames = self.frames_per_clip - video_frames.shape[0]
                padding = self.fill_value * torch.ones(
                    (missing_frames, *video_frames.shape[1:]),
                    dtype=video_frames.dtype
                )
                video_frames = torch.cat([padding, video_frames], dim=0)

            # Experiment 6: shuffle TUBELET order (groups of tubelet_size consecutive
            # frames), not individual frames -- keeps each tubelet's internal frame
            # pairing intact and only permutes when each 2-frame chunk occurs in time.
            # Same permutation applied to the mask so padded positions stay correctly flagged.
            #
            # Only the REAL (non-padded) tubelets are permuted among themselves; padding
            # tubelets are left at their original (always-leading) slots. The encoder
            # itself is never given a padding mask (see forward()), so it self-attends
            # over the whole clip including padding -- if padding were shuffled along
            # with real content, the experiment would conflate "does frame order matter"
            # with "does moving padding out of its training-time position matter",
            # since training/baseline eval always keeps padding contiguous at the start.
            if self.encoder_config.get("shuffle_frames", False):
                if self.encoder_config["model_name"].startswith("x3d"):
                    real_tubelet_mask = mask.view(tubelets, self.tubelet_size).any(dim=1)
                else:
                    real_tubelet_mask = mask.view(tubelets, patches).any(dim=1)
                real_idx = real_tubelet_mask.nonzero(as_tuple=True)[0]
                perm = torch.arange(tubelets)
                perm[real_idx] = real_idx[torch.randperm(len(real_idx))]

                video_frames = video_frames.view(
                    tubelets, self.tubelet_size, *video_frames.shape[1:]
                )[perm].reshape(video_frames.shape[0], *video_frames.shape[1:])
                if self.encoder_config["model_name"].startswith("x3d"):
                    mask = mask.view(tubelets, self.tubelet_size)[perm].reshape(-1)
                else:
                    mask = mask.view(tubelets, patches)[perm].reshape(-1)

            clips.append(video_frames)
            masks.append(mask)

        # Stack clips (num_segments, T, C, H, W)
        clips = torch.stack(clips, dim=0)
        # Stack masks (num_segments, patches*tubelets)
        masks = torch.stack(masks, dim=0)

        del decoder

        return clips, label, video_id, masks

    def _get_video_id(self, video_path):
        video_id = str(video_path).split("/")[-1].split(".")[0]
        if self.addFolderToID == -2:
            video_id = os.path.join(str(video_path).split("/")[-2], video_id)
        elif self.addFolderToID == -3:
            video_id = os.path.join(str(video_path).split("/")[-3], str(video_path).split("/")[-2], video_id)

        return video_id


def setup(rank, world_size):
    """Initialize the distributed environment."""
    os.environ['MASTER_ADDR'] = 'localhost' # set master address, here same machine
    os.environ['MASTER_PORT'] = '12355' # communication port
    # CRITICAL: Set device before init_process_group
    torch.cuda.set_device(rank) # assign GPU to this process/rank (each process uses a different GPU)
    # Initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size) # creates communication between processes

world_size = torch.cuda.device_count()
# Single-GPU eval: no dist.init_process_group() here. setup() is designed to be
# called once per spawned process (each with its own rank), as in eval_ddp via
# mp.spawn in frameBased_encoders.py. Calling it once by hand with rank=0 while
# world_size=4 will hang forever waiting for ranks 1-3 to also call it.
# CHANGE ENGAING/BRAINTREE
hypothesis = "v2_ordered"
model_name = "vjepa2"
approach = "baseline"
model_ID = "facebook/vjepa2-vitl-fpc64-256"
checkpoint_path = f"/braintree/home/aicha/Ego4D/model_eval/checkpoitns/vjepa2/checkpoint_final.pt"
base_dir = "/braintree/data2/active/users/aicha/Ego4D_data"
low_dynamicity_only = False
shuffle_frames = False
n_permutations = 10  # number of independent random shuffles to average over
n_permutations = n_permutations if shuffle_frames else 1
one_frame = False
#base_dir = "/home/aicha/orcd/pool/Ego4D_data"
#result_dir = f'/orcd/data/dicarlo/001/om/lynnka/{model_name}_{approach}_aicha'
#if not os.path.exists(result_dir):
#    os.makedirs(result_dir, exist_ok=True)
if base_dir.startswith('/orcd'):
    dataset_root_path = pathlib.Path("/home/lynnka/orcd/pool/Ego4D_videos/")
elif base_dir.startswith('/braintree'):
    dataset_root_path = pathlib.Path("/braintree/data2/active/users/aicha/Ego4D_data")
elif base_dir.startswith('/home'):
    dataset_root_path = pathlib.Path("/home/aicha/orcd/pool/Ego4D_data")
else:
    raise ValueError('dataset root directory not defined')
if one_frame:
    frame = 1
else:
    frame = 60
num_epochs = 15
lr = 0.001
wd = 0.01
experiment_id = f'{model_name}_{frame}frames_{num_epochs}epochs_lr_{lr}_wd_{wd}_{approach}_fps_4'
if base_dir.startswith('/braintree'):
    exp_results_dir = f'/braintree/home/aicha/part_2/{model_name}_{hypothesis}'
else:
    exp_results_dir = f'{base_dir}/{model_name}_{hypothesis}'
os.makedirs(exp_results_dir, exist_ok=True)

# CONFIG
approach = "baseline"
fps = 4
sampling = 30 // fps
max_segments_per_video = 1
num_workers = 4
batch_size = 6
prefetch_factor = 2

# FOR test_ds
categories = ['Cup', 'Knife', 'Chair', 'Person', 'Car', 'Bike', 'Dog', 'Cat', 'Table', 'Book', 'Plant', 'Bed']
test_df = pd.read_csv(dataset_root_path / "test_humanReports_rebalanced.csv")

if low_dynamicity_only:
    test_df = test_df.loc[test_df['n_clusters'].between(0, 2)].reset_index(drop=True)
    print(f"Videos after low-dynamicity filter: {len(test_df)}")

test_video_file_paths = dataset_root_path / (test_df['stimulus_video_url'].str.split('/',expand=True)[4]) / (test_df['stimulus_video_url'].str.split('/',expand=True)[5]) 
test_labels = test_df[categories].values

if model_name == 'vjepa2':
    processor = VJEPA2VideoProcessor.from_pretrained(model_ID,
                                                        use_fast=True,
                                                        do_center_crop=False,
                                                        do_resize=True,
                                                        crop_pct=1,
                                                        size={"shortest_edge": 224}, # This doesn't work
                                                        )

    model = VJEPA2Model.from_pretrained(model_ID)

    encoder_config = {
        "model_name": model_name,
        "patch_size": 16,
        "added_tokens": 0,
        "hidden_size": model.config.hidden_size,
        "num_patches": None,
        "image_size": 224,
        "feature_map_size": None,
        "tubelet_size": 2,
        "shuffle_frames": shuffle_frames,
    }

    encoder_config["num_patches"] = encoder_config["added_tokens"] + encoder_config["patch_size"] ** 2
    
else:
    processor = None
    model = None
    encoder_config = None

tubelet_size = encoder_config["tubelet_size"]
added_tokens = encoder_config["added_tokens"]
patch_size = encoder_config["patch_size"]

if one_frame:
    frames_per_clip = tubelet_size #when running with 1 frame
else:
    frames_per_clip = 60

mean = processor.image_mean
std  = processor.image_std
resample = processor.resample

if model_name == "vjepa2":
    size = 256
else:
    size = None

eval_transforms = v2.Compose([
    v2.Resize((size, size), interpolation=resample),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=mean, std=std)
])

device = torch.device(f"cuda:{0}")
config = AutoConfig.from_pretrained("facebook/vjepa2-vitl-fpc64-256")
config.hidden_size = encoder_config["hidden_size"]
config.num_labels = 12
model_SH = VideoEncoder_ForHumanSensoryHistoryReports(model, config, encoder_config).to(device)

print(f"Loading checkpoint from {checkpoint_path}")
checkpoint = torch.load(checkpoint_path, map_location=device)
model_SH.load_state_dict(checkpoint['model_state_dict'])

all_predictions = []
test_ids_ref = None
test_labels_ref = None

for perm_idx in range(n_permutations):
    print(f"--- Permutation {perm_idx + 1}/{n_permutations} ---", flush=True)
    torch.manual_seed(1000 + perm_idx)  # different, reproducible shuffle each pass

    test_ds = CustomVideoDataset(
        test_video_file_paths.tolist(),
        test_labels,
        frames_per_clip=frames_per_clip,
        sampling=sampling,
        frame_jitter=0,
        max_segments=max_segments_per_video,
        decoder_seek_mode='approximate',
        tubelet_size=tubelet_size,
        patch_size=patch_size,
        added_tokens=added_tokens,
        num_ffmpeg_threads=2,
        encoder_config=encoder_config
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=partial(collate_fn_all_segments, transforms=eval_transforms),
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=prefetch_factor,
        persistent_workers=True if num_workers > 0 else False,
    )

    preds, ids, labels_out = get_predictions(test_loader, model_SH, processor, device, 0)

    seed_df = pd.DataFrame(preds.cpu().numpy(), columns=categories)
    seed_df['videoID'] = ids
    seed_df['seed'] = perm_idx
    seed_df = seed_df.melt(id_vars=['videoID', 'seed'], value_vars=categories,
                            var_name='category', value_name='prediction')
    seed_df.to_csv(f'{exp_results_dir}/vjepa2_shuffled_seed{perm_idx}.csv', index=False)
    print(f'Saved shuffled results for seed {1000 + perm_idx} to {exp_results_dir}/vjepa2_shuffled_seed{perm_idx}.csv')

    # Sanity check: with shuffle=False and a fixed video list, order should be identical
    # across iterations. If it isn't, averaging by position would silently misalign videos.
    if test_ids_ref is None:
        test_ids_ref = ids
        test_labels_ref = labels_out.cpu()
    else:
        assert list(ids) == list(test_ids_ref), (
            f"Video order changed on permutation {perm_idx} -- predictions would "
            f"misalign if averaged. Aborting."
        )

    all_predictions.append(preds.cpu())

    del test_loader, test_ds
    torch.cuda.empty_cache()

# Average predictions across the n_permutations independent shuffles
test_predictions = torch.stack(all_predictions, dim=0).mean(dim=0)
test_ids = test_ids_ref
test_labels = test_labels_ref

loss_func = nn.MSELoss()
test_mse = loss_func(test_labels, test_predictions)
print(f"Test MSE ({n_permutations}-permutation average): {test_mse:.4f}")

test_predictions = test_predictions.cpu().numpy()
test_labels = test_labels.cpu().numpy()
results = []

for i, cat in enumerate(categories):
    results.append(pd.DataFrame({'videoID': test_ids, 'category': cat,
            'prediction': test_predictions[:, i], 'video_type': 'test',
            'labels': test_labels[:, i]}))
    
results = pd.concat(results)
results.to_csv(
    f'{exp_results_dir}/{hypothesis}_{model_name}-{experiment_id}_preds_{frames_per_clip}Frames_AttentionalPooling.csv'
)

print("Saved results to:", f'{exp_results_dir}/{hypothesis}_{model_name}-{experiment_id}_preds_{frames_per_clip}Frames_AttentionalPooling.csv')