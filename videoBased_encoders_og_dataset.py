import os
import argparse
from platform import processor

#os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
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

import sys, os
VIDEOMAMBA_MODELS_DIR = os.path.expanduser(
    "~/Ego4D/model_optimization/VideoMamba/videomamba/video_sm/models"
)
if VIDEOMAMBA_MODELS_DIR not in sys.path:
    sys.path.insert(0, VIDEOMAMBA_MODELS_DIR)

from VideoMamba.videomamba.video_sm.models.videomamba import videomamba_middle

from data_augmentation import apply_frame_drop_2

model_registry = {'vjepa2': "facebook/vjepa2-vitl-fpc64-256",
                  'vjepa2_base': "facebook/vjepa2-vith-fpc64-256",
                  # size variation vjepa2
                  'videomae_b': "MCG-NJU/videomae-base",
                  'videomae_l': 'MCG-NJU/videomae-large',
                  'videomae_v2_b': "OpenGVLab/VideoMAEv2-Base",
                  'videomae_v2_l': 'OpenGVLab/VideoMAEv2-Large',
                  'videomae_v2_g': 'OpenGVLab/VideoMAEv2-giant',
                  'x3d_s':"facebookresearch/pytorchvideo",
                  'x3d_m':"facebookresearch/pytorchvideo",
                  'x3d_l':"facebookresearch/pytorchvideo",
                  # VideoMamba-Middle (74M params): largest officially released size
                  # (Base, ~98M, was excluded by the authors for overfitting). Weights
                  # are a plain HF Hub file, not an AutoModel-compatible repo.
                  'videomamba_m': "OpenGVLab/VideoMamba",
                  }


def get_encoder(model_name):
    model_ID = model_registry[model_name]

    encoder_config = {
        "model_name": model_name,
        "patch_size": None,
        "added_tokens": 0,
        "hidden_size": None,
        "num_patches": None,
        "image_size": 224,
        "feature_map_size": None,
        "tubelet_size": 1,
    }

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
        }

        encoder_config["num_patches"] = encoder_config["added_tokens"] + encoder_config["patch_size"] ** 2

    elif model_name.startswith('x3d'):
        base_model = torch.hub.load(
            "facebookresearch/pytorchvideo",
            model=model_name,
            pretrained=True
        )

        class X3DFeatureExtractor(nn.Module):
            """
            Strips the classification head (blocks[5]) from X3D and
            returns a token sequence [B, T'*H'*W', C] compatible with
            the attentive pooler.
            """
            def __init__(self, model):
                super().__init__()
                # blocks[0]=stem, blocks[1-4]=4 residual stages, blocks[5]=head
                self.blocks = nn.Sequential(*list(model.blocks[:-1]))

            def forward(self, x):
                # x: [B, T, C, H, W]  →  permute to [B, C, T, H, W] for 3D convs
                x = x.permute(0, 2, 1, 3, 4)
                x = self.blocks(x)          # [B, 192, T', H', W']
                B, C, T, H, W = x.shape
                # Reshape to token sequence: [B, T'*H'*W', C]
                x = x.permute(0, 2, 3, 4, 1).reshape(B, T * H * W, C)
                return x

        model = X3DFeatureExtractor(base_model)
        processor = None

        if model_name == 'x3d_s':
            # 13 input frames, 160×160, 4 spatial stride-2 stages → 5×5
            T_out = 13   # temporal stride=1 across stages, so T' ≈ T_in
            spatial = 5
            image_size = 160
        elif model_name == 'x3d_m':
            # 16 input frames, 224×224 → 7×7
            T_out = 16
            spatial = 7
            image_size = 224
        elif model_name == 'x3d_l':
            # 16 input frames, 312×312 → 10×10
            T_out = 16
            spatial = 10
            image_size = 312

        encoder_config = {
            "model_name": model_name,
            "patch_size": None,
            "added_tokens": 0,
            "hidden_size": 192,
            "num_patches": T_out * spatial * spatial,  # total tokens
            "image_size": image_size,
            "feature_map_size": spatial,
            "tubelet_size": 1,
        }

    elif model_name == 'videomae_v2_b':
        config = AutoConfig.from_pretrained("OpenGVLab/VideoMAEv2-Base", trust_remote_code=True)
        processor = VideoMAEImageProcessor.from_pretrained("OpenGVLab/VideoMAEv2-Base")
        model = AutoModel.from_pretrained('OpenGVLab/VideoMAEv2-Base', config=config, trust_remote_code=True)

        encoder_config = {
            "model_name": model_name,
            "patch_size": 16,
            "added_tokens": 0,
            "hidden_size": config.hidden_size,
            "num_patches": None,
            "image_size": 224,
            "feature_map_size": None,
            "tubelet_size": 2,
        }

        grid = 224 // encoder_config["patch_size"]
        encoder_config["num_patches"] = grid * grid + encoder_config["added_tokens"]

    elif model_name in ('videomae_b', 'videomae_l'):
        # patch_size/tubelet_size are architectural constants shared by both VideoMAE
        # v1 variants; hidden_size is read dynamically from the loaded model below, so
        # this branch needs no other change to cover 'videomae_l' (MCG-NJU/videomae-large)
        # -- it was previously only reachable for 'videomae_b', despite being registered
        # in model_registry and listed as a valid --encoder choice (NotImplementedError
        # at the bottom of this function would have fired for videomae_l before this fix).
        processor = VideoMAEImageProcessor.from_pretrained(model_ID)
        model = VideoMAEModel.from_pretrained(model_ID)

        encoder_config = {
            "model_name": model_name,
            "patch_size": 16,
            "added_tokens": 0,
            "hidden_size": model.config.hidden_size,
            "num_patches": None,
            "image_size": 224,
            "feature_map_size": None,
            "tubelet_size": 2,
        }

        grid = 224 // encoder_config["patch_size"]
        encoder_config["num_patches"] = grid * grid + encoder_config["added_tokens"]

    elif model_name == 'videomamba_m':
        # NOTE: unlike the other branches, VideoMamba is not on the HF `transformers`
        # AutoModel path. Requires:
        #   1. Cloning the repo so `videomamba.video_sm.models.videomamba` is
        #      importable: https://github.com/OpenGVLab/VideoMamba
        #   2. `mamba_ssm` + `causal-conv1d` (CUDA-compiled) installed/importable.
        # Uses VideoMamba-Middle (embed_dim=576, depth=32, 74M params) -- the
        # largest officially released size (Base, ~98M, was dropped by the authors
        # for overfitting) -- with the masked-pretrained + K400-finetuned checkpoint,
        # 16 frames, matching the pretraining lineage used for videomae_b.

        num_frames = 16  # must match frames_per_clip for this branch (see main())

        base_model = videomamba_middle(
            num_frames=num_frames,
            num_classes=400,  # matches this checkpoint's K400 head shape
        )

        ckpt_path = hf_hub_download(
            repo_id="OpenGVLab/VideoMamba",
            filename="videomamba_m16_k400_mask_ft_f16_res224.pth",
        )
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        state_dict = checkpoint.get("model", checkpoint)
        base_model.load_state_dict(state_dict, strict=True)

        class VideoMambaFeatureExtractor(nn.Module):
            """
            Returns the full patch-token sequence (drops [CLS]) as [B, L, C],
            compatible with the attentive pooler.

            VideoMamba's own forward_features() computes the correct full-sequence
            hidden states internally but only *returns* the pooled [CLS] token
            (`return hidden_states[:, 0, :]`). Rather than re-deriving the internal
            patch_embed/pos_embed/temporal_pos_embed plumbing by hand (fragile --
            easy to get subtly wrong), we let the model run its own real forward
            pass and grab the pre-slice tensor via a forward hook on the final
            norm layer (`norm_f`), which computes the full [B, 1+L, C] sequence
            right before that CLS-only slice happens.
            """
            def __init__(self, model):
                super().__init__()
                self.model = model
                self._features = None
                # sanity-checked assumption: verify `norm_f` exists on your
                # installed version via `print(base_model)` if this raises
                self.model.norm_f.register_forward_hook(self._capture)

            def _capture(self, module, inputs, output):
                self._features = output  # [B, 1+L, C], full sequence post-norm

            def forward(self, x):
                # x: [B, C, T, H, W] -- VideoMamba's native input layout
                self.model.forward_features(x)  # runs the real forward; return value unused
                return self._features[:, 1:, :]  # drop [CLS] -> [B, L, C]

        model = VideoMambaFeatureExtractor(base_model)
        processor = None  # no HF processor; normalization handled manually (see train_ddp/eval_ddp)

        patch_size = 16
        grid = 224 // patch_size  # 14

        encoder_config = {
            "model_name": model_name,
            "patch_size": patch_size,
            "added_tokens": 0,
            "hidden_size": 576,  # VideoMamba-Middle embed_dim
            "num_patches": grid * grid,  # per-frame patch count; tubelet_size=1
            "image_size": 224,
            "feature_map_size": grid,
            "tubelet_size": 1,
        }

    else:
        raise NotImplementedError

    return model, processor, encoder_config

class VJEPA2AttentivePoolerMasked(nn.Module):
    """Attentive Pooler"""

    def __init__(self, config: VJEPA2Config):
        super().__init__()
        self.query_tokens = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        self.cross_attention_layer = VJEPA2PoolerCrossAttentionLayer(config)
        self.self_attention_layers = nn.ModuleList(
            [VJEPA2PoolerSelfAttentionLayer(config) for _ in range(config.num_pooler_layers)]
        )

    def forward(self, hidden_state: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,) -> torch.Tensor:

        if attention_mask is not None:
            # Step 1: Expand dimensions
            attention_mask = attention_mask[:, None, None, :]
            # Step 2: Convert to additive form
            attention_mask = (1.0 - attention_mask) * torch.finfo(hidden_state.dtype).min * 0.0005

        for layer in self.self_attention_layers:
            hidden_state = layer(hidden_state, attention_mask=attention_mask)[0]
        queries = self.query_tokens.repeat(hidden_state.shape[0], 1, 1)
        hidden_state = self.cross_attention_layer(queries, hidden_state)[0]
        return hidden_state.squeeze(1)

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

        if self.model_name == "vjepa2":
            outputs = self.encoder(pixel_values_videos=pixel_values_videos, output_hidden_states=True)
            last_hidden_states = outputs.last_hidden_state
        elif self.model_name.startswith("videomae"):
            # VideoMAE has fixed-length learned position embeddings (unlike vjepa2's
            # RoPE), so it can only ever be called on exactly the frame count it was
            # pretrained on (16). Longer clips are covered by chopping into
            # `max_segments` 16-frame segments (see CustomVideoDataset), each encoded
            # independently, then concatenated into one long token sequence per video
            # before pooling -- mirrors the frame-based branch's per-frame encode +
            # flatten pattern, just at segment granularity instead of per-frame.
            #
            # pixel_values_videos arrives as (batch_size * max_segments, T, C, H, W),
            # segments for a given video contiguous in that order (built via
            # torch.cat in train_ddp/eval_ddp), so a single batched encoder call
            # followed by a reshape recovers "one video's tokens = its segments'
            # token sequences concatenated in order" with no explicit Python loop.
            max_segments = self.encoder_config["max_segments"]
            total_segments = pixel_values_videos.shape[0]
            batch_size = total_segments // max_segments

            outputs = self.encoder(pixel_values=pixel_values_videos)
            last_hidden_states = outputs.last_hidden_state  # (batch*max_segments, tokens_per_segment, hidden)
            hidden_size = last_hidden_states.shape[-1]
            last_hidden_states = last_hidden_states.reshape(batch_size, -1, hidden_size)

            # context_masks: (batch*max_segments, tubelets*patches) -> (batch, max_segments*tubelets*patches)
            context_masks = context_masks.reshape(batch_size, -1)
        elif self.model_name.startswith("videomamba"):
            # VideoMambaFeatureExtractor.forward returns the raw patch-token
            # tensor [B, L, C] directly (no HF-style output object, no CLS)
            last_hidden_states = self.encoder(pixel_values_videos)
        elif self.model_name.startswith('x3d'):
            # X3D is a 3D CNN whose spatial stages (and hidden_size) are tied to a
            # specific input resolution/frame count per variant (13 frames for
            # x3d_s, 16 for x3d_m/x3d_l) -- same hard per-call frame limit as
            # VideoMAE above, covered the same way: `max_segments` frames_per_clip
            # -sized segments, each encoded independently, concatenated into one
            # long token sequence per video before pooling.
            #
            # X3DFeatureExtractor.forward handles permute, blocks, and reshape
            # internally, returning [total_segments, T'*H'*W', C] for a
            # (batch_size * max_segments, T, C, H, W) input -- same contiguous
            # per-video-segment layout as the videomae branch above.
            max_segments = self.encoder_config["max_segments"]
            total_segments = pixel_values_videos.shape[0]
            batch_size = total_segments // max_segments

            x = self.encoder(pixel_values_videos)  # (batch*max_segments, T'*H'*W', C)
            total, N, C = x.shape

            # Recover T' for mask expansion (mask is built per-frame in
            # CustomVideoDataset, shape (batch*max_segments, frames_per_clip); X3D's
            # tubelet_size=1 means T' == frames_per_clip, so N == T' * spatial**2).
            spatial = self.encoder_config["feature_map_size"]

            context_masks = context_masks.unsqueeze(-1)
            context_masks = context_masks.expand(-1, -1, spatial * spatial)
            context_masks = context_masks.reshape(total, N)  # (batch*max_segments, N)

            x = x.reshape(batch_size, max_segments * N, C)
            context_masks = context_masks.reshape(batch_size, max_segments * N)

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
                # Temporal mask: 1 for real frames, 0 for padding
                n_real = n_real_original
                n_padded = self.frames_per_clip - n_real
                # Padding is prepended 
                # positions [0 : n_padded] are padding, [n_padded : T] are real
                mask = torch.zeros(self.frames_per_clip)
                mask[n_padded:] = 1.0
                masks.append(mask)

            else:
                # Create mask for this segment
                mask_idx = torch.arange(
                    math.ceil((self.frames_per_clip - video_frames.shape[0])/ self.tubelet_size) * patches,
                    tubelets * patches
                )
                mask = torch.zeros((tubelets * patches,))
                mask[mask_idx] = 1
                masks.append(mask)

            # Pad if necessary
            if video_frames.shape[0] < self.frames_per_clip:
                missing_frames = self.frames_per_clip - video_frames.shape[0]
                padding = self.fill_value * torch.ones(
                    (missing_frames, *video_frames.shape[1:]),
                    dtype=video_frames.dtype
                )
                video_frames = torch.cat([padding, video_frames], dim=0)

            clips.append(video_frames)

        # Pad the SEGMENT COUNT itself up to max_segments (not just within-segment
        # frames, handled above) -- needed when a video is short enough that
        # generate_segments() returns fewer than max_segments real segments (e.g.
        # videomae's 16-frame segments covering a long clip). Prepended, so real
        # content stays at the end of the sequence, same "padding precedes real
        # content" convention used for within-segment frame padding. No-op for every
        # other model, since they all use max_segments=1 and generate_segments()
        # always returns at least 1 segment.
        n_pad_segments = self.max_segments - len(clips)
        if n_pad_segments > 0:
            fake_clip = self.fill_value * torch.ones_like(clips[0])
            fake_mask = torch.zeros_like(masks[0])
            clips = [fake_clip] * n_pad_segments + clips
            masks = [fake_mask] * n_pad_segments + masks

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


def generate_segments(video_frames, frames_per_clip, sampling=1, frame_jitter=0):
    segments = []
    start = video_frames
    while start > 0:
        end = max(start - frames_per_clip * sampling, 0)
        segments.append(np.sort(np.arange(start, end, -sampling)) - 1)
        start = end

        if frame_jitter > 0:
            jitter_frames = np.random.randint(-frame_jitter, frame_jitter+1, len(segments[-1]) )
            segments[-1] += jitter_frames
            segments[-1] = np.clip(segments[-1], 0, video_frames-1) # ensure that it fall inside the video for the last and first frame.

    return segments


def collate_fn_all_segments(samples, transforms=None, frame_drop_ratio_2=0.0):
    """
    Simplified collate function - decoding is already done in workers
    """
    clips_list, labels, ids, masks = [], [], [], []

    for clips, lbl, vid_id, mask in samples:
        if frame_drop_ratio_2 > 0.0:
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


def setup(rank, world_size):
    """Initialize the distributed environment."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    torch.cuda.set_device(rank)  # CRITICAL: Set device before init_process_group
    # Initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    """Clean up the distributed environment."""
    dist.destroy_process_group()

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
            print(step)
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


def train_ddp(rank, world_size, model_name, result_dir, experiment_id, frames, 
              dataset_root_path, debug=False, num_epochs=20, batch_size=2, 
              base_lr=1e-3, weight_decay=0.8,
              continue_training=False, checkpoint_path=None,
              fps=4, 
              no_augmentations=False,
              tube_mask_ratio=0.0, frame_drop_ratio=0.0, speed_jitter=False, static_vids_T=False, upsample_rare_categories=True,
              frame_drop_ratio_2=0.0):

    """Main training function for each process."""

    # Setup distributed training
    setup(rank, world_size)

    # Set device for this process
    device = torch.device(f"cuda:{rank}")
    #torch.cuda.set_device(device)
    categories = ['Cup', 'Knife', 'Chair', 'Person', 'Car', 'Bike', 'Dog', 'Cat', 'Table', 'Book', 'Plant', 'Bed']
    label2id = {label: i for i, label in enumerate(categories)}

    # Load data
    train_df = pd.read_csv(dataset_root_path / "train_humanReports_rebalanced_v0.csv")
    val_df = pd.read_csv(dataset_root_path / "val_humanReports_rebalanced_v0.csv")
    test_df = pd.read_csv(dataset_root_path / "test_humanReports_rebalanced_v0.csv")

    if upsample_rare_categories:
        print('Upsampling rare categories')
        # Oversample for the rare categories
        rare_categories = ['Cat', 'Dog', 'Bike']  # all with a mean report rate in the training set of less than 0.02
        # Estimate the probability of each category report and this using the inverse

        additional_trials_per_category = 1000
        additional_trials = []
        for category in rare_categories:
            kernel = gaussian_kde(
                train_df[category].values)
            train_df[f'{category}_pdf'] = 1 / kernel(train_df[category].values)

            additional_trials.append(
                train_df.sample(additional_trials_per_category, weights=f'{category}_pdf', replace=True, random_state=34))

        additional_trials = pd.concat(additional_trials, ignore_index=True)

        train_df = pd.concat([train_df, additional_trials], ignore_index=True)

    # VideoMAE and X3D both have a hard per-call frame limit (VideoMAE: fixed-length
    # learned position embeddings, unlike vjepa2's RoPE; X3D: a 3D CNN whose spatial
    # stages/hidden_size are tied to a specific input size per variant -- 13 frames
    # for x3d_s, 16 for x3d_m/x3d_l). Both cover longer clips via multiple
    # frames-per-clip-sized segments concatenated before pooling (see
    # CustomVideoDataset / VideoEncoder_ForHumanSensoryHistoryReports.forward)
    # instead of one big frames_per_clip like vjepa2. max_segments_per_video =
    # ceil(60 / frames) covers the same ~15s duration cap the other full-clip
    # models use (60 frames @ 4fps, e.g. vjepa2's frames_per_clip=60), regardless
    # of this model's own per-segment frame budget -- instead of restricting
    # training to clips no longer than one segment (e.g. <=4s for a 16-frame
    # segment at 4fps).
    if model_name.startswith("videomae") or model_name.startswith("x3d"):
        max_segments_per_video = math.ceil(60 / frames)
    else:
        max_segments_per_video = 1

    # Only include videos that fit within the total frame budget across all
    # segments (frames_per_clip * max_segments_per_video), to avoid an
    # overreliance on context beyond what the model actually gets to see.
    duration_cap_frames = frames * max_segments_per_video
    train_df = train_df.loc[train_df['videoDuration (sec)'] <= np.max([duration_cap_frames/fps, 0.5])]
    val_df = val_df.loc[val_df['videoDuration (sec)'] <= np.max([duration_cap_frames/fps, 0.5])]

    # if debug:
    #     train_df = train_df.sample(n=128, replace=False, random_state=42)
    #     test_df = test_df.sample(n=64, replace=False, random_state=42)

    if rank == 0:
        video_count_train = len(train_df)

        video_total = video_count_train #+  video_count_test #+ video_count_vis + video_count_bench
        print(f"Total videos: {video_total}")


    # Prepare datasets
    train_video_file_paths = dataset_root_path / (train_df['stimulus_video_url'].str.split('/',expand=True)[4]) / (train_df['stimulus_video_url'].str.split('/',expand=True)[5])
    val_video_file_paths = dataset_root_path / (val_df['stimulus_video_url'].str.split('/',expand=True)[4]) /(val_df['stimulus_video_url'].str.split('/',expand=True)[5])
    test_video_file_paths = dataset_root_path / (test_df['stimulus_video_url'].str.split('/',expand=True)[4]) / (test_df['stimulus_video_url'].str.split('/',expand=True)[5])

    train_labels = train_df[categories].values
    val_labels = val_df[categories].values
    test_labels = test_df[categories].values

    # Create datasets with decoding in workers
    # Create data loaders with distributed samplers
    num_workers = 4#4  # 4#2  # Reduced for stability
    prefetch_factor = 2  # 4qq
    sampling = 30 // fps  # videos were encoded at 30fps, we're sampling at 5Hz, thus every 200ms
    frames_per_clip = frames  # 40#2#config.frames_per_clip  # this is 64, thus 12.8s (64 x 0.2s) maximum video duration.
    frame_jitter = 3 # +/-100ms since videas are encoded at 30fps

    model, processor, encoder_config = get_encoder(model_name)
    tubelet_size = encoder_config["tubelet_size"]
    added_tokens = encoder_config["added_tokens"]
    patch_size = encoder_config["patch_size"]
    # Read by VideoEncoder_ForHumanSensoryHistoryReports.forward's videomae branch
    # to reshape the flat (batch*max_segments, ...) tensors back per-video.
    encoder_config["max_segments"] = max_segments_per_video

    train_ds = CustomVideoDataset(
        train_video_file_paths.tolist(),
        train_labels,
        frames_per_clip=frames_per_clip,
        sampling=sampling,
        frame_jitter=frame_jitter,
        max_segments=max_segments_per_video,
        tubelet_size=tubelet_size,
        patch_size=patch_size,
        added_tokens=added_tokens,
        decoder_seek_mode='approximate',
        #device='cuda' if torch.cuda.is_available() else 'cpu',
        num_ffmpeg_threads=num_workers,
        encoder_config=encoder_config,
    )

    val_ds = CustomVideoDataset(
        val_video_file_paths.tolist(),
        val_labels,
        frames_per_clip=frames_per_clip,
        sampling=sampling,
        frame_jitter=0,  # No jitter for validation
        max_segments=max_segments_per_video,
        tubelet_size=tubelet_size,
        patch_size=patch_size,
        added_tokens=added_tokens,
        decoder_seek_mode='approximate',
        #device='cuda' if torch.cuda.is_available() else 'cpu',
        num_ffmpeg_threads=num_workers,
        encoder_config=encoder_config,
    )

    test_ds = CustomVideoDataset(
        test_video_file_paths.tolist(),
        test_labels,
        frames_per_clip=frames_per_clip,
        sampling=sampling,
        frame_jitter=0,  # No jitter for test
        max_segments=max_segments_per_video,
        tubelet_size=tubelet_size,
        patch_size=patch_size,
        added_tokens=added_tokens,
        #device='cuda' if torch.cuda.is_available() else 'cpu',
        num_ffmpeg_threads=num_workers,
        encoder_config=encoder_config,
    )

    config = AutoConfig.from_pretrained("facebook/vjepa2-vitl-fpc64-256") # vJEPA2 config
    config.hidden_size = encoder_config["hidden_size"]
    config.num_labels = 12
    model_SH = VideoEncoder_ForHumanSensoryHistoryReports(model, config, encoder_config).to(device)

    # Freeze backbone if needed
    for param in model_SH.encoder.parameters():
        param.requires_grad = False

    if continue_training:
        if not os.path.exists(checkpoint_path):
            if rank == 0:
                print(f"Checkpoint not found at {checkpoint_path}")
            cleanup()
            return

        if rank == 0:
            print(f"Loading checkpoint from {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=device)
        model_SH.load_state_dict(checkpoint['model_state_dict'])

    model_SH = DDP(
        model_SH,
        device_ids=[rank],
        output_device=rank,
        find_unused_parameters=False,  # More efficient
        broadcast_buffers=False,  # Don't sync buffers if not needed
        gradient_as_bucket_view=True  # Memory optimization
    )

    # Setup transforms
    if model_name == "vjepa2":
        size = 256
    elif model_name.startswith("videomae"):
        size = 224
    elif model_name.startswith('x3d'):
        size = encoder_config["image_size"]
    elif model_name.startswith('videomamba'):
        size = encoder_config["image_size"]
    else:
        raise ValueError(f"Unknown model_name for size: {model_name}")

    if model_name.startswith('x3d'):
        mean = [0.45, 0.45, 0.45]
        std  = [0.225, 0.225, 0.225]
        resample = 2  # bilinear
    elif model_name.startswith('videomamba'):
        # No HF processor for VideoMamba -- standard ImageNet/DeiT normalization
        # stats and bicubic resizing, per the paper's training recipe.
        mean = [0.485, 0.456, 0.406]
        std  = [0.229, 0.224, 0.225]
        resample = 3  # bicubic
    else:
        mean = processor.image_mean
        std  = processor.image_std
        resample = processor.resample

    train_transforms = v2.Compose([
        v2.AutoAugment(),
        v2.RandomResizedCrop((size, size), scale=(0.9, 1.0), interpolation=resample),
        v2.RandomHorizontalFlip(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=mean, std=std)
    ])
    eval_transforms = v2.Compose([
        v2.Resize((size, size), interpolation=resample),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=mean, std=std)
    ])

    # Create distributed samplers
    train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler = DistributedSampler(val_ds, num_replicas=world_size, rank=rank, shuffle=False)
    test_sampler = DistributedSampler(test_ds, num_replicas=world_size, rank=rank, shuffle=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=train_sampler,
        collate_fn=partial(collate_fn_all_segments, transforms=train_transforms, frame_drop_ratio_2=frame_drop_ratio_2),
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=prefetch_factor,
        persistent_workers=True if num_workers > 0 else False,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        sampler=val_sampler,
        collate_fn=partial(collate_fn_all_segments, transforms=eval_transforms),
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=prefetch_factor,
        persistent_workers=True if num_workers > 0 else False,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        sampler=test_sampler,
        collate_fn=partial(collate_fn_all_segments, transforms=eval_transforms),
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=prefetch_factor,
        persistent_workers=True if num_workers > 0 else False,
    )


    # Setup tensorboard writer (only on main process)
    writer = None
    if (rank == 0) & (debug == False):
        print('Writing to tensorboard...')
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(f"runs/{experiment_id}")

    # Optimizer
    trainable = [p for p in model_SH.parameters() if p.requires_grad]

    if writer:
        writer.add_scalar(f"Base LR", base_lr)
        writer.add_scalar(f"weight_decay", weight_decay)

    optimizer = torch.optim.AdamW(trainable, lr=base_lr, weight_decay=weight_decay)
    if continue_training:
        print('Loading optimizer state...')
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # Training loop
    accumulation_steps = 1 #if batch_size < 16 else 2  # Reduced since we're using 8 GPUs

    grad_clip_value = 1.0  # Gradient clipping

    loss_func =nn.MSELoss()

    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs * len(train_loader) // accumulation_steps)
    if continue_training:
        if (checkpoint['scheduler_state_dict'] is not None):
            print('Loading scheduler state...')
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        start_epoch = checkpoint['epoch'] + 1
    else:
        start_epoch=1


    for epoch in range(start_epoch, num_epochs + 1):
        if rank == 0:
            print(f"Epoch {epoch}")

        # Set epoch for distributed sampler
        train_sampler.set_epoch(epoch)

        model_SH.train()

        running_mse = 0.0
        optimizer.zero_grad()
        t0 = time.time()

        for step, (vids, labels, ids, masks) in enumerate(train_loader, start=1):
            if step % 1 == 0 and rank == 0:
                print(f"{rank}: Step {step}", flush=True)
            t1 = time.time()
            labels = labels.to(device)
            inputs = torch.cat(vids, dim=0).to(device)#[]#[processor(v.squeeze(), return_tensors="pt").to(device) for v in vids]
            masks = torch.cat(masks, dim=0).to(device)
            t2 = time.time()

            # Forward pass
            outputs = model_SH(pixel_values_videos=inputs, context_masks=masks)

            loss = loss_func(labels, outputs.logits)/ accumulation_steps

            loss.backward()
            t3 = time.time()
            running_mse += loss.item()
            # if step % 1 == 0 and rank == 0:
            #     print(f"data_load: {t1 - t0:.3f}s, preprocess: {t2 - t1:.3f}s, fwd+bwd: {t3 - t2:.3f}s", flush=True)

            if step % accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model_SH.parameters(), grad_clip_value)

                optimizer.step()
                optimizer.zero_grad()
                if scheduler is not None:
                    scheduler.step()

                lr = scheduler.get_last_lr()[0]
                t4 = time.time()

                print(
                    f"Epoch {epoch} Step {step} (Rank {rank}):  MSE = {running_mse:.4f}, LR = {lr:.2e}, Duration: {t4-t0}s", flush=True)

                if writer:
                    if rank == 0:
                        writer.add_scalar(f"Train MSE (rank{rank})", running_mse,
                                      (epoch - 1) * len(train_loader) + step)

                        writer.add_scalar("Learning rate", lr,
                                          (epoch - 1) * len(train_loader) + step)

                running_mse = 0.0
            #t0 = time.time()

        with torch.no_grad():
            model_SH.eval()
            t0 = time.time()
            if rank == 0:
                print(f"Epoch {epoch}: Computing validation loss", flush=True)

            running_mse_val = 0.0
            total_samples = 0
            for step, (vids, labels, ids, masks) in enumerate(val_loader, start=1):

                # print(f"{rank}: Step {step}")
                labels = labels.to(device)
                inputs = torch.cat(vids, dim=0).to(device)#inputs = inputs.to(device)#[processor(v.squeeze(), return_tensors="pt").to(device) for v in vids]
                masks = torch.cat(masks, dim=0).to(device)
                # Forward pass
                outputs = model_SH(pixel_values_videos=inputs, context_masks=masks)

                mse = loss_func(labels, outputs.logits)
                running_mse_val += mse.item() * labels.size(0)
                total_samples += labels.size(0)

        model_SH.train()
        running_mse_val = torch.tensor(running_mse_val).to(device)
        total_samples = torch.tensor(total_samples).to(device)

        dist.all_reduce(running_mse_val, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_samples, op=dist.ReduceOp.SUM)

        val_mse = running_mse_val / total_samples

        torch.cuda.empty_cache()
        
        if rank == 0:
            t5 = time.time()
            print(f"Epoch {epoch} Validation MSE: {val_mse:.4f} Duration: {t5-t0}s")

            if writer:
                writer.add_scalar("Val MSE", val_mse, (epoch - 1) * len(train_loader) + step)

            if not os.path.exists(f"{result_dir}/{experiment_id}"):
                os.makedirs(f"{result_dir}/{experiment_id}")

            # Save the model (only from main process)
            # Save the full model with trained pooler and classifier
            torch.save({
                'model_state_dict': model_SH.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'epoch': epoch,
                'val_mse': val_mse,
            }, f"{result_dir}/{experiment_id}/checkpoint_final.pt")

            with open(f"{result_dir}/{experiment_id}/status.json", "w") as f:
                json.dump({"epoch": epoch, "val_mse": float(val_mse)}, f)

    # Final test evaluation
    if rank == 0:
        print(f"Training completed: Computing test predictions", flush=True)

    test_predictions, test_ids, test_labels = get_predictions(test_loader, model_SH, processor, device, rank)

    if rank == 0:
        test_mse = loss_func(test_labels, test_predictions)
        print(f"Test MSE: {test_mse:.4f}")
        if writer:
            writer.add_scalar("Test MSE", test_mse, num_epochs * len(train_loader))

        test_predictions = test_predictions.cpu().numpy()
        test_labels = test_labels.cpu().numpy()
        results = []

        for i, cat in enumerate(categories):
            results.append(pd.DataFrame({'videoID': test_ids, 'category': cat,
                    'prediction': test_predictions[:, i], 'video_type': 'test',
                    'labels': test_labels[:, i]}))

        results = pd.concat(results)
        results.to_csv(
            f'{result_dir}/{experiment_id}_preds_{frames_per_clip}Frames_AttentionalPooling.csv'
        )


    if rank == 0:
        if not os.path.exists(f"{result_dir}/{experiment_id}"):
            os.makedirs(f"{result_dir}/{experiment_id}")

        # Save the model (only from main process)
        # Save the full model with trained pooler and classifier
        if start_epoch <= num_epochs:
            torch.save({
                'model_state_dict': model_SH.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'epoch': num_epochs,
                'val_mse': val_mse,
            }, f"{result_dir}/{experiment_id}/checkpoint_final.pt")

            with open(f"{result_dir}/{experiment_id}/status.json", "w") as f:
                json.dump({"epoch": epoch, "val_mse": float(val_mse)}, f)

        if processor is not None:
            processor.save_pretrained(f"{result_dir}/{experiment_id}")

        # Optionally save just the trainable parts
        torch.save({
            'pooler': model_SH.module.pooler.state_dict(),
            'classifier': model_SH.module.classifier.state_dict(),
        }, f"{result_dir}/{experiment_id}/trained_head.pt")



    if dist.is_initialized():
        dist.barrier()


    # Clean up process group to stop NCCL watchdog
    cleanup()


    torch.cuda.empty_cache()
    # if dist.is_initialized():
    #     dist.barrier()

    #cleanup()


def eval_ddp(rank, world_size, model_name, result_dir, experiment_id, frames, dataset_root_path, batch_size=2,fps=4):
    """Evaluation function to compute activations for benchmark and visualization datasets."""

    # Setup distributed training
    setup(rank, world_size)

    # Set device for this process
    device = torch.device(f"cuda:{rank}")
    #torch.cuda.set_device(device)

    categories = ['Cup', 'Knife', 'Chair', 'Person', 'Car', 'Bike', 'Dog', 'Cat', 'Table', 'Book', 'Plant', 'Bed']

    # Load data
    vis_df = pd.read_csv(dataset_root_path / "visualization_humanReports_rebalanced.csv")
    bench_df = pd.read_csv(dataset_root_path / "benchmarks_humanReports_rebalanced.csv")
    seq_df = pd.read_csv(dataset_root_path / "clipSequences_humanReports_rebalanced.csv")

    bench_df.loc[bench_df['Benchmark_subcondition'].isna(), 'Benchmark_subcondition'] = ''

    # Run videos once with and without noise.
    # bench_df.loc[bench_df['Benchmark_condition'] == 'SensoryHistory', 'Benchmark_condition'] = 'SensoryHistory_withoutNoise'
    bench_SH_noiseless = bench_df.loc[bench_df['Benchmark_condition'] == 'SensoryHistory'].copy()
    bench_SH_noiseless['Benchmark_condition'] = 'SensoryHistory_withoutNoise'

    bench_df = pd.concat([bench_df, bench_SH_noiseless], ignore_index=True)

    if rank == 0:
        video_count_vis = len(vis_df)
        video_count_bench = len(bench_df)
        video_count_seq = len(seq_df)
        print(f"Visualization videos: {video_count_vis}")
        print(f"Benchmark videos: {video_count_bench}")
        print(f"ClipSeq videos: {video_count_seq}")

    # Prepare datasets
    vis_video_file_paths = dataset_root_path / 'Benchmarks' / 'Visualization' / (
                vis_df['videoDuration (sec)'].astype(str) + 's') / (vis_df['File_name'] + '.mp4')

    bench_video_file_paths = dataset_root_path / 'Benchmarks' / bench_df['Benchmark_condition'] / bench_df[
        'Benchmark_subcondition'] / (bench_df['File_name'] + '.mp4')

    seq_video_file_paths = dataset_root_path / 'clipSequences' / seq_df['Seq_condition']  / (seq_df['File_name'] + '.mp4')

    vis_labels = vis_df[categories].values
    bench_labels = bench_df[categories].values
    seq_labels = seq_df[categories].values

    num_workers = 4  # 4  # 4#2  # Reduced for stability
    # See train_ddp for why videomae/x3d need multiple frames_per_clip-sized segments per video.
    if model_name.startswith("videomae") or model_name.startswith("x3d"):
        max_segments_per_video = math.ceil(60 / frames)
    else:
        max_segments_per_video = 1
    prefetch_factor = 2  # 4qq
    sampling = 30 // fps  # videos were encoded at 30fps, we're sampling at 5Hz, thus every 200ms
    frames_per_clip = frames  # 40#2#config.frames_per_clip  # this is 64, thus 12.8s (64 x 0.2s) maximum video duration.
    frame_jitter = 3

    model, processor, encoder_config = get_encoder(model_name)
    tubelet_size = encoder_config["tubelet_size"]
    added_tokens = encoder_config["added_tokens"]
    patch_size = encoder_config["patch_size"]
    encoder_config["max_segments"] = max_segments_per_video

    vis_ds = CustomVideoDataset(
        vis_video_file_paths.tolist(),
        vis_labels,
        frames_per_clip=frames_per_clip,
        sampling=sampling,
        frame_jitter=0,  # No jitter for test
        max_segments=max_segments_per_video,
        tubelet_size=tubelet_size,
        patch_size=patch_size,
        added_tokens=added_tokens,
        # device='cuda' if torch.cuda.is_available() else 'cpu',
        num_ffmpeg_threads=num_workers,
        addFolderToID=-2,
        encoder_config=encoder_config,
    )

    seq_ds = CustomVideoDataset(
        seq_video_file_paths.tolist(),
        seq_labels,
        frames_per_clip=frames_per_clip,
        sampling= 5//fps, # these videos were encoded at 5fps, the model runs 4fps -> yield 1
        frame_jitter=0,  # No jitter for test
        max_segments=max_segments_per_video,
        tubelet_size=tubelet_size,
        patch_size=patch_size,
        added_tokens=added_tokens,
        # device='cuda' if torch.cuda.is_available() else 'cpu',
        num_ffmpeg_threads=num_workers,
        addFolderToID=-2,
        encoder_config=encoder_config,
    )
    bench_ds = CustomVideoDataset(
        bench_video_file_paths.tolist(),
        bench_labels,
        frames_per_clip=frames_per_clip,
        sampling=sampling,
        frame_jitter=0,  # No jitter for test
        max_segments=max_segments_per_video,
        tubelet_size=tubelet_size,
        patch_size=patch_size,
        added_tokens=added_tokens,
        # device='cuda' if torch.cuda.is_available() else 'cpu',
        num_ffmpeg_threads=num_workers,
        addFolderToID=-3,
        encoder_config=encoder_config,
    )

    # Load trained model configuration
    config = AutoConfig.from_pretrained("facebook/vjepa2-vitl-fpc64-256")
    config.num_labels = 12
    model_SH = VideoEncoder_ForHumanSensoryHistoryReports(model, config, encoder_config).to(device)

    # Load checkpoint
    checkpoint_path = f"{result_dir}/{experiment_id}/checkpoint_final.pt"
    if not os.path.exists(checkpoint_path):
        if rank == 0:
            print(f"Checkpoint not found at {checkpoint_path}")
        cleanup()
        return

    if rank == 0:
        print(f"Loading checkpoint from {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_SH.load_state_dict(checkpoint['model_state_dict'])

    # Wrap with DDP
    model_SH = DDP(
        model_SH,
        device_ids=[rank],
        output_device=rank,
        find_unused_parameters=False,
        broadcast_buffers=False,
        gradient_as_bucket_view=True
    )

    # Create distributed samplers
    vis_sampler = DistributedSampler(vis_ds, num_replicas=world_size, rank=rank, shuffle=False)
    bench_sampler = DistributedSampler(bench_ds, num_replicas=world_size, rank=rank, shuffle=False)
    seq_sampler = DistributedSampler(seq_ds, num_replicas=world_size, rank=rank, shuffle=False)

    if model_name == "vjepa2":
        size = 256
    elif model_name.startswith("videomae"):
        size = 224
    elif model_name.startswith('x3d'):
        size = encoder_config["image_size"]
    elif model_name.startswith('videomamba'):
        size = encoder_config["image_size"]
    else:
        raise ValueError(f"Unknown model_name for size: {model_name}")

    # x3d and videomamba have no HF processor (processor is None), so
    # processor.resample/.image_mean below would crash for them.
    if model_name.startswith('x3d'):
        mean = [0.45, 0.45, 0.45]
        std  = [0.225, 0.225, 0.225]
        resample = 2  # bilinear
    elif model_name.startswith('videomamba'):
        mean = [0.485, 0.456, 0.406]
        std  = [0.229, 0.224, 0.225]
        resample = 3  # bicubic
    else:
        mean = processor.image_mean
        std  = processor.image_std
        resample = processor.resample

    eval_transforms = v2.Compose([
        v2.Resize((size, size), interpolation=resample),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=mean, std=std)
    ])


    vis_loader = DataLoader(
        vis_ds,
        batch_size=batch_size,
        sampler=vis_sampler,
        collate_fn=partial(collate_fn_all_segments, transforms=eval_transforms),
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=prefetch_factor,
        persistent_workers=True if num_workers > 0 else False,
    )

    bench_loader = DataLoader(
        bench_ds,
        batch_size=batch_size,
        sampler=bench_sampler,
        collate_fn=partial(collate_fn_all_segments, transforms=eval_transforms),
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=prefetch_factor,
        persistent_workers=True if num_workers > 0 else False,
    )

    seq_loader = DataLoader(
        seq_ds,
        batch_size=batch_size,
        sampler=seq_sampler,
        collate_fn=partial(collate_fn_all_segments, transforms=eval_transforms),
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=prefetch_factor,
        persistent_workers=True if num_workers > 0 else False,
    )

    # Compute activations
    if rank == 0:
        print('Computing visualization set activations...')
    vis_activations, vis_ids, vis_labels_tensor = get_predictions(
        vis_loader, model_SH, processor, device, rank,
        #interpolate_pos_encoding=interpolate_pos_encoding
    )

    if rank == 0:
        print('Computing benchmark set activations...')
    bench_activations, bench_ids, bench_labels_tensor = get_predictions(
        bench_loader, model_SH, processor, device, rank,
        #interpolate_pos_encoding=interpolate_pos_encoding
    )

    if rank == 0:
        print('Computing clipSequence set activations...')
    seq_activations, seq_ids, seq_labels_tensor = get_predictions(
        seq_loader, model_SH, processor, device, rank,
        #interpolate_pos_encoding=interpolate_pos_encoding
    )

    # Save results (only from main process)
    if rank == 0:
        vis_predictions = vis_activations.cpu().numpy()
        vis_labels_np = vis_labels_tensor.cpu().numpy()

        bench_predictions = bench_activations.cpu().numpy()
        bench_labels_np = bench_labels_tensor.cpu().numpy()

        seq_predictions = seq_activations.cpu().numpy()
        seq_labels_np = seq_labels_tensor.cpu().numpy()

        results_vis = []
        results_bench = []
        results_seq = []

        for i, cat in enumerate(categories):
            results_vis.append(pd.DataFrame({
                'videoID': vis_ids,
                'category': cat,
                'prediction': vis_predictions[:, i],
                'video_type': 'visualization',
                'labels': vis_labels_np[:, i]
            }))

            results_bench.append(pd.DataFrame({
                'videoID': bench_ids,
                'category': cat,
                'prediction': bench_predictions[:, i],
                'video_type': 'benchmark',
                'labels': bench_labels_np[:, i]
            }))

            results_seq.append(pd.DataFrame({
                'videoID': seq_ids,
                'category': cat,
                'prediction': seq_predictions[:, i],
                'video_type': 'clipSequences',
                'labels': seq_labels_np[:, i]
            }))

        results_vis = pd.concat(results_vis, ignore_index=True)
        results_bench = pd.concat(results_bench, ignore_index=True)
        results_seq = pd.concat(results_seq, ignore_index=True)


        results_vis.to_csv(
            f'{result_dir}/{experiment_id}_preds_{frames_per_clip}Frames_AttentionalPooling_visualization.csv',
            index=False
        )
        results_bench.to_csv(
            f'{result_dir}/{experiment_id}_preds_{frames_per_clip}Frames_AttentionalPooling_benchmark.csv',
            index=False
        )

        results_seq.to_csv(
            f'{result_dir}/{experiment_id}_preds_{frames_per_clip}Frames_AttentionalPooling_clipSequences.csv',
            index=False
        )


    if dist.is_initialized():
        dist.barrier()

    cleanup()
    torch.cuda.empty_cache()

def main():
    """Main entry point for multi-GPU training."""
    parser = argparse.ArgumentParser()  
    parser.add_argument("--base_dir", type=str, required=True,
                            help="Mandatory path to the base directory containing data"
                            # /braintree/data2/active/users/aicha/Ego4D_data
                            # /home/aicha/orcd/pool/Ego4D_data
                            )
    parser.add_argument("--approach", type=str, default="baseline",
                        choices=["baseline", "tubing", "frame_dropping", "frame_dropping_2", "speed", "remove_static", "upsample_rare_categories", "vanilla_model"],
                        help="Data augmentation approach")
    parser.add_argument("--encoder", type=str, default="vjepa2",
                        choices=["vjepa2", "x3d_s", "x3d_m", "x3d_l", "videomae_b", "videomae_l", "videomamba_m"],
                        help="Frozen Encoder Backbone")
    parser.add_argument("--batch_size", type=int, default=10)
    args = parser.parse_args()

    print("SLURM_CPUS_PER_TASK =", os.environ.get("SLURM_CPUS_PER_TASK"))
    print("torch.cuda.device_count() =", torch.cuda.device_count())
    print("CUDA_VISIBLE_DEVICES =", os.environ.get("CUDA_VISIBLE_DEVICES"))
    debug = False
    base_dir = args.base_dir 
    if debug:
        world_size = 1
    else:
        world_size = torch.cuda.device_count()
    print(f"Using {world_size} GPUs")

    modes = ['train']#, 'eval']
    model_name = args.encoder

    if base_dir == None:
        base_dir = f'{Path(__file__).parent}'

    if base_dir.startswith('/orcd'):
        dataset_root_path = pathlib.Path("/home/lynnka/orcd/pool/Ego4D_videos/")
    elif base_dir.startswith('/braintree'):
        dataset_root_path = pathlib.Path("/braintree/data2/active/users/aicha/Ego4D_data")
    elif base_dir.startswith('/home'):
        dataset_root_path = pathlib.Path("/home/aicha/orcd/pool/Ego4D_data")
    else:
        raise ValueError('dataset root directory not defined')

    fps=4
    num_epochs=15
    batch_size=args.batch_size
    approach = args.approach # all approaches that we try to optimize the model (data augmentation...)
    # Set augmentation ratios based on approach (0.0 = disabled for baseline)
    tube_mask_ratio = 0.00001 if approach == "tubing" else 0.0
    frame_drop_ratio = 0.00001 if approach == "frame_dropping"  else 0.0
    frame_drop_ratio_2 = 0.5 if approach == "frame_dropping_2"  else 0.0
    speed_jitter = True if approach == "speed" else False
    static_vids_T = 0.8 if approach == "remove_static" else 0.0 # if we want to remove static videos, we remove STATIC videos (1 cluster or less) that are shorter than static_vids_T (sec)
    upsample_rare_categories = True if approach == "upsample_rare_categories" else False
    no_augmentations = True if approach == "vanilla_model" else False

    if model_name.startswith("videomae"):
        frames = [16]
    elif model_name == 'vjepa2':
        frames = [60]
    elif model_name.startswith('x3d'):
        frames = [13] if model_name == 'x3d_s' else [16]  # match model's native T
    elif model_name.startswith('videomamba'):
        frames = [16]  # must match num_frames hardcoded in get_encoder() and
                        # the videomamba_m16_k400_mask_ft_f16_res224.pth checkpoint

    if base_dir.startswith('/braintree'):
        result_dir = f'/braintree/home/aicha/tests/{model_name}_{approach}_og_dataset'
    else:
        result_dir = f'/orcd/data/dicarlo/001/om/lynnka/{model_name}_{approach}_og_dataset'
    if not os.path.exists(result_dir):
        os.makedirs(result_dir, exist_ok=True)

    #experiment_number = [1]#[2, 3, 4]
    learning_rates = [1e-3]
    weight_decays = [0.01]

    for lr in learning_rates:
        for wd in weight_decays:
            for frame in frames:
                experiment_id = f'{model_name}_{frame}frames_{num_epochs}epochs_lr_{lr}_wd_{wd}_{approach}_fps_4'
                checkpoint_path = f"{result_dir}/{experiment_id}/checkpoint_final.pt"
                status_path = f"{result_dir}/{experiment_id}/status.json"
                if os.path.exists(status_path):
                    try:
                        with open(status_path) as f:
                            last_epoch = json.load(f)["epoch"]
                    except (json.JSONDecodeError, KeyError):
                        last_epoch = None  # empty/corrupt file -> treat as not-yet-trained
                    if last_epoch is not None:
                        results_path = f'{result_dir}/{model_name}-{experiment_id}_preds_{frame}Frames_AttentionalPooling.csv'
                        if last_epoch >= num_epochs and os.path.exists(results_path):
                            print(f'Skipping {experiment_id}: already fully trained and evaluated.')
                            continue
                        else:
                            print(f'Resuming {experiment_id} from epoch {last_epoch}/{num_epochs}.')
                            continue_training = True
                else:
                    continue_training = False
                for mode in modes:
                    if mode == 'train':
                        print(f'Training {experiment_id} for {num_epochs} epochs (batch size {batch_size}, fps {fps}, lr {lr}, weight decay {wd})...')
                        # Spawn processes for distributed training
                        if model_name.startswith('x3d'):
                            torch.hub.load("facebookresearch/pytorchvideo", model=model_name, pretrained=True)
                        elif model_name.startswith('videomamba'):
                            hf_hub_download(repo_id="OpenGVLab/VideoMamba", filename="videomamba_m16_k400_mask_ft_f16_res224.pth")
                        mp.spawn(
                            train_ddp,
                            args=(world_size, model_name, result_dir, experiment_id, frame, 
                                    dataset_root_path, debug, num_epochs, batch_size, 
                                    lr, wd,
                                    continue_training, checkpoint_path,
                                    fps, 
                                    no_augmentations,
                                    tube_mask_ratio, frame_drop_ratio, speed_jitter, static_vids_T, upsample_rare_categories,
                                    frame_drop_ratio_2),  # world_size and num_epochs
                            nprocs=world_size,
                            join=True
                        )
                    elif mode == 'eval':
                        if os.path.exists(result_dir + '/' + experiment_id):

                            print(f'Evaluating {experiment_id} ...')
                            # Evaluate
                            mp.spawn(
                                eval_ddp,
                                args=(world_size, model_name, result_dir, experiment_id, frame, dataset_root_path, batch_size, fps),
                                nprocs=world_size,
                                join=True
                            )

                        else:
                            print(f'{model_name}-{experiment_id} has not yet been trained.')


if __name__ == "__main__":
    main()