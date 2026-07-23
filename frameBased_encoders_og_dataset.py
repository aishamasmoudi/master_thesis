import os
import argparse

#os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
import torch
import json
from pathlib import Path
from typing import Callable, Optional, Union
from torchcodec.decoders import VideoDecoder
from transformers import AutoVideoProcessor, AutoModel, AutoConfig
from transformers import VJEPA2PreTrainedModel, VJEPA2ForVideoClassification, VJEPA2VideoProcessor, VJEPA2Config
from transformers import AutoImageProcessor, ResNetForImageClassification, ViTForImageClassification, IJepaForImageClassification, AutoModelForImageClassification, ConvNextForImageClassification, ConvNextV2ForImageClassification
from transformers import ConvNextModel
import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download
import tarfile
import pathlib
from torch.utils.data import Dataset, DataLoader
from torchcodec.samplers import clips_at_random_indices, clips_at_regular_indices
from torchvision.transforms import v2
from functools import partial
from transformers import AutoProcessor
from transformers.modeling_outputs import ImageClassifierOutput
from torch import nn

# Import distributed training modules
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from sklearn.random_projection import SparseRandomProjection
from scipy.stats import gaussian_kde

from transformers.models.vjepa2.modeling_vjepa2 import VJEPA2AttentivePooler, VJEPA2PoolerCrossAttentionLayer, VJEPA2PoolerSelfAttentionLayer
from torch.optim.lr_scheduler import CosineAnnealingLR
#from analysis.utils import plot_video_frames
import torch.multiprocessing as mp
import time
import datetime
import random

from data_augmentation import apply_tube_masks, apply_frame_drop, apply_frame_drop_2

resnet50_adv_path = "/home/aicha/orcd/pool/Ego4D_data/resnet50_adv"
#resnet50_adv_path = "/braintree/data2/active/users/aicha/Ego4D_data/resnet50_adv"

model_registry = {
    'resnet-50': "microsoft/resnet-50",
    'resnet-50_adv_eps0':    f'{resnet50_adv_path}/resnet50_l2_eps0.ckpt',
    'resnet-50_adv_eps0.01': f'{resnet50_adv_path}/resnet50_l2_eps0.01.ckpt',
    'resnet-50_adv_eps0.03': f'{resnet50_adv_path}/resnet50_l2_eps0.03.ckpt',
    'resnet-50_adv_eps0.05': f'{resnet50_adv_path}/resnet50_l2_eps0.05.ckpt',
    'resnet-50_adv_eps0.1':  f'{resnet50_adv_path}/resnet50_l2_eps0.1.ckpt',
    'resnet-50_adv_eps0.25': f'{resnet50_adv_path}/resnet50_l2_eps0.25.ckpt',
    'resnet-50_adv_eps0.5':  f'{resnet50_adv_path}/resnet50_l2_eps0.5.ckpt',
    'resne1t-50_adv_eps1':    f'{resnet50_adv_path}/resnet50_l2_eps1.ckpt',
    'resnet-50_adv_eps3':    f'{resnet50_adv_path}/resnet50_l2_eps3.ckpt',
    'resnet-50_adv_eps5':    f'{resnet50_adv_path}/resnet50_l2_eps5.ckpt',
    'ViT': 'google/vit-base-patch16-224',
    'ViT_large': 'google/vit-large-patch16-224',
    'iJEPA_14_1k': 'facebook/ijepa_vith14_1k',
    'iJEPA_14_22k': 'facebook/ijepa_vith14_22k',
    'iJEPA_16_1k': 'facebook/ijepa_vith16_1k',
    'iJEPA_16_22k': 'facebook/ijepa_vitg16_22k', 
    'dino_v2_small': 'facebook/dinov2-small-imagenet1k-1-layer',
    'dino_v2_base': 'facebook/dinov2-base-imagenet1k-1-layer',
    'dino_v2': 'facebook/dinov2-large-imagenet1k-1-layer',
    'dino_v2_giant': 'facebook/dinov2-giant-imagenet1k-1-layer',
    'dino_v2_robust_small': 'facebook/dinov2-small-imagenet1k-1-layer',
    'dino_v2_robust_base':  'facebook/dinov2-base-imagenet1k-1-layer',
    'dino_v2_robust_large': 'facebook/dinov2-large-imagenet1k-1-layer',
    'dino_v3': 'facebook/dinov3-vitb16-pretrain-lvd1689m',
    'dino_v3_large': 'facebook/dinov3-vitl16-pretrain-lvd1689m',
    'convnext_small':  "facebook/convnext-small-224",
    'convnext':        "facebook/convnext-base-224",
    'convnext_large':  "facebook/convnext-large-224",
    'convnext_xlarge': "facebook/convnext-xlarge-224-22k-1k",
    'convnext_v2':       'facebook/convnextv2-base-1k-224',
    'convnext_v2_large': 'facebook/convnextv2-large-1k-224',
    'convnext_v2_huge':  'facebook/convnextv2-huge-1k-224',
    'alexnet': "torchvision/alexnet",
    'siglip2': "google/siglip2-giant-opt-patch16-384",
    'perception-lm': "facebook/Perception-LM-1B",
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

    # add a model_config dictionary with keys patch_size, added_tokens, hidden_size, tokens ?

    # Load model and processor (same as in train_ddp)
    if model_name == 'resnet-50':
        processor = AutoImageProcessor.from_pretrained(model_ID,
                                                    use_fast=True,
                                                    do_center_crop=False,
                                                    do_resize=True,
                                                    size={"shortest_edge": 224}
                                                    )
        model = AutoModel.from_pretrained(model_ID)

        encoder_config = {
            "model_name": model_name,
            "patch_size": None,
            "added_tokens": 0,
            "hidden_size": 2048,
            "image_size": 224,
            "feature_map_size": 7,
            "tubelet_size": 1,
        }

        encoder_config["num_patches"] = 7 * 7

    elif model_name.startswith('resnet-50_adv'):

        processor = AutoImageProcessor.from_pretrained(
                                    "microsoft/resnet-50",
                                    use_fast=True,
                                    do_center_crop=False,
                                    do_resize=True,
                                    size={"shortest_edge": 224}
                                )

        from torchvision.models import resnet50
        # build torchvision ResNet-50
        model = resnet50(weights=None)
        # remove classification head
        model.fc = nn.Identity()
        # load adversarial checkpoint
        checkpoint = torch.load(model_ID, map_location="cpu")

        # robustbench-style checkpoints often store weights under state_dict
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint

        # remove possible "module." prefixes
        state_dict = {
            k.replace("module.", ""): v
            for k, v in state_dict.items()
        }

        model.load_state_dict(state_dict, strict=False)

        encoder_config = {
            "model_name": model_name,
            "patch_size": None,
            "added_tokens": 0,
            "hidden_size": 2048,
            "image_size": 224,
            "feature_map_size": 7,
            "tubelet_size": 1,
        }

        encoder_config["num_patches"] = 7 * 7

    elif model_name == 'siglip2':
        from types import SimpleNamespace
        from PIL import Image as PILImage

        _processor = AutoProcessor.from_pretrained(model_ID)
        _full_model = AutoModel.from_pretrained(model_ID)
        model = _full_model.vision_model
        del _full_model

        processor = SimpleNamespace(
            size={"shortest_edge": 384},
            resample=PILImage.BICUBIC,
            image_mean=_processor.image_processor.image_mean,
            image_std=_processor.image_processor.image_std,
        )
        del _processor

        encoder_config = {
            "model_name": model_name,
            "patch_size": 16,
            "added_tokens": 0,
            "hidden_size": model.config.hidden_size,
            "num_patches": None,
            "image_size": 384,
            "feature_map_size": None,
            "tubelet_size": 1,
        }

        grid = 384 // encoder_config["patch_size"]
        encoder_config["num_patches"] = grid * grid + encoder_config["added_tokens"]

    elif model_name == 'alexnet':
            # Reuse a standard processor configuration for 224x224 resizing
            processor = AutoImageProcessor.from_pretrained(
                "microsoft/resnet-50",
                use_fast=True,
                do_center_crop=False,
                do_resize=True,
                size={"shortest_edge": 224}
            )
            
            from torchvision.models import alexnet, AlexNet_Weights
            # Load pre-trained weights from torchvision
            model = alexnet(weights=AlexNet_Weights.IMAGENET1K_V1)

            encoder_config = {
                "model_name": model_name,
                "patch_size": None,
                "added_tokens": 0,
                "hidden_size": 256,       # AlexNet features output 256 channels
                "image_size": 224,
                "feature_map_size": 6,    # Feature map grid sizing is 6x6
                "tubelet_size": 1,
            }
            encoder_config["num_patches"] = 6 * 6 # 36 patches total

    elif model_name.startswith('convnext_v2'):
        processor = AutoImageProcessor.from_pretrained(model_ID,
                                                    use_fast=True,
                                                    do_center_crop=False,
                                                    do_resize=True,
                                                    crop_pct=1,
                                                    size={"shortest_edge": 224}
                                                    )
        from transformers import ConvNextV2Model
        model = ConvNextV2Model.from_pretrained(model_ID)

        hidden_size = model.config.hidden_sizes[-1]

        encoder_config = {
            "model_name": model_name,
            "patch_size": None,
            "added_tokens": 0,
            "hidden_size": hidden_size,
            "image_size": 224,
            "feature_map_size": 7,
            "tubelet_size": 1,
        }

        encoder_config["num_patches"] = 7 * 7

    elif model_name.startswith('convnext'):
        processor = AutoImageProcessor.from_pretrained(model_ID,
                                                    use_fast=True,
                                                    do_center_crop=False,
                                                    do_resize=True,
                                                    crop_pct=1,
                                                    size={"shortest_edge": 224}
                                                    )
        from transformers import ConvNextModel
        model = ConvNextModel.from_pretrained(model_ID)

        hidden_size = model.config.hidden_sizes[-1]

        encoder_config = {
            "model_name": model_name,
            "patch_size": None,
            "added_tokens": 0,
            "hidden_size": hidden_size,
            "image_size": 224,
            "feature_map_size": 7,
            "tubelet_size": 1,
        }

        encoder_config["num_patches"] = 7 * 7

    elif model_name == "ViT":
        processor = AutoImageProcessor.from_pretrained(model_ID,
                                                    use_fast=True,
                                                    do_center_crop=False,
                                                    do_resize=True,
                                                    crop_pct=1,
                                                    size={"shortest_edge": 224}
                                                    )
        from transformers import ViTModel
        model = ViTModel.from_pretrained(model_ID)

        encoder_config = {
            "model_name": model_name,
            "patch_size": 16,
            "added_tokens": 1,
            "hidden_size": model.config.hidden_size,
            "image_size": 224,
            "tubelet_size": 1,
        }

        # derived field
        grid = 224 // encoder_config["patch_size"]
        encoder_config["num_patches"] = grid * grid + encoder_config["added_tokens"]

    elif model_name.startswith('dino_v2_robust'):
        print("[DEBUG] Running robust dino_v2")
        processor = AutoImageProcessor.from_pretrained(model_ID,
                                               use_fast=True,
                                               do_center_crop=False,
                                               do_resize=True,
                                               crop_pct=1,
                                               size={"shortest_edge": 224})
        model = AutoModel.from_pretrained(model_ID)

        path_to_tokens = '/braintree/data2/active/users/aicha/Ego4D_data'
        #path_to_tokens = '/home/aicha/orcd/pool/Ego4D_data/robustness_tokens_dinov2'

        token_files = {
            'dino_v2_robust_small': f'{path_to_tokens}/robustness_tokens_dinov2_small.pt',
            'dino_v2_robust_base':  f'{path_to_tokens}/robustness_tokens_dinov2_base.pt',
            'dino_v2_robust_large': f'{path_to_tokens}/robustness_tokens_dinov2_large.pt',
        }

        # load 10 pretrained robustness tokens
        robustness_tokens = torch.load(
            token_files[model_name],
            map_location='cpu',
        )['rtokens']

        model.embeddings.robustness_tokens = nn.Parameter(
            robustness_tokens.squeeze(0),
            requires_grad = False,
        )

        original_embed_forward = model.embeddings.forward
        def patched_embed_forward(pixel_values, bool_masked_pos=None):
            out = original_embed_forward(pixel_values, bool_masked_pos)
            B = out.shape[0]
            rob = model.embeddings.robustness_tokens.unsqueeze(0).expand(B, -1, -1)
            return torch.cat([out[:, :1], rob, out[:, 1:]], dim=1)
        model.embeddings.forward = patched_embed_forward

        encoder_config = {
            "model_name": model_name,
            "patch_size": 14,
            "added_tokens": 1 + 10,
            "hidden_size": model.config.hidden_size,
            "num_patches": None,
            "image_size": 224,
            "feature_map_size": None,
            "tubelet_size": 1,
        }

        grid = 224 // encoder_config["patch_size"]
        encoder_config["num_patches"] = grid * grid + encoder_config["added_tokens"]

    elif model_name.startswith("dino_v2"):
        processor = AutoImageProcessor.from_pretrained(model_ID,
                                                       use_fast=True,
                                                       do_center_crop=False,
                                                       do_resize=True,
                                                       crop_pct=1,
                                                       size={"shortest_edge": 224}
                                                       )
        model = AutoModel.from_pretrained(model_ID)

        encoder_config = {
            "model_name": model_name,
            "patch_size": 14,
            "added_tokens": 1, 
            "hidden_size": model.config.hidden_size,
            "image_size": 224,
            "tubelet_size": 1,
        }

        grid = 224 // 14
        encoder_config["num_patches"] = grid * grid + encoder_config["added_tokens"]
        
    elif model_name.startswith('dino_v3'):
        processor = AutoImageProcessor.from_pretrained(model_ID,
                                                    do_resize=True,
                                                    size={"shortest_edge": 224})
        model = AutoModel.from_pretrained(model_ID)

        encoder_config = {
            "model_name": model_name,
            "patch_size": 16,
            "added_tokens": 1 + 4,
            "hidden_size": model.config.hidden_size,
            "image_size": 224,
            "tubelet_size": 1,
        }

        # derived field
        grid = 224 // encoder_config["patch_size"]
        encoder_config["num_patches"] = grid * grid + encoder_config["added_tokens"]

    elif model_name == 'iJEPA':
        processor = AutoImageProcessor.from_pretrained(model_ID,
                                                    use_fast=True,
                                                    do_center_crop=False,
                                                    do_resize=True,
                                                    crop_pct=1,
                                                    size={"shortest_edge": 224}
                                                    )
        from transformers import IJepaModel
        model = IJepaModel.from_pretrained(model_ID)

        encoder_config = {
            "model_name": model_name,
            "patch_size": 14,
            "added_tokens": 0,
            "hidden_size": model.config.hidden_size,
            "image_size": 224,
            "tubelet_size": 1,
        }

        # derived field
        grid = 224 // encoder_config["patch_size"]
        encoder_config["num_patches"] = grid * grid + encoder_config["added_tokens"]

    else:
        raise NotImplementedError

    return model, processor, encoder_config

# Helpers
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

class FrameEncoder_ForHumanSensoryHistoryReports(VJEPA2PreTrainedModel):
    """
    Custom video classificaiton/regression model built on top of:
    - An encoder
    - An attentive pooler that summarizes the sequence
    - A linear classifier head that predicts the ifnal labels
    """
    def __init__(self, encoder, config: VJEPA2Config, encoder_config: dict):
        super().__init__(config)

        self.encoder_config = encoder_config
        self.model_name = self.encoder_config["model_name"]
        self.num_labels = config.num_labels
        self.encoder = encoder # feature extractor from input frames
        # Classifier head
        self.pooler = VJEPA2AttentivePoolerMasked(config) # takes embeddings and reduces them to one vector per video
        self.classifier = nn.Linear(config.hidden_size, config.num_labels, bias=True) # linear layer, so the final output has shape (batch_size, num_labels)
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

    #    labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
    #        Labels for computing the image classification/regression loss. Indices should be in `[0, ...,
    #        config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
    #        `config.num_labels > 1` a classification loss is computed (Cross-Entropy).

        # outputs = self.encoder(
        #         pixel_values_videos=pixel_values_videos,
        #         skip_predictor=True,
        #         output_attentions=output_attentions,
        #         output_hidden_states=output_hidden_states,
        # )

        # pass videos to the encoder
        # (we're doing this by batch, but we're looping over the temporal dimension of each video to have the correct shape)
        
        if self.model_name == 'resnet-50' or self.model_name.startswith('convnext'):
            outputs = [self.encoder(pixel_values=pixel_values_videos[v], output_hidden_states=True) for v in range(pixel_values_videos.shape[0])]
            last_hidden_states = []
            for o in outputs:
                feat = o.last_hidden_state  # (T, C, 7, 7) -> resnet C=2048, convnext C=1536 
                T, C, H, W = feat.shape
                feat = feat.permute(0, 2, 3, 1)  # (T, 7, 7, 2048)
                feat = feat.reshape(T, H * W, C)  # (T, 49, 2048)
                last_hidden_states.append(feat)
                # need to hardcode the mask for CNNs (need the same shape as the activations)
                # features.shape = [B, 49, C]
                # mask = torch.ones(B, 49).to(features.device)
        elif self.model_name.startswith('resnet-50_adv'):
            last_hidden_states = []

            for v in range(pixel_values_videos.shape[0]):

                x = pixel_values_videos[v]  # (T, C, H, W)

                # manually run ResNet up to layer4
                x = self.encoder.conv1(x)
                x = self.encoder.bn1(x)
                x = self.encoder.relu(x)
                x = self.encoder.maxpool(x)

                x = self.encoder.layer1(x)
                x = self.encoder.layer2(x)
                x = self.encoder.layer3(x)
                x = self.encoder.layer4(x)

                # x shape = (T, 2048, 7, 7)

                T, C, H, W = x.shape

                x = x.permute(0, 2, 3, 1)   # (T, 7, 7, 2048)
                x = x.reshape(T, H * W, C)  # (T, 49, 2048)

                last_hidden_states.append(x)
        elif self.model_name == 'alexnet':
            last_hidden_states = []
            for v in range(pixel_values_videos.shape[0]):
                x = pixel_values_videos[v]          # (T, 3, H, W)
                x = self.encoder.features(x)        # (T, 256, 6, 6)
                x = self.encoder.avgpool(x)         # (T, 256, 6, 6)  — keeps 6×6 with AdaptiveAvgPool
                T, C, H, W = x.shape
                x = x.permute(0, 2, 3, 1)           # (T, 6, 6, 256)
                x = x.reshape(T, H * W, C)          # (T, 36, 256)
                last_hidden_states.append(x)
        else:
            outputs = [self.encoder(pixel_values=pixel_values_videos[v], output_hidden_states=True) for v in range(pixel_values_videos.shape[0])]
            last_hidden_states = [o['hidden_states'][-1] for o in outputs]
        
        # stack batch 
        last_hidden_states = torch.stack(last_hidden_states, dim=0)
        
        if self.model_name == 'alexnet' or self.model_name.startswith('resnet') or self.model_name.startswith('convnext'):            
            B, T, P, C = last_hidden_states.shape
            last_hidden_states = last_hidden_states.view(B, T * P, C)

        else:
            # flatten time and features
            last_hidden_states = torch.flatten(last_hidden_states, start_dim=1, end_dim=2)

        # pass to the attentive pooler
        # IMPORTANT: attention_mask tells the model which parts of the input are real data and which are padding
        # without attention_mask, the model would treat padded frames (fake data) as real, therefore learning wrong patterns
        pooler_output = self.pooler(last_hidden_states, attention_mask=context_masks)

        # final prediction
        logits = self.classifier(pooler_output)

        #logits = torch.sigmoid(logits)

        loss = None
        if labels is not None:
            loss = self.loss_function_manual(logits, labels) # MSE
            #loss = self.loss_function(pooled_logits=logits, labels=labels.unsqueeze(0), config=self.config)

        # Hugging Face-style output object containing loss and logits
        return ImageClassifierOutput(
            loss=loss,
            logits=logits
            #hidden_states=outputs.hidden_states,
            #attentions=outputs.attentions,
        )

class CustomVideoDataset(Dataset):
    """
    Modified Dataset that performs video decoding (=make videos ready to be input to models for training) in __getitem__ (worker process)
    For each index, it:
    - Opens video
    - Sub-samples segments (short video clips)
    - Decodes frames (i.e. turns them into pixel data/images, ready for model-training)
    - Pads short clips if needed
    - Builds a mask to keep track of real vs "fake" (padded) frames
    - Return clips, label, video ID and masks
    """

    def __init__(self, video_file_paths, labels, frames_per_clip, sampling=1,
                 frame_jitter=0, max_segments=8,
                fill_value=128, addFolderToID=None,
                 decoder_seek_mode = "exact", device=None, num_ffmpeg_threads=None,
                 speed_jitter=False,
                 encoder_config=None):
        self.video_file_paths = video_file_paths
        self.labels = labels
        self.frames_per_clip = frames_per_clip
        self.sampling = sampling
        self.speed_jitter = speed_jitter
        self.frame_jitter = frame_jitter
        self.max_segments = max_segments
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

        # generate "timestamps" to slice video into small segments
        # we do this to decode the video per segments, and not the entire video at once
        # speed jitter (data augmentation): randomly vary the sampling rate to simulate fast/slow motion
        sampling = random.choice([3, 4, 6]) if self.speed_jitter else self.sampling
        segments = generate_segments(
            num_frames,
            self.frames_per_clip,
            sampling, #self.sampling
            self.frame_jitter
        )

        # max number of segments for this video
        segments = segments[:self.max_segments]

        # Decode all segments in worker
        clips = []
        masks = []

        patch_size = self.encoder_config["patch_size"]
        added_tokens = self.encoder_config["added_tokens"]
        tubelet_size = self.encoder_config["tubelet_size"]
        #patches = self.added_tokens + self.patch_size ** 2
        ## ??
        patches = self.encoder_config["num_patches"]
        tubelets = self.frames_per_clip // tubelet_size

        for seg in segments[::-1]:
            video_frames = decoder.get_frames_at(indices=seg).data

            # Create mask for this segment
            # the mask marks vali data (0 for missing tokens, 1 for valid tokens)
            mask_idx = torch.arange(
                (self.frames_per_clip - len(seg)) // tubelet_size * patches,
                tubelets * patches
            )
            mask = torch.zeros((tubelets * patches,))
            mask[mask_idx] = 1
            masks.append(mask)

            # Pad if necessary
            # if there are any missing tokens, they are replaced in the data by constant frames (value = fill_value)
            # this is important because if there are missing frames, the clip will not have shape (frames_per_clip, C, H, W), and NNs require fixed-size inputs
            if video_frames.shape[0] < self.frames_per_clip:
                missing_frames = self.frames_per_clip - video_frames.shape[0]
                padding = self.fill_value * torch.ones(
                    (missing_frames, *video_frames.shape[1:]),
                    dtype=video_frames.dtype
                )
                video_frames = torch.cat([padding, video_frames], dim=0)

            clips.append(video_frames)

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

def setup(rank, world_size):
    """Initialize the distributed environment."""
    os.environ['MASTER_ADDR'] = 'localhost' # set master address, here same machine
    os.environ['MASTER_PORT'] = '12355' # communication port
    # CRITICAL: Set device before init_process_group
    torch.cuda.set_device(rank) # assign GPU to this process/rank (each process uses a different GPU)
    # Initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size) # creates communication between processes

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

# Main train/eval functions
def train_ddp(rank, world_size, model_name, result_dir, experiment_id, frames, 
              dataset_root_path, debug=False, num_epochs=20, batch_size=2, 
              continue_training=False, checkpoint_path=None,
              fps=4, 
              no_augmentations=False,
              tube_mask_ratio=0.0, frame_drop_ratio=0.0, speed_jitter=False, static_vids_T=False, upsample_rare_categories=True,
              frame_drop_ratio_2=0.0,
              base_lr=1e-3, weight_decay=0.8):
    """Main training function for each process."""

    # Setup distributed training (across multiple GPUs/processes)
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

    # vis_df = pd.read_csv(dataset_root_path / "visualization_humanReports.csv")
    # bench_df = pd.read_csv(dataset_root_path / "benchmarks_humanReports.csv")
    # bench_df.loc[bench_df[
    #     'Benchmark_subcondition'].isna(), 'Benchmark_subcondition'] = ''  # This is necessary to make the file paths work. Otherwise, it's coded as nan.

    #train_df = pd.concat([train_df, val_df], ignore_index=True) #

    # Only include short videos for fitting to avoid an overreliance on the context
    # Necessary because we train different models to correspond to the duration covered by the model (due to different input sizes)

    if upsample_rare_categories:
        print("Upsampling Rare Categories")
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

    train_df = train_df.loc[train_df['videoDuration (sec)'] <= np.max([frames/fps, 0.5])]
    val_df = val_df.loc[val_df['videoDuration (sec)'] <= np.max([frames/fps, 0.5])]

    # remove static (1 cluster or less) train videos shorter than static_vids_T
    if static_vids_T > 0.0:
        static_vids = train_df[train_df['n_clusters'] <= 1]
        static_vids_to_remove = static_vids[static_vids['videoDuration (sec)'] <= static_vids_T]
        train_df = train_df.drop(index=static_vids_to_remove.index)

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
    max_segments_per_video = 1 # this will keep the last part of the video (which could need padding if the video is not long enough)
    prefetch_factor = 2  # 4qq
    sampling = 30 // fps  # videos were encoded at 30fps, we're sampling at 5Hz, thus every 200ms
    # so the original video has 30 frames per second (which was defined when the dataset was created)
    # and the desired sampling is 5 frames per second
    # so we keep 1 every 30 // fps = 6 frames
    frames_per_clip = frames  # 40#2#config.frames_per_clip  # this is 64, thus 12.8s (64 x 0.2s) maximum video duration.
    frame_jitter = 3

    model, processor, encoder_config = get_encoder(model_name)
    
    train_ds = CustomVideoDataset(
        train_video_file_paths.tolist(),
        train_labels,
        frames_per_clip=frames_per_clip,
        sampling=sampling,
        frame_jitter=frame_jitter,
        max_segments=max_segments_per_video,
        decoder_seek_mode='approximate',
        #device='cuda' if torch.cuda.is_available() else 'cpu',
        num_ffmpeg_threads=num_workers,
        speed_jitter=speed_jitter,
        encoder_config=encoder_config,
    )

    val_ds = CustomVideoDataset(
        val_video_file_paths.tolist(),
        val_labels,
        frames_per_clip=frames_per_clip,
        sampling=sampling,
        frame_jitter=0,  # No jitter for validation
        max_segments=max_segments_per_video,
        decoder_seek_mode='approximate',
        #device='cuda' if torch.cuda.is_available() else 'cpu',
        num_ffmpeg_threads=num_workers,
        encoder_config=encoder_config
    )

    test_ds = CustomVideoDataset(
        test_video_file_paths.tolist(),
        test_labels,
        frames_per_clip=frames_per_clip,
        sampling=sampling,
        frame_jitter=0,  # No jitter for test
        max_segments=max_segments_per_video,
        #device='cuda' if torch.cuda.is_available() else 'cpu',
        num_ffmpeg_threads=num_workers,
        encoder_config=encoder_config
    )

    # add encoder_config as an output of get_encoder, so no need to hardcode hidden size

    config = AutoConfig.from_pretrained("facebook/vjepa2-vitl-fpc64-256") # vJEPA2 config
    config.hidden_size = encoder_config["hidden_size"]
    config.num_labels = 12
    model_SH = FrameEncoder_ForHumanSensoryHistoryReports(model, config, encoder_config).to(device)

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
    if no_augmentations:
        train_transforms = v2.Compose([
            v2.Resize((processor.size["shortest_edge"], processor.size["shortest_edge"]), interpolation=processor.resample),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=processor.image_mean, std=processor.image_std),
        ])
    else:
        train_transforms = v2.Compose([
            v2.AutoAugment(),
            v2.RandomResizedCrop((processor.size["shortest_edge"], processor.size["shortest_edge"]), scale=(0.9, 1.0), interpolation=processor.resample),#, ratio=(3/4, 4/3)), ineffective because videos are square.
            v2.RandomHorizontalFlip(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=processor.image_mean, std=processor.image_std),
        ])


    eval_transforms = v2.Compose([
        v2.Resize((processor.size["shortest_edge"],  processor.size["shortest_edge"]), interpolation=processor.resample),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=processor.image_mean, std=processor.image_std)
    ])

    # from PIL import Image
    # test_img = torch.randint(0, 256, size=(3, 1080, 1080), dtype=torch.uint8)
    # #test_img_np = test_img.permute(1, 2, 0).numpy()
    # #test_img_pil = Image.fromarray(test_img_np)
    #
    #
    # test_img_processed_v0 = processor(test_img, return_tensors="pt")["pixel_values"][0]
    # test_img_processed_v1 = eval_transforms(test_img)

    # [processor(v.squeeze(), return_tensors="pt").to(device) for v in vids]

    train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True)  

    val_sampler = DistributedSampler(val_ds, num_replicas=world_size, rank=rank, shuffle=False)
    test_sampler = DistributedSampler(test_ds, num_replicas=world_size, rank=rank, shuffle=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=train_sampler,
        collate_fn=partial(collate_fn_all_segments, transforms=train_transforms, tube_mask_ratio=tube_mask_ratio, frame_drop_ratio=frame_drop_ratio, frame_drop_ratio_2=frame_drop_ratio_2),
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

    loss_func = nn.MSELoss()

    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs * len(train_loader) // accumulation_steps)

    if continue_training:
        if (checkpoint['scheduler_state_dict'] is not None):
            print('Loading scheduler state...')
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        start_epoch = checkpoint['epoch'] + 1
    else:
        start_epoch=1

    total_t0 = time.time()
    for epoch in range(start_epoch, num_epochs + 1):
        if rank == 0:
            print(f"Epoch {epoch}")

        # set epoch for distributed sampler
        if hasattr(train_sampler, 'set_epoch'):
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
        tot_train_time = (time.time() - total_t0) / 3600
        print(f"Total training time: {tot_train_time:.2f}h", flush=True)

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
            f'{result_dir}/{model_name}-{experiment_id}_preds_{frames_per_clip}Frames_AttentionalPooling.csv'
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

            if model_name != "siglip2":
                processor.save_pretrained(f"{result_dir}/{experiment_id}")

            with open(f"{result_dir}/{experiment_id}/status.json", "w") as f:
                json.dump({"epoch": epoch, "val_mse": float(val_mse)}, f)

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

    bench_df.loc[bench_df['Benchmark_subcondition'].isna(), 'Benchmark_subcondition'] = ''

    # Update file paths to SH videos, to evaluate videos without noise
    bench_df.loc[bench_df['Benchmark_condition'] == 'SensoryHistory', 'Benchmark_condition'] = 'SensoryHistory_withoutNoise'

    if rank == 0:
        video_count_vis = len(vis_df)
        video_count_bench = len(bench_df)
        print(f"Visualization videos: {video_count_vis}")
        print(f"Benchmark videos: {video_count_bench}")

    # Prepare datasets
    vis_video_file_paths = dataset_root_path / 'Benchmarks' / 'Visualization' / (
                vis_df['videoDuration (sec)'].astype(str) + 's') / (vis_df['File_name'] + '.mp4')
    bench_video_file_paths = dataset_root_path / 'Benchmarks' / bench_df['Benchmark_condition'] / bench_df[
        'Benchmark_subcondition'] / (bench_df['File_name'] + '.mp4')

    vis_labels = vis_df[categories].values
    bench_labels = bench_df[categories].values

    num_workers = 4  # 4  # 4#2  # Reduced for stability
    max_segments_per_video = 1
    prefetch_factor = 2  # 4qq
    sampling = 30 // fps  # videos were encoded at 30fps, we're sampling at 5Hz, thus every 200ms
    frames_per_clip = frames  # 40#2#config.frames_per_clip  # this is 64, thus 12.8s (64 x 0.2s) maximum video duration.
    frame_jitter = 3

    #vis_ds = CustomVideoDataset(vis_video_file_paths.tolist(), vis_labels, addFolderToID=-2)
    #bench_ds = CustomVideoDataset(bench_video_file_paths.tolist(), bench_labels, addFolderToID=-3)
        
    model, processor, encoder_config = get_encoder(model_name)  

    vis_ds = CustomVideoDataset(
        vis_video_file_paths.tolist(),
        vis_labels,
        frames_per_clip=frames_per_clip,
        sampling=sampling,
        frame_jitter=0,  # No jitter for test
        max_segments=max_segments_per_video,
        # device='cuda' if torch.cuda.is_available() else 'cpu',
        num_ffmpeg_threads=num_workers,
        addFolderToID=-2,
        encoder_config=encoder_config
    )
    bench_ds = CustomVideoDataset(
        bench_video_file_paths.tolist(),
        bench_labels,
        frames_per_clip=frames_per_clip,
        sampling=sampling,
        frame_jitter=0,  # No jitter for test
        max_segments=max_segments_per_video,
        # device='cuda' if torch.cuda.is_available() else 'cpu',
        num_ffmpeg_threads=num_workers,
        addFolderToID=-3,
        encoder_config=encoder_config
    )

    # Load trained model configuration
    config = AutoConfig.from_pretrained("facebook/vjepa2-vitl-fpc64-256")
    config.hidden_size = encoder_config["hidden_size"]
    config.num_labels = 12
    model_SH = FrameEncoder_ForHumanSensoryHistoryReports(model, config, encoder_config).to(device)

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

    eval_transforms = v2.Compose([
        v2.Resize((processor.size["shortest_edge"], processor.size["shortest_edge"]), interpolation=processor.resample),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=processor.image_mean, std=processor.image_std)
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

    # Compute activations
    if rank == 0:
        print('Computing visualization set activations...')
    total_t0 = time.time()
    vis_activations, vis_ids, vis_labels_tensor = get_predictions(
        vis_loader, model_SH, processor, device, rank,
        #interpolate_pos_encoding=interpolate_pos_encoding
    )
    tot_eval_time = (time.time() - total_t0) / 3600
    print(f"Total eval time: {tot_eval_time:.2f}h", flush=True)

    if rank == 0:
        print('Computing benchmark set activations...')
    total_t0 = time.time()
    bench_activations, bench_ids, bench_labels_tensor = get_predictions(
        bench_loader, model_SH, processor, device, rank,
        #interpolate_pos_encoding=interpolate_pos_encoding
    )
    tot_bench_time = (time.time() - total_t0) / 3600
    print(f"Total eval time: {tot_bench_time:.2f}h", flush=True)

    # Save results (only from main process)
    if rank == 0:
        vis_predictions = vis_activations.cpu().numpy()
        vis_labels_np = vis_labels_tensor.cpu().numpy()

        bench_predictions = bench_activations.cpu().numpy()
        bench_labels_np = bench_labels_tensor.cpu().numpy()

        results_vis = []
        results_bench = []

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

        results_vis = pd.concat(results_vis, ignore_index=True)
        results_bench = pd.concat(results_bench, ignore_index=True)


        results_vis.to_csv(
            f'{result_dir}/{model_name}-{experiment_id}_preds_{frames_per_clip}Frames_AttentionalPooling_visualization.csv',
            index=False
        )
        results_bench.to_csv(
            f'{result_dir}/{model_name}-{experiment_id}_preds_{frames_per_clip}Frames_AttentionalPooling_benchmark.csv',
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
    parser.add_argument("--encoder", type=str, default="dino_v2_robust_small",
                        choices=[
                            "alexnet", "siglip2",
                            "resnet-50",
                            "resnet-50_adv_eps0", "resnet-50_adv_eps0.01", "resnet-50_adv_eps0.03",
                            "resnet-50_adv_eps0.05", "resnet-50_adv_eps0.1", "resnet-50_adv_eps0.25",
                            "resnet-50_adv_eps0.5", "resnet-50_adv_eps1", "resnet-50_adv_eps3", "resnet-50_adv_eps5",
                            "ViT", "ViT_large", "iJEPA",
                            "dino_v2_small", "dino_v2_base", "dino_v2", "dino_v2_giant",
                            "dino_v2_robust_small", "dino_v2_robust_base", "dino_v2_robust_large",
                            "dino_v3",
                            "convnext", "convnext_small", "convnext_large", "convnext_xlarge",
                            "convnext_v2", "convnext_v2_large", "convnext_v2_huge",
                        ],
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

    modes = ['train'] #, 'eval']
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

    frames = [60] # [1, 3, 10, 20, 40]
    num_epochs=15
    fps = 4
    batch_size=args.batch_size # * world_size # 10 for A100/H100, L40S do 6
    approach = args.approach # all approaches that we try to optimize the model (data augmentation...)
    # Set augmentation ratios based on approach (0.0 = disabled for baseline)
    tube_mask_ratio = 0.00001 if approach == "tubing" else 0.0
    frame_drop_ratio = 0.00001 if approach == "frame_dropping"  else 0.0
    frame_drop_ratio_2 = 0.5 if approach == "frame_dropping_2"  else 0.0
    speed_jitter = True if approach == "speed" else False
    static_vids_T = 0.8 if approach == "remove_static" else 0.0 # if we want to remove static videos, we remove STATIC videos (1 cluster or less) that are shorter than static_vids_T (sec)
    upsample_rare_categories = True if approach == "upsample_rare_categories" else False
    no_augmentations = True if approach == "vanilla_model" else False

    if base_dir.startswith('/braintree'):
        result_dir = f'/braintree/home/aicha/tests/{model_name}_{approach}_og_dataset'
    else:
        #result_dir = f'{base_dir}/{model_name}_{approach}'
        result_dir = f'/orcd/data/dicarlo/001/om/lynnka/{model_name}_{approach}_og_dataset'
    if not os.path.exists(result_dir):
        os.makedirs(result_dir, exist_ok=True)

    learning_rates = [1e-3]
    weight_decays = [0.8]

    for lr in learning_rates:
        for wd in weight_decays:
            for frame in frames:
                experiment_id = f'{model_name}_{frame}frames_{num_epochs}epochs_lr_{lr}_wd_{wd}_{approach}_fps_4'
                status_path = f"{result_dir}/{experiment_id}/status.json"
                checkpoint_path = f"{result_dir}/{experiment_id}/checkpoint_final.pt"
                if os.path.exists(status_path):
                    with open(status_path) as f:
                        last_epoch = json.load(f)["epoch"]
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
                        print(f'Training {model_name}-{experiment_id} for {num_epochs} epochs (batch size {batch_size}) using approach {approach}...')
                        # Spawn processes for distributed training
                        mp.spawn( 
                            train_ddp,
                            args=(world_size, model_name, result_dir, experiment_id, frame, 
                                    dataset_root_path, debug, num_epochs, batch_size, 
                                    continue_training, checkpoint_path,
                                    fps, 
                                    no_augmentations,
                                    tube_mask_ratio, frame_drop_ratio, speed_jitter, static_vids_T, upsample_rare_categories,
                                    frame_drop_ratio_2,
                                    lr, wd),  # world_size and num_epochs
                            nprocs=world_size,
                            join=True
                        )
                    elif mode == 'eval':
                        if os.path.exists(result_dir + '/' + experiment_id):

                            print(f'Evaluating {model_name}-{experiment_id} ...')
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
