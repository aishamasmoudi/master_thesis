import torch


def apply_tube_mask(clips, masks, tube_mask_ratio=0.5, n_blocks=2):
    """
    Apply n_blocks contiguous rectangular tube mask to a batch of video clips,
    and update the attention mask to mark masked tokens as invalid (0).

    Args:
        clips: (num_segments, T, C, H, W) tensor
        masks: (num_segments, T * patches_per_frame) tensor
        tube_mask_ratio: fraction of spatial patches to mask
        n_blocks: number of blocks

    Returns:
        clips : unchanged
        masks : clone with masked-patch entries set to 0
    """
    masks = masks.clone()
    num_segments, T, C, H, W = clips.shape

    patches_per_frame = masks.shape[-1] // T
    added_tokens = patches_per_frame - (H // 16) * (W // 16)

    # patch_size refers to the size in pixels of each patch, with 16x16 being the standard ViT patch size
    # it is only used to convert between patch grid coordinates and pixel coordinates
    # this is necessary to avoid masking at non-patch-aligned pixel boundaries
    # NOTE: this would never happen in our situation since H and W are ALWAYS 224, but we still don't hardcode it
    patch_size = 16
    n_patches_h = H // patch_size # nb of patches "available for masking" along the height axis
    n_patches_w = W // patch_size # nb of patches "available for masking" along the width axis
    n_patches = n_patches_h * n_patches_w # total number of patches in the spatial grid
    n_masked = int(n_patches * tube_mask_ratio) # amount of patches to mask tube_mask_ratio*100 of the frame
    if n_masked == 0:
        return clips, masks
    
    # place n_blocks blocks whose union approximates the target area (determined by n_masked)
    # each block covers 1/n_blocks of the target area on average
    patches_per_block = max(1, n_masked // n_blocks)

    for _ in range(n_blocks):

        # randomly pick block height (in patches), then derive width from target area
        block_h = torch.randint(1, n_patches_h + 1, (1,)).item()
        block_w = min(round(patches_per_block / block_h), n_patches_w)
        block_w = max(1, block_w)  # ensure at least 1

        # randomly pick top-left corner (to ensure the block fits entirely within the grid)
        start_row = torch.randint(0, max(1, n_patches_h - block_h + 1), (1,)).item() 
        start_col = torch.randint(0, max(1, n_patches_w - block_w + 1), (1,)).item()

        # only update the mask, leave clips untouched
        for r in range(start_row, start_row + block_h):
            for c in range(start_col, start_col + block_w):
                patch_idx = r * n_patches_w + c
                token_idx = added_tokens + patch_idx
                for t in range(T):
                    masks[:, t * patches_per_frame + token_idx] = 0

    return clips, masks


def apply_tube_masks(clips, masks, tube_mask_ratio=0.5):
    """
    Calls apply_tube_mask twice, once short-range (8 blocks, each small) and once long-range (2 blocks, each large).
    """
    # short-range: 8 small blocks, together covering ~tube_mask_ratio of the frame
    clips, masks = apply_tube_mask(clips, masks, tube_mask_ratio=tube_mask_ratio, n_blocks=8)

    # long-range: 2 large blocks, together covering ~tube_mask_ratio of the frame
    clips, masks = apply_tube_mask(clips, masks, tube_mask_ratio=tube_mask_ratio, n_blocks=2)

    return clips, masks


def apply_frame_drop(clips, masks, frame_drop_ratio=0.3):
    """
    Apply random frame dropping via the attention mask.
    
    The function receives the mask in its (num_segments, T * patches_per_frame) shape
    (which is later used as an input to the attention pooler to ignore positions where the mask is 0).
    For each segment independently, it randomly selects a fraction of frames and
    zeros out the entires corresponding to dropped frames to also mark them as invalid in the attention mask. 
 
    Args:
        clips           : (num_segments, T, C, H, W) tensor
        masks           : (num_segments, T * patches_per_frame) tensor
        frame_drop_ratio: fraction of frames to drop per segment, e.g. 0.3 = 30%
 
    Returns:
        clips : unchanged
        masks : clone with dropped-frame entries set to 0
    """
    # make a copy of the original mask
    masks = masks.clone()
    num_segments, T, C, H, W = clips.shape
    # infer tokens per frame
    patches = masks.shape[-1] // T  # e.g. 257 for DINOv2 patch16
 
    # how many frames to drop
    n_drop = int(T * frame_drop_ratio)
    if n_drop == 0:
        return clips, masks
    
    # for each segment, pick random frames and zero their mask entries
    for s in range(num_segments):
        # each segment gets an independent random set of dropped frames
        drop_ids = torch.randperm(T)[:n_drop]
        for fid in drop_ids:
            # for each dropped frame, zero out its corresponding slice in the mask
            masks[s, fid * patches : (fid + 1) * patches] = 0 # e.g. all 257 tokens for this frame are marked invalid
 
    return clips, masks

def apply_frame_drop_2(clips, masks, frame_drop_ratio=0.3):
    # make a copy of the original mask
    masks = masks.clone()
    num_segments, T, C, H, W = clips.shape
    # infer tokens per frame
    patches = masks.shape[-1] // T  # e.g. 257 for DINOv2 patch16

    for s in range(num_segments):
        # identify which frames are real (not already zeroed by padding)
        real_frames = [
            fid for fid in range(T)
            if masks[s, fid * patches : (fid + 1) * patches].sum() > 0
        ]

        if len(real_frames) == 0:
            continue  # fully padded clip, nothing to drop

        # drop only from real frames
        n_drop = int(len(real_frames) * frame_drop_ratio)
        if n_drop == 0:
            continue

        drop_ids = torch.randperm(len(real_frames))[:n_drop]
        for idx in drop_ids:
            fid = real_frames[idx]
            masks[s, fid * patches : (fid + 1) * patches] = 0

    return clips, masks