# Copyright (c) Facebook, Inc. and its affiliates.
import numpy as np
from typing import Tuple
import torch
from PIL import Image
from torch.nn import functional as F

from detectron2.structures import Boxes

__all__ = ["paste_masks_in_image"]


BYTES_PER_FLOAT = 4
# TODO: This memory limit may be too much or too little. It would be better to
# determine it based on available resources.
GPU_MEM_LIMIT = 1024 ** 3  # 1 GB memory limit


def _do_paste_mask(masks, boxes, img_h: int, img_w: int, skip_empty: bool = True):
    """
    Args:
        masks: N, 1, H, W
        boxes: N, 4
        img_h, img_w (int):
        skip_empty (bool): only paste masks within the region that
            tightly bound all boxes, and returns the results this region only.
            An important optimization for CPU.

    Returns:
        if skip_empty == False, a mask of shape (N, img_h, img_w)
        if skip_empty == True, a mask of shape (N, h', w'), and the slice
            object for the corresponding region.
    """
    # On GPU, paste all masks together (up to chunk size)
    # by using the entire image to sample the masks
    # Compared to pasting them one by one,
    # this has more operations but is faster on COCO-scale dataset.
    device = masks.device

    mask_boxes = boxes

    if boxes.shape[1] == 5:
        return _do_paste_mask_rotated(masks, boxes, img_h, img_w, skip_empty)
        #mask_boxes = RotatedBoxes(boxes).to_upright_boxes()

    if skip_empty and not torch.jit.is_scripting():
        x0_int, y0_int = torch.clamp(mask_boxes.min(dim=0).values.floor()[:2] - 1, min=0).to(
            dtype=torch.int32
        )
        x1_int = torch.clamp(mask_boxes[:, 2].max().ceil() + 1, max=img_w).to(dtype=torch.int32)
        y1_int = torch.clamp(mask_boxes[:, 3].max().ceil() + 1, max=img_h).to(dtype=torch.int32)
    else:
        x0_int, y0_int = 0, 0
        x1_int, y1_int = img_w, img_h
    x0, y0, x1, y1 = torch.split(mask_boxes, 1, dim=1)  # each is Nx1

    N = masks.shape[0]

    img_y = torch.arange(y0_int, y1_int, device=device, dtype=torch.float32) + 0.5
    img_x = torch.arange(x0_int, x1_int, device=device, dtype=torch.float32) + 0.5
    img_y = (img_y - y0) / (y1 - y0) * 2 - 1
    img_x = (img_x - x0) / (x1 - x0) * 2 - 1
    # img_x, img_y have shapes (N, w), (N, h)

    gx = img_x[:, None, :].expand(N, img_y.size(1), img_x.size(1))
    gy = img_y[:, :, None].expand(N, img_y.size(1), img_x.size(1))
    grid = torch.stack([gx, gy], dim=3)

    if not torch.jit.is_scripting():
        if not masks.dtype.is_floating_point:
            masks = masks.float()
    img_masks = F.grid_sample(masks, grid.to(masks.dtype), align_corners=False)

    if skip_empty and not torch.jit.is_scripting():
        return img_masks[:, 0], (slice(y0_int, y1_int), slice(x0_int, x1_int))
    else:
        return img_masks[:, 0], ()


def _do_paste_mask_rotated(masks, boxes, img_h: int, img_w: int, skip_empty: bool = True):
    import cv2
    
    assert len(masks) == len(boxes)

    img_masks = []

    for mask, box in zip(masks[:,0,:,:], boxes):
        assert len(box) == 5  # xc yc w h angle
        
        img_mask = np.zeros((img_h, img_w))

        # generate the mapping of points from roi_image to an image
        w, h = int(np.round(box[2].cpu())), int(np.round(box[3].cpu()))
        w, h = max(w, 1), max(h, 1) 
        center = (box[0].cpu(), box[1].cpu())
        theta = np.deg2rad(-box[4].cpu())


        try:
            mask = cv2.resize(mask.cpu().numpy(), (w,h))
        except:
            print(f"ERROR : Got box with size [{w}, {h}]")

        # paste mask onto image via rotated rect mapping
        v_x = (np.cos(theta), np.sin(theta))
        v_y = (-np.sin(theta), np.cos(theta))
        s_x = center[0] - v_x[0] * ((w - 1) / 2) - v_y[0] * ((h - 1) / 2)
        s_y = center[1] - v_x[1] * ((w - 1) / 2) - v_y[1] * ((h - 1) / 2)

        M = np.array([[v_x[0], v_y[0], s_x],
                    [v_x[1], v_y[1], s_y]])

        x_grid, y_grid = np.meshgrid(np.arange(w), np.arange(h))
        x_grid = x_grid.reshape(-1)
        y_grid = y_grid.reshape(-1)
        map_pts_x = x_grid * M[0, 0] + y_grid * M[0, 1] + M[0, 2]
        map_pts_y = x_grid * M[1, 0] + y_grid * M[1, 1] + M[1, 2]
        map_pts_x = np.round(map_pts_x).astype(np.int32)
        map_pts_y = np.round(map_pts_y).astype(np.int32)

        valid_x = np.logical_and(map_pts_x >= 0, map_pts_x < img_w)
        valid_y = np.logical_and(map_pts_y >= 0, map_pts_y < img_h)
        valid = np.logical_and(valid_x, valid_y)
        img_mask[map_pts_y[valid], map_pts_x[valid]] = mask[y_grid[valid], x_grid[valid]]

        # close holes that arise due to rounding from the pixel mapping phase
        kernel = np.ones((5, 5), np.uint8)
        img_mask = cv2.morphologyEx(img_mask, cv2.MORPH_CLOSE, kernel)

        img_masks.append(img_mask)

    return torch.tensor(np.array(img_masks)).to(masks.device), ()


def paste_masks_in_image(
    masks: torch.Tensor, boxes: Boxes, image_shape: Tuple[int, int], threshold: float = 0.5
):
    """
    Paste a set of masks that are of a fixed resolution (e.g., 28 x 28) into an image.
    The location, height, and width for pasting each mask is determined by their
    corresponding bounding boxes in boxes.

    Note:
        This is a complicated but more accurate implementation. In actual deployment, it is
        often enough to use a faster but less accurate implementation.
        See :func:`paste_mask_in_image_old` in this file for an alternative implementation.

    Args:
        masks (tensor): Tensor of shape (Bimg, Hmask, Wmask), where Bimg is the number of
            detected object instances in the image and Hmask, Wmask are the mask width and mask
            height of the predicted mask (e.g., Hmask = Wmask = 28). Values are in [0, 1].
        boxes (Boxes or Tensor): A Boxes of length Bimg or Tensor of shape (Bimg, 4).
            boxes[i] and masks[i] correspond to the same object instance.
        image_shape (tuple): height, width
        threshold (float): A threshold in [0, 1] for converting the (soft) masks to
            binary masks.

    Returns:
        img_masks (Tensor): A tensor of shape (Bimg, Himage, Wimage), where Bimg is the
        number of detected object instances and Himage, Wimage are the image width
        and height. img_masks[i] is a binary mask for object instance i.
    """

    assert masks.shape[-1] == masks.shape[-2], "Only square mask predictions are supported"
    N = len(masks)
    if N == 0:
        return masks.new_empty((0,) + image_shape, dtype=torch.uint8)
    if not isinstance(boxes, torch.Tensor):
        boxes = boxes.tensor
    device = boxes.device
    assert len(boxes) == N, boxes.shape

    img_h, img_w = image_shape

    # The actual implementation split the input into chunks,
    # and paste them chunk by chunk.
    if device.type == "cpu" or torch.jit.is_scripting():
        # CPU is most efficient when they are pasted one by one with skip_empty=True
        # so that it performs minimal number of operations.
        num_chunks = N
    else:
        # GPU benefits from parallelism for larger chunks, but may have memory issue
        # int(img_h) because shape may be tensors in tracing
        num_chunks = int(np.ceil(N * int(img_h) * int(img_w) * BYTES_PER_FLOAT / GPU_MEM_LIMIT))
        assert (
            num_chunks <= N
        ), "Default GPU_MEM_LIMIT in mask_ops.py is too small; try increasing it"
    chunks = torch.chunk(torch.arange(N, device=device), num_chunks)

    img_masks = torch.zeros(
        N, img_h, img_w, device=device, dtype=torch.bool if threshold >= 0 else torch.uint8
    )
    for inds in chunks:
        masks_chunk, spatial_inds = _do_paste_mask(
            masks[inds, None, :, :], boxes[inds], img_h, img_w, skip_empty=device.type == "cpu"
        )

        if threshold >= 0:
            masks_chunk = (masks_chunk >= threshold).to(dtype=torch.bool)
        else:
            # for visualization and debugging
            masks_chunk = (masks_chunk * 255).to(dtype=torch.uint8)

        if torch.jit.is_scripting():  # Scripting does not use the optimized codepath
            img_masks[inds] = masks_chunk
        else:
            img_masks[(inds,) + spatial_inds] = masks_chunk
    return img_masks


# The below are the original paste function (from Detectron1) which has
# larger quantization error.
# It is faster on CPU, while the aligned one is faster on GPU thanks to grid_sample.


def paste_mask_in_image_old(mask, box, img_h, img_w, threshold):
    """
    Paste a single mask in an image.
    This is a per-box implementation of :func:`paste_masks_in_image`.
    This function has larger quantization error due to incorrect pixel
    modeling and is not used any more.

    Args:
        mask (Tensor): A tensor of shape (Hmask, Wmask) storing the mask of a single
            object instance. Values are in [0, 1].
        box (Tensor): A tensor of shape (4, ) storing the x0, y0, x1, y1 box corners
            of the object instance.
        img_h, img_w (int): Image height and width.
        threshold (float): Mask binarization threshold in [0, 1].

    Returns:
        im_mask (Tensor):
            The resized and binarized object mask pasted into the original
            image plane (a tensor of shape (img_h, img_w)).
    """
    # Conversion from continuous box coordinates to discrete pixel coordinates
    # via truncation (cast to int32). This determines which pixels to paste the
    # mask onto.
    box = box.to(dtype=torch.int32)  # Continuous to discrete coordinate conversion
    # An example (1D) box with continuous coordinates (x0=0.7, x1=4.3) will map to
    # a discrete coordinates (x0=0, x1=4). Note that box is mapped to 5 = x1 - x0 + 1
    # pixels (not x1 - x0 pixels).
    samples_w = box[2] - box[0] + 1  # Number of pixel samples, *not* geometric width
    samples_h = box[3] - box[1] + 1  # Number of pixel samples, *not* geometric height

    # Resample the mask from it's original grid to the new samples_w x samples_h grid
    mask = Image.fromarray(mask.cpu().numpy())
    mask = mask.resize((samples_w, samples_h), resample=Image.BILINEAR)
    mask = np.array(mask, copy=False)

    if threshold >= 0:
        mask = np.array(mask > threshold, dtype=np.uint8)
        mask = torch.from_numpy(mask)
    else:
        # for visualization and debugging, we also
        # allow it to return an unmodified mask
        mask = torch.from_numpy(mask * 255).to(torch.uint8)

    im_mask = torch.zeros((img_h, img_w), dtype=torch.uint8)
    x_0 = max(box[0], 0)
    x_1 = min(box[2] + 1, img_w)
    y_0 = max(box[1], 0)
    y_1 = min(box[3] + 1, img_h)

    im_mask[y_0:y_1, x_0:x_1] = mask[
        (y_0 - box[1]) : (y_1 - box[1]), (x_0 - box[0]) : (x_1 - box[0])
    ]
    return im_mask


# Our pixel modeling requires extrapolation for any continuous
# coordinate < 0.5 or > length - 0.5. When sampling pixels on the masks,
# we would like this extrapolation to be an interpolation between boundary values and zero,
# instead of using absolute zero or boundary values.
# Therefore `paste_mask_in_image_old` is often used with zero padding around the masks like this:
# masks, scale = pad_masks(masks[:, 0, :, :], 1)
# boxes = scale_boxes(boxes.tensor, scale)


def pad_masks(masks, padding):
    """
    Args:
        masks (tensor): A tensor of shape (B, M, M) representing B masks.
        padding (int): Number of cells to pad on all sides.

    Returns:
        The padded masks and the scale factor of the padding size / original size.
    """
    B = masks.shape[0]
    M = masks.shape[-1]
    pad2 = 2 * padding
    scale = float(M + pad2) / M
    padded_masks = masks.new_zeros((B, M + pad2, M + pad2))
    padded_masks[:, padding:-padding, padding:-padding] = masks
    return padded_masks, scale


def scale_boxes(boxes, scale):
    """
    Args:
        boxes (tensor): A tensor of shape (B, 4) representing B boxes with 4
            coords representing the corners x0, y0, x1, y1,
        scale (float): The box scaling factor.

    Returns:
        Scaled boxes.
    """
    w_half = (boxes[:, 2] - boxes[:, 0]) * 0.5
    h_half = (boxes[:, 3] - boxes[:, 1]) * 0.5
    x_c = (boxes[:, 2] + boxes[:, 0]) * 0.5
    y_c = (boxes[:, 3] + boxes[:, 1]) * 0.5

    w_half *= scale
    h_half *= scale

    scaled_boxes = torch.zeros_like(boxes)
    scaled_boxes[:, 0] = x_c - w_half
    scaled_boxes[:, 2] = x_c + w_half
    scaled_boxes[:, 1] = y_c - h_half
    scaled_boxes[:, 3] = y_c + h_half
    return scaled_boxes
