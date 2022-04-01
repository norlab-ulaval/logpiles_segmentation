from tkinter import Image
from albumentations.augmentations.functional import _maybe_process_in_chunks
from detectron2 import data
import numpy as np
import copy, cv2
import torch

from detectron2.structures import BoxMode
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T

import albumentations as A
from pycocotools.coco import maskUtils



class MaskRCNNMapper:

    def __init__(self, cfg, train, rotated=False, mask_on=True):

        self.rotated = rotated
        self.mask_on = mask_on

        img_size = cfg.INPUT.IMAGE_SIZE
        min_scale = cfg.INPUT.MIN_SCALE - 1
        max_scale = cfg.INPUT.MAX_SCALE - 1

        # Configure data augmentation -> https://albumentations.ai/docs/getting_started/transforms_and_targets/
        if train:
            self.transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.LongestMaxSize(max_size=img_size, interpolation=1, p=1),
                A.RandomScale((min_scale, max_scale), interpolation=cv2.INTER_LINEAR, p=1.0),
                A.PadIfNeeded(min_height=img_size, min_width=img_size, position=A.PadIfNeeded.PositionType.TOP_LEFT, border_mode=cv2.BORDER_CONSTANT, value=(128,128,128), p=1.0),
                A.Crop(x_min=0, x_max=img_size, y_min=0, y_max=img_size, p=1.0),
            ], bbox_params=A.BboxParams(format='coco', label_fields=["category_id", "bbox_ids"], min_visibility=0.1)
            if not rotated else None
            )
        else:
            self.transform = A.Compose([
                A.SmallestMaxSize(max_size=img_size, interpolation=1, always_apply=False, p=1),
            ], bbox_params=A.BboxParams(format='coco', label_fields=["category_id", "bbox_ids"], min_visibility=0.1)
            if not rotated else None
            )


    def __call__(self, dataset_dict):

        dataset_dict = generic_albu_mapper(dataset_dict, 
                                        self.transform, 
                                        bbox_on=True, 
                                        mask_on=self.mask_on, 
                                        keypoints_on=False, 
                                        bbox_mode=BoxMode.XYWHA_ABS if self.rotated else BoxMode.XYWH_ABS,
                                        mask_type="polygon")

        return dataset_dict


def generic_albu_mapper(dataset_dict, transform, bbox_on, mask_on, keypoints_on, bbox_mode=BoxMode.XYWH_ABS, mask_type="polygon"):

    dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below

    image = utils.read_image(dataset_dict["file_name"], format="BGR")
    labels = [ann['category_id'] for ann in dataset_dict['annotations']]

    rbbox_on = (bbox_on and bbox_mode == BoxMode.XYWHA_ABS)

    if bbox_on and bbox_mode != BoxMode.XYWHA_ABS:
        bboxes = [ann["bbox"] for ann in dataset_dict['annotations']]
    
    if keypoints_on:
        keypoints = np.array([ann['keypoints'] for ann in dataset_dict['annotations']]).reshape((-1, 3))    # X, Y, visibility

    if (mask_on or rbbox_on) and mask_type == "bitmask":
        masks = [maskUtils.decode(ann['segmentation']) for ann in dataset_dict['annotations']]
    elif (mask_on or rbbox_on) and mask_type == "polygon":
        # Convert polygons to bitmasks for albumentations
        segmentations = [ann['segmentation'] for ann in dataset_dict['annotations']]
        masks = []
        for polygons in segmentations:
            for i in range(len(polygons)):
                polygons[i] = np.reshape(polygons[i], (-1, 2)).astype(np.int32)
            mask = np.zeros(image.shape[:2])
            cv2.fillPoly(mask, pts=polygons, color=255)
            masks.append(mask)
            
    
    args = {
        "image":image, 
        "category_id": labels,
        "masks": masks if (mask_on or rbbox_on) else None,
        "bboxes": bboxes if bbox_on and not rbbox_on else None,
        "keypoints": keypoints if keypoints_on else None, 
        "bbox_ids": np.arange(len(labels)) if bbox_on and not rbbox_on else None
    }
    args = {k: v for k, v in args.items() if v is not None}
    transformed = transform(**args)

    transformed_image = transformed["image"]
    transformed_labels = np.array(transformed['category_id'])
    transformed_bboxes = []
    transformed_masks = []
    transformed_keypoints = []

    if rbbox_on:
        visible_ids = np.arange(len(transformed_labels))        # Keep all masks
    else:
        visible_ids = transformed['bbox_ids']

    if mask_on or rbbox_on:
        transformed_masks = np.array(transformed["masks"], dtype=np.uint8)[visible_ids]

    if rbbox_on:
        # find rotated bounding boxes from transformed masks
        to_remove = []
        for i in range(len(transformed_masks)):
            contours, _ = cv2.findContours(transformed_masks[i], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours = [point for contour in contours for point in contour]
            if len(contours) > 0:
                rotated_rect = np.array(cv2.minAreaRect(np.float32(contours)), dtype=object)
                if (rotated_rect[1][0] > rotated_rect[1][1]):
                    rotated_rect[2] -= 90
                rotated_rect[1] = sorted(rotated_rect[1])
                transformed_bboxes.append([rotated_rect[0][0], rotated_rect[0][1], rotated_rect[1][0]*1.2, rotated_rect[1][1]*1.05, -rotated_rect[2]])
            else:
                to_remove.append(i)
        transformed_labels = np.delete(transformed_labels, to_remove)   # Remove labels linked to invisible masks/boxes
        transformed_masks = np.delete(transformed_masks, to_remove, axis=0)
    elif bbox_on and mask_on:
        for i in range(len(transformed_masks)):
            rect = cv2.boundingRect(transformed_masks[i])
            transformed_bboxes.append(list(rect))
    elif bbox_on:
        transformed_bboxes = transformed["bboxes"]
    
    if keypoints_on:
        transformed_keypoints = np.array(transformed["keypoints"])[visible_ids]

    dataset_dict["image"] = torch.as_tensor(transformed_image.transpose(2, 0, 1).astype("float32"))
    
    annos = []
    for i in range(len(transformed_labels)):
        anno = {'iscrowd': 0, 'category_id': transformed_labels[i], 'bbox_mode': bbox_mode}
        if bbox_on or rbbox_on:
            anno['bbox'] = transformed_bboxes[i]
        if mask_on:
            anno['segmentation'] = maskUtils.encode(np.asfortranarray(transformed_masks[i]))
        if keypoints_on:
            anno['keypoints'] = transformed_keypoints[i].tolist()
        annos.append(anno)

    dataset_dict['annotations'] = annos

    instances = None
    if rbbox_on:
        instances = utils.annotations_to_instances_rotated(annos, image.shape[:2], mask_format="bitmask")
    else: 
        instances = utils.annotations_to_instances(annos, image.shape[:2], mask_format="bitmask")

    dataset_dict["instances"] = utils.filter_empty_instances(instances)

    return dataset_dict
