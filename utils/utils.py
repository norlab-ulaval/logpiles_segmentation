from curses import meta
import json
from detectron2.structures.instances import Instances
import numpy as np
import cv2
import torch

from detectron2.data import detection_utils as utils
from detectron2.utils.visualizer import ColorMode, Visualizer

from tqdm import tqdm
from shapely.geometry import Polygon, box


def clip_rbbox(corners, img_shape):

    rbbox = Polygon(corners)
    image_rect = box(0, 0, img_shape[0], img_shape[1])

    clipped_rbbox = rbbox.intersection(image_rect)
    if clipped_rbbox.area < 0.3 * rbbox.area:
        return None
    else:
        return np.array(clipped_rbbox.exterior.coords)[:-1]


def visualize_annotations(dicts, metadata):
    # https://github.com/facebookresearch/detectron2/blob/master/tools/visualize_data.py
    for dic in dicts:
        img = utils.read_image(dic["file_name"], "BGR")
        print(dic["file_name"])
        visualizer = Visualizer(img[:, :, ::-1], metadata=metadata, scale=1)
        gt_image = visualizer.draw_dataset_dict(dic)

        image = gt_image.get_image()[:, :, ::-1]
        scale = 1000 / np.max(image.shape[:2])
        image = cv2.resize(image, (int(image.shape[1] * scale), int(image.shape[0] * scale)))
        cv2.imshow("Annotated image (ESC to quit)", image)
        k = cv2.waitKey(0)

        # exit loop if esc is pressed
        if k == 27:
            cv2.destroyAllWindows()
            return


def visualize_data_augmentation(train_loader, metadata, image_format="RGB"):
    for train_image_batch in train_loader:
        for train_image in train_image_batch:  
            # Pytorch tensor is in (C, H, W) format
            img = train_image["image"].permute(1, 2, 0).cpu().detach().numpy()
            img = utils.convert_image_to_rgb(img, image_format)

            visualizer = Visualizer(img, metadata=metadata, scale=1)
            target_fields = train_image["instances"].get_fields()
            labels = [metadata.thing_classes[i] for i in target_fields["gt_classes"]]
            vis = visualizer.overlay_instances(
                labels=labels,
                boxes=target_fields.get("gt_boxes", None),
                masks=target_fields.get("gt_masks", None),
                keypoints=target_fields.get("gt_keypoints", None),
            )  
               
            cv2.imshow("Augmented image (ESC to quit)", vis.get_image()[:, :, ::-1])
            k = cv2.waitKey(0)

            # exit loop if esc is pressed
            if k == 27:
                cv2.destroyAllWindows()
                return 


def visualize_predictions(predictor, dicts, metadata, threshold=0.8):
    for dic in tqdm(dicts):
        img = utils.read_image(dic["file_name"], "BGR")
        outputs = predictor(img)
        instances = filter_instances_with_score(outputs["instances"].to("cpu"), threshold)
        visualizer = Visualizer(img[:, :, ::-1], metadata=metadata, scale=1, instance_mode=ColorMode.SEGMENTATION)
        predictions = visualizer.draw_instance_predictions(instances)

        image = predictions.get_image()[:, :, ::-1]
        scale = 1000 / np.max(image.shape[:2])
        image = cv2.resize(image, (int(image.shape[1] * scale), int(image.shape[0] * scale)))
        cv2.imshow('Predictions (ESC to quit)', image)
        k = cv2.waitKey(0)

        # exit loop if esc is pressed
        if k == 27:
            cv2.destroyAllWindows()
            return


def filter_instances_with_score(instances, threshold):
    filt_inst = Instances(instances.image_size)
    idxs = np.argwhere(instances.scores > threshold)[0]
    filt_inst.pred_masks = instances.pred_masks[idxs] 
    filt_inst.pred_boxes = instances.pred_boxes[idxs] 
    filt_inst.scores = instances.scores[idxs]
    filt_inst.pred_classes = instances.pred_classes[idxs]
    return filt_inst


def filter_instances_with_area(instances, frac):
    filt_inst = Instances(instances.image_size)
    area = instances.image_size[0] * instances.image_size[1]
    idxs = np.argwhere(torch.sum(instances.pred_masks, [1,2]) > frac * area)[0]
    filt_inst.pred_masks = instances.pred_masks[idxs] 
    filt_inst.pred_boxes = instances.pred_boxes[idxs] 
    filt_inst.scores = instances.scores[idxs]
    filt_inst.pred_classes = instances.pred_classes[idxs]
    return filt_inst


def remove_overlap(instances, threshold):
    filt_inst = Instances(instances.image_size)
    masks = [mask for mask in instances.pred_masks]
    scores = [score for score in instances.scores]
    to_remove = []
    for i in range(len(masks)):
        mask_size = torch.sum(masks[i])
        for j in range(len(masks)):
            if i != j and scores[i] < scores[j] and mask_size > 0:
                intersection = torch.bitwise_and(masks[i].bool(), masks[j].bool())
                inter_size = torch.sum(intersection)
                overlap_frac = inter_size / mask_size
                if overlap_frac > threshold:
                    to_remove.append(i)

    if(len(to_remove) > 0):  
        print(len(set(to_remove)))
    idxs = np.delete(np.arange(len(masks)), to_remove)
    filt_inst.pred_masks = instances.pred_masks[idxs] 
    filt_inst.pred_boxes = instances.pred_boxes[idxs] 
    filt_inst.scores = instances.scores[idxs]
    filt_inst.pred_classes = instances.pred_classes[idxs]
    return filt_inst

     
def get_metadata_from_annos_file(annos_file):
    with open(annos_file, "r") as f:
        data = json.load(f)
        classes = [cat["name"] for cat in data["categories"]]
        metadata = {"thing_classes": classes}
    return metadata