import datetime
from random import random
import torch
import os
import numpy as np

from sklearn.model_selection import KFold
from detectron2.data.catalog import DatasetCatalog 
from detectron2.data.datasets.coco import load_coco_json
from detectron2.utils import logger

from maskformer2 import init_mask2former, train_mask2former
from standard_maskrcnn import init_maskrcnn, train_maskrcnn
from rotated_maskrcnn import init_rotated_maskrcnn, train_rotated_maskrcnn

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

#########################
### PROGRAM VARIABLES ###
#########################

MODEL_NAME = "maskrcnn"      # "maskrcnn", "maskrcnn-rotated", "mask2former"
N_FOLDS = 5                  # Number of folds for cross-validation

DATASET_FILENAMES = {
    "mask2former": 'data/prescaled/coco_annotation.json',
    "maskrcnn": 'data/prescaled/coco_annotation.json',
    "maskrcnn-rotated": 'data/prescaled/coco_annotation_rotated.json'
                    }
CONFIG_FILES = {
    "mask2former": "configs/config_mask2former_swinB.yaml", 
    "maskrcnn": "configs/config_standard_maskrcnn.yaml", 
    "maskrcnn-rotated": "configs/config_rotated_maskrcnn.yaml"
               }

IMAGE_DIR = 'data/prescaled'           # Folder where data is located
INITIAL_WEIGHTS = None                 # Path to a previous checkpoint to finetune or None

#########################


def kfold_train(n_folds, model_name, dataset_dicts, config_file):

    # Init output folder 
    base_output_dir = f'./outputs/{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")}'
    
    random_state = np.random.RandomState(42)
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)

    fold_num = 0

    for train_index, test_index in kf.split(dataset_dicts):     # For each fold

        # Add fold index to output_dir
        output_dir = os.path.join(base_output_dir, f"fold_{fold_num}")

        train_dicts = dataset_dicts[train_index]
        test_dicts = dataset_dicts[test_index]

        if model_name == "mask2former":
            cfg = init_mask2former(config_file, train_dicts, test_dicts, output_dir, fold_num)
            train_mask2former(cfg)
        elif model_name == "maskrcnn":
            cfg = init_maskrcnn(config_file, train_dicts, test_dicts, output_dir, fold_num)
            train_maskrcnn(cfg)
        elif model_name == "maskrcnn-rotated":
            cfg = init_rotated_maskrcnn(config_file, train_dicts, test_dicts, output_dir, fold_num)
            train_rotated_maskrcnn(cfg)

        fold_num += 1
        DatasetCatalog.remove(cfg.DATASETS.TRAIN[0])
        DatasetCatalog.remove(cfg.DATASETS.TEST[0])


if __name__ == "__main__":

    print('GPU available :', torch.cuda.is_available())
    print('Torch version :', torch.__version__, '\n')
    logger.setup_logger(name=__name__)

    dataset_dicts = np.array(load_coco_json(DATASET_FILENAMES[MODEL_NAME], IMAGE_DIR))
    kfold_train(N_FOLDS, MODEL_NAME, dataset_dicts, CONFIG_FILES[MODEL_NAME])
