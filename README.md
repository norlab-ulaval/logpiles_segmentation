# Instance Segmentation for Autonomous Log Grasping in Forestry Operations

Jean-Michel Fortin, Olivier Gamache, Vincent Grondin, François Pomerleau, Philippe Giguère

[[`arXiv`](https://arxiv.org/abs/2203.01902)] [[`BibTeX`](#CitingThisPaper)]

<div align="center">
  <img src="https://github.com/norlab-ulaval/logpiles_segmentation/blob/main/images/graphical_abstract.png" width="100%" height="100%"/>
</div><br/>

## Dataset 

The TimberSeg 1.0 dataset is publicly available [here](https://data.mendeley.com/datasets/y5npsm3gkj/). It comes with an original and a prescaled version of the images. We recommend using the prescaled version for faster dataloading and to avoid CUDA out-of-memory errors.

## Installation

### Requirements
- Linux or macOS with Python ≥ 3.6
- If using GPU, make sure you have at least 20 GB of memory and [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit) installed.
- We recommend that you first create a virtual env : 
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```
- PyTorch ≥ 1.9 and [torchvision](https://github.com/pytorch/vision/) that matches the PyTorch installation.
  Install them together at [pytorch.org](https://pytorch.org/get-started/locally/) to make sure of this. Make sure to select the correct CUDA version if using GPU.
- Then install project requirements 
  ```bash
  pip install -r requirements.txt
  ```

Detectron2 and Mask2Former were copied in this repository since we modified some files for rotated Mask R-CNN.

### Compile Detectron2 

```bash
python -m pip install -e detectron2
```

### CUDA kernel for MSDeformAttn (for Mask2Former)
After preparing the required environment, run the following command to compile CUDA kernel for MSDeformAttn:

`CUDA_HOME` must be defined and points to the directory of the installed CUDA toolkit.

```bash
cd mask2former/modeling/pixel_decoder/ops
sh make.sh
cd ../../../..
```

## Usage

This repo contains multiple scripts to reproduce our experiments. Parameters can be changed at the beginning of each file. 
Start by fetching the TimberSeg 1.0 dataset using the following commands :
```bash
sudo chmod u+x fetch_dataset.sh
./fetch_dataset.sh
```

Also, fetch the weights file for Mask2Former with the following commands :
```bash
sudo chmod u+x fetch_weights.sh
./fetch_weights.sh
```

### Model training 

Three instance segmentation networks are evaluated in the paper: Mask R-CNN, Rotated Mask R-CNN and Mask2Former. The following scripts lets you train and test each of them using our best configuration.

```bash
python3 standard_maskrcnn.py
python3 rotated_maskrcnn.py
python3 maskformer2.py
```

Training outputs will be generated in the ./outputs folder.

### Cross-validation training 

To run a cross-validation training, set the desired network architecture and the number of folds you wish in the script's parameters and run :

```bash
python3 kfold_train.py
```

### Inference

We provide a demo script for inference on a folder containing test images. You need to provide the correct output folder from a previous training in the script's parameters. 

```bash
python3 inference.py
```


## <a name="CitingThisPaper"></a>Citing This Paper

```bash
@article{fortin2022instance,
  title={Instance Segmentation for Autonomous Log Grasping in Forestry Operations},
  author={Fortin, Jean-Michel and Gamache, Olivier and Grondin, Vincent and Pomerleau, Fran{\c{c}}ois and Gigu{\`e}re, Philippe},
  journal={arXiv preprint arXiv:2203.01902},
  year={2022}
}
```