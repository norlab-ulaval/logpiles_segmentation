#!/bin/bash
# This script downloads the TimberSeg 1.0 dataset. 

sudo apt-get install unzip
wget -O mask2former/weights/swin_base_patch4_window12_384_22k.pkl https://dl.fbaipublicfiles.com/maskformer/mask2former/coco/instance/maskformer2_swin_base_IN21k_384_bs16_50ep/model_final_83d103.pkl