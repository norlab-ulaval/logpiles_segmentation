#!/bin/bash
# This script downloads the TimberSeg 1.0 dataset. 

sudo apt-get install unzip
wget -O temp.zip https://data.mendeley.com/api/datasets-v2/datasets/y5npsm3gkj/zip/download?version=1
unzip temp.zip -d ./data
rm temp.zip
mv ./data/y5npsm3gkj*/* ./data
rm -r ./data/y5npsm3gkj*