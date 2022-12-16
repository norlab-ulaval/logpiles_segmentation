#!/bin/bash
# This script downloads the TimberSeg 1.0 dataset. 

sudo apt-get install unzip
wget -O temp.zip https://prod-dcd-datasets-cache-zipfiles.s3.eu-west-1.amazonaws.com/y5npsm3gkj-2.zip
unzip temp.zip -d ./data
rm temp.zip
mv ./data/y5npsm3gkj*/* ./data
rm -r ./data/y5npsm3gkj*