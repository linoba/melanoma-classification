#!/bin/sh

echo "Preparing for execution."

echo "Downloading repository."
git clone https://github.com/linoba/melanoma-classification
echo "Downloading done."

echo "Downloading images."
wget "www.dominikmorgen.de/data/ISIC_MSK-2_1_sorted.zip" # CHANGEME
echo "Downloading done."

unzip ISIC_MSK-2_1_sorted.zip

mv ISIC_MSK-2_1_sorted data

wget "https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels.h5"

cd melanoma-classification 
git config core.editor "vim"

echo "Preparing done."

