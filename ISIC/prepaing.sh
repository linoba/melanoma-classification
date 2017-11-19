#!/bin/sh

echo "Preparing for execution."

git clone https://github.com/linoba/melanoma-classification

echo "Downloading images."
wget "x/data/ISIC_MSK-2_1_sorted.zip"
echo "Downloading done."

unzip ISIC_MSK-2_1_sorted.zip

mv ISIC_MSK-2_1_sorted data

echo "Preparing done."

