#!/bin/sh

echo "Preparing for execution."

echo "Downloading repository."
git clone https://github.com/linoba/melanoma-classification
echo "Downloading done."

echo "Downloading images."
wget "www.dominikmorgen.de/data/ISIC_MSK-2_1_sorted.zip"
echo "Downloading done."

unzip ISIC_MSK-2_1_sorted.zip

mv ISIC_MSK-2_1_sorted data

echo "Preparing done."

