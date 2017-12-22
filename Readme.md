
# Overview

# ISIC

## ISIC datasets

http://isic-archive.com/

Currently using:
- ISIC_MSK-2_1

identified as possibly usable:
- ISIC_MSK-1_1
- ISIC_MSK-1_2
- ISIC_MSK-2_1
- ISIC_MSK-4_1
- ISIC_UDA-1_1
- ISIC_UDA-2_1

metadata and information: view **ISIC/csv_metadata/**


## Running it all

To run the scripts and notebooks, follow these steps:
1. read and execute ISIC/notebooks_metadata_and_images/get_3_images.ipynb
1. sort files by e.g. using script ISIC/dataordering.py or split with ISIC/0_datasplitting.ipynb
1. update ISIC/preparing.sh
1. run notebooks in (numbered) order



## Working with a Spot Instance

After setting up a spot instance, copy the IP, paste it in a new tab in your browser, log in (deep_learning) and do the following:

1. click New -> Terminal
2. Copy, paste and execute the following lines:

```bash
wget https://raw.githubusercontent.com/linoba/melanoma-classification/master/ISIC/preparing.sh
chmod +x preparing.sh
./preparing.sh
pwd
```

Continue with: 

3. Copy the path that appeared after last line
4. Open the notebook you want to run
5. Make sure that the path to your datafolder is set to the path you copied above with the suffix "/data/"
6. Run your tests

## Weight files

Our best result with file 4_10.h5 can be found here: www.dominikmorgen.de/data/dl/4_110epochs.zip


# Cats_and_dogs

Code taken from tutorial.
