README


# ISIC


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
