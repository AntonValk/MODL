#!/bin/bash

# Make directories
mkdir -p Datasets/SUSY/data
mkdir -p Datasets/SUSY/mask
mkdir -p Datasets/HIGGS/data
mkdir -p Datasets/HIGGS/mask

# Retrieve HIGGS data
wget -c -N https://figshare.com/ndownloader/files/40323670?private_link=0cd0d6ad4d30a9e91e9a -O Datasets/HIGGS/data/HIGGS_1M.csv.gz
wget -c -N https://figshare.com/ndownloader/articles/22705321?private_link=644fe204eb591e104184 -O Datasets/HIGGS/mask/temp.gz
unzip -u Datasets/HIGGS/mask/temp.gz -d Datasets/HIGGS/mask/

# Retrieve SUSY data
wget -c -N https://figshare.com/ndownloader/files/40323727?private_link=f4098ce6635f702c89b2 -O Datasets/SUSY/data/SUSY_1M.csv.gz
wget -c -N https://figshare.com/ndownloader/articles/22705330?private_link=87330bbbbc31b15d44e5 -O Datasets/SUSY/mask/temp.gz
unzip -u Datasets/SUSY/mask/temp.gz -d Datasets/SUSY/mask/
