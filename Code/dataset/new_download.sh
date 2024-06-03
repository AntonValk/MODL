#!/bin/bash

# Make directories
mkdir -p Datasets/CD1/data
mkdir -p Datasets/CD1/mask
mkdir -p Datasets/CD2/data
mkdir -p Datasets/CD2/mask
mkdir -p Datasets/SYN8/data
mkdir -p Datasets/SYN8/mask
mkdir -p Datasets/HIGGS_5M/data
mkdir -p Datasets/HIGGS_5M/mask
mkdir -p Datasets/SUSY_5M/data
mkdir -p Datasets/SUSY_5M/mask

# Retrieve CD1 data
wget -c -N 'https://www.googleapis.com/drive/v3/files/1bNtxIgqu4kbcRLQQslNfmkMQdbBbT7nH/?key=AIzaSyCfFYmUZFxexTU8zdghOk9sQwwQfDGhQSo&alt=media' -O Datasets/CD1/data/CD1.npz 
unzip -u Datasets/CD1/data/CD1.npz -d Datasets/CD1/data/ && rm -rf Datasets/CD1/data/CD1.npz

# Retrieve CD2 data
wget -c -N 'https://www.googleapis.com/drive/v3/files/1N1fqzZnwKNoZFyEnAzBCLkotW2dHRBmj/?key=AIzaSyCfFYmUZFxexTU8zdghOk9sQwwQfDGhQSo&alt=media' -O Datasets/CD2/data/CD2.npz
unzip -u Datasets/CD2/data/CD2.npz -d Datasets/CD2/data/ && rm Datasets/CD2/data/CD2.npz

# Retrieve HIGGS data
wget -c -N 'https://www.googleapis.com/drive/v3/files/1zR_2p_yz5rMBHu8ZCLXFObfjwCljnQRq/?key=AIzaSyCfFYmUZFxexTU8zdghOk9sQwwQfDGhQSo&alt=media' -O Datasets/HIGGS_5M/data/HIGGS_5M.npz 
unzip -u Datasets/HIGGS_5M/data/HIGGS_5M.npz -d Datasets/HIGGS_5M/data/ && rm -rf Datasets/HIGGS_5M/data/HIGGS_5M.npz

# Retrieve SUSY data
wget -c -N 'https://www.googleapis.com/drive/v3/files/1JUhxQW335wJGfFRcIxJxkOSyF2WxXq5c/?key=AIzaSyCfFYmUZFxexTU8zdghOk9sQwwQfDGhQSo&alt=media' -O Datasets/SUSY_5M/data/SUSY_5M.npz 
unzip -u Datasets/SUSY_5M/data/SUSY_5M.npz -d Datasets/SUSY_5M/data/ && rm -rf Datasets/SUSY_5M/data/SUSY_5M.npz

# Retrieve SYN8 data
wget -c -N 'https://www.googleapis.com/drive/v3/files/1ae3lRCA9fx4l_52LHrRJnIJ6X-ZjQN76/?key=AIzaSyCfFYmUZFxexTU8zdghOk9sQwwQfDGhQSo&alt=media' -O Datasets/SUSY/data/SYN8.npz && unzip -u Datasets/SUSY/data/SYN8.npz -d Datasets/SYN8/data/ && rm -rf Datasets/SYN8/data/SYN8.npz
