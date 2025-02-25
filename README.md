# MODL
MODL: Multilearner Online Deep Learning (Accepted AISTATS 2025)

### Citation
If you use MODL or the code base in any way (such as running experiments) please cite paper: 
```
@inproceedings{
valkanas2025modl,
title={{MODL}: Multilearner Online Deep Learning},
author={Antonios Valkanas and Boris N. Oreshkin and Mark Coates},
booktitle={The 28th International Conference on Artificial Intelligence and Statistics},
year={2025},
url={https://openreview.net/forum?id=WspiEX6v3r}
}
```

# Using this repository

## Create workspace and clone this repository

```mkdir workspace```

```cd workspace```

```git clone [ANOYNYMOUS LINK]```

## Build docker image and launch container 
```
docker build -f Dockerfile -t MODL:$USER .
nvidia-docker run -p 8888:8888 -p 6006:6006 -v ~/workspace/MODL:/workspace/MODL -t -d --shm-size="1g" --name MODL_$USER MODL:$USER
```

## Enter docker container and launch training session

```
docker exec -i -t MODL$USER  /bin/bash 
```
Once inside docker container, this launches the training session for the proposed model. Checkpoints and tensorboard logs are stored in ./lightning_logs/

But first, download the data. Note that cifar-10 is available from pytorch and imnist is generated from original source available at https://leon.bottou.org/projects/infimnist
```
./download.sh
```
To obtain the reported results for MODL run the following commands (each one command corresponds to one dataset)
```
python run_hydra.py --config=configs/soe_proto_german.yaml
python run_hydra.py --config=configs/soe_proto_svm.yaml
python run_hydra.py --config=configs/soe_proto_magic.yaml
python run_hydra.py --config=configs/soe_proto_a8a.yaml
python run_hydra.py --config=configs/soe_proto_higgs.yaml
python run_hydra.py --config=configs/soe_proto_susy.yaml
python run_hydra.py --config=configs/soe_proto_imnist.yaml
python run_hydra.py --config=configs/soe_proto_cifar.yaml
```

The results can be viewed on tensorboard by selecting appropriate ./lightning_logs/<dataset_name> directory

```
tensorboard --logdir ./lightning_logs/<dataset_name>
```
