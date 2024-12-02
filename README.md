# MODL
MODL: Multilearner Online Deep Learning

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
python run_hydra.py --config=configssoe_proto_german.yaml
python run_hydra.py --config=configssoe_proto_svm.yaml
python run_hydra.py --config=configssoe_proto_magic.yaml
python run_hydra.py --config=configssoe_proto_a8a.yaml
python run_hydra.py --config=configssoe_proto_higgs.yaml
python run_hydra.py --config=configssoe_proto_susy.yaml
python run_hydra.py --config=configssoe_proto_imnist.yaml
python run_hydra.py --config=configssoe_proto_cifar.yaml
```

The results can be viewed tensorboard logging of appropriate ./lightning_logs/<dataset_name> directory
