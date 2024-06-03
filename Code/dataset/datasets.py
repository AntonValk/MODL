from typing import List
from torch.utils.data import Dataset, DataLoader, Subset
from glob import glob
import os
import pathlib
import pandas as pd
import numpy as np
import pytorch_lightning as pl

import torch.nn as nn
import torch.nn.functional as F
import torch

import bisect

import pickle

import subprocess
from glob import glob
import logging

import contextlib


logger = logging.getLogger("dataset")


@contextlib.contextmanager
def temp_seed(seed):
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


def collate_fn_flat_deal(batch):
    out = {}
    for b in batch:
        for k, bv in b.items():
            v = out.get(k, [])
            v.append(bv)
            out[k] = v
            
    for k,v in out.items():
        if k == 'Y':
            v = torch.as_tensor(v, dtype=torch.long)[0]
        else:
            v = torch.as_tensor(v, dtype=torch.float)
        out[k] = v
    return out

class GermanDataset(Dataset):
    def __init__(self, 
                 name: str, 
                 task_type: str,
                 aux_feat_prob: float,
                 use_cuda: bool,
                 seed: int):
        super().__init__()
        self.name = name
        self.type = task_type

        temp_seed(seed)
        
        data_path = os.path.join(os.path.dirname(__file__), 'Datasets', name, 'german.data-numeric')
        n_feat = 24
        n_aux_feat = 22
        n_base_feat = n_feat - n_aux_feat
        number_of_instances = 1000

        data_initial =  pd.read_csv(data_path, sep = "  " , header = None, engine = 'python')
        data_initial.iloc[np.array(data_initial[24].isnull()), 24] = 2.0
        data_shuffled = data_initial.sample(frac = 1) # Randomly shuffling the dataset
        self.label = np.array(data_shuffled[24] == 1)*1
        data = data_shuffled.iloc[: , :24]
        data.insert(0, column='class', value=self.label)
        for i in range(data.shape[0]):
                data.iloc[i,3] = int(data.iloc[i,3].split(" ")[1])
        
        if self.type == "trapezoidal":
                num_chunks = 10
                chunk_size = int(number_of_instances/10)
                self.aux_mask = np.zeros((number_of_instances, n_aux_feat))
                aux_feat_chunk_list = [round((n_feat/num_chunks)*i) - n_base_feat for i in range(1, num_chunks+1)]
                if aux_feat_chunk_list[0] == -1:
                        aux_feat_chunk_list[0] = 0
                aux_feat_chunk_list
                for i in range(num_chunks):
                        self.aux_mask[chunk_size*i:chunk_size*(i+1), :aux_feat_chunk_list[i]] = 1
        elif self.type == "variable_p":
                self.aux_mask = (np.random.random((number_of_instances, n_aux_feat)) < aux_feat_prob).astype(float)
        else:
                raise Exception(f"Please choose the type as \"variable_p\" for ", name, " dataset")

        # Data division
        self.n_base_feat = n_base_feat
        self.n_aux_feat = n_aux_feat
        self.n_base_feat = data.shape[1] - 1 - n_aux_feat
        self.Y = np.array(data.iloc[:,:1])
        self.X_base = np.array(data.iloc[:,1:n_base_feat+1])
        self.X_aux = np.array(data.iloc[:,n_base_feat+1:], dtype = float)
        self.X_aux_new = np.where(self.aux_mask, self.X_aux, 0)
        self.n_classes = 2
    
    def __getitem__(self, index):
        item = { 
            "X_base": self.X_base[index], 
            "X_aux_new": self.X_aux_new[index], 
            "aux_mask": self.aux_mask[index], 
            "Y": self.Y[index], 
            "label": self.label[index],
        }
        return item

    def __len__(self):
        return len(self.Y)

class GermanDataModule(pl.LightningDataModule):
    def __init__(self,
                 name: str = 'german',
                 task_type: str = "variable_p",
                 aux_feat_prob: float = 0.73,
                 num_workers: int = -1,
                 persistent_workers: bool = True,
                 use_cuda: bool = False,
                 batch_size: int = 1,
                 seed: int = 0
                ):
        super().__init__()
        self.name = name
        self.task_type = task_type
        self.aux_feat_prob = aux_feat_prob
        self.use_cuda = use_cuda
        self.num_workers = num_workers
        self.persistent_workers = persistent_workers
        self.batch_size = batch_size
        self.n_classes = 2
        self.seed = seed

    def setup(self, stage=None):
        self.dataset = GermanDataset(name=self.name, 
                                     task_type=self.task_type,
                                     aux_feat_prob=self.aux_feat_prob,
                                     use_cuda=self.use_cuda,
                                     seed=self.seed)
    
    def train_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, 
                          shuffle=False, pin_memory=True, 
                          persistent_workers=self.persistent_workers,
                          num_workers=self.num_workers, collate_fn=collate_fn_flat_deal,
                          multiprocessing_context='fork')


class SvmGuideDataset(Dataset):
    def __init__(self, 
                 name: str, 
                 task_type: str,
                 aux_feat_prob: float,
                 use_cuda: bool,
                 seed: int):
        super().__init__()
        self.name = name
        self.type = task_type
        self.aux_mask = None

        temp_seed(seed)
        
        data_path = os.path.join(os.path.dirname(__file__), 'Datasets', name, 'svmguide3.txt')
        n_feat = 21
        n_aux_feat = 19
        n_base_feat = n_feat - n_aux_feat
        number_of_instances = 1243

        # reading csv files
        # data_initial =  arff.loadarff(data_path)
        data_initial =  pd.read_csv(data_path, sep=" ", header=None)
        data_initial = data_initial.iloc[:, :22]
        for j in range(1, data_initial.shape[1]):
                for i in range(data_initial.shape[0]):
                        data_initial.iloc[i, j] = float(data_initial.iloc[i, j].split(":")[1])
        for i in range(data_initial.shape[0]):
                data_initial.iloc[i, 0] = (data_initial.iloc[i, 0] == -1)*1
        data = data_initial.sample(frac = 1)
        self.label = np.asarray(data[0])

        # Masking
        if self.type == "trapezoidal":
                num_chunks = 10
                chunk_size = int(number_of_instances/10)
                self.aux_mask = np.zeros((number_of_instances, n_aux_feat))
                aux_feat_chunk_list = [round((n_feat/num_chunks)*i) - n_base_feat for i in range(1, num_chunks+1)]
                if aux_feat_chunk_list[0] == -1:
                        aux_feat_chunk_list[0] = 0
                aux_feat_chunk_list
                for i in range(num_chunks):
                        self.aux_mask[chunk_size*i:chunk_size*(i+1), :aux_feat_chunk_list[i]] = 1
        elif self.type == "variable_p":
                self.aux_mask = (np.random.random((number_of_instances, n_aux_feat)) < aux_feat_prob).astype(float)
        else:
                raise Exception("Please choose the type as \"variable_p\" for ", name, " dataset")

        # Data division
        self.n_base_feat = n_base_feat
        self.n_aux_feat = n_aux_feat
        self.n_base_feat = data.shape[1] - 1 - n_aux_feat
        self.Y = np.array(data.iloc[:,:1])
        self.X_base = np.array(data.iloc[:,1:n_base_feat+1])
        self.X_aux = np.array(data.iloc[:,n_base_feat+1:], dtype = float)
        self.X_aux_new = np.where(self.aux_mask, self.X_aux, 0)
        self.n_classes = 2
    
    def __getitem__(self, index):
        item = { 
            "X_base": self.X_base[index], 
            "X_aux_new": self.X_aux_new[index], 
            "aux_mask": self.aux_mask[index], 
            "Y": self.Y[index], 
            "label": self.label[index],
        }
        return item

    def __len__(self):
        return len(self.Y)
    

class SvmGuideDataModule(pl.LightningDataModule):
    def __init__(self,
                 name: str = 'svmguide3',
                 task_type: str = "variable_p",
                 aux_feat_prob: float = 0.72,
                 num_workers: int = -1,
                 persistent_workers: bool = False,
                 use_cuda: bool = False,
                 batch_size: int = 1,
                 seed: int = 0
                ):
        super().__init__()
        self.name = name
        self.task_type = task_type
        self.aux_feat_prob = aux_feat_prob
        self.use_cuda = use_cuda
        self.num_workers = num_workers
        self.persistent_workers = persistent_workers
        self.batch_size = batch_size
        self.n_classes = 2
        self.seed = seed

    def setup(self, stage=None):
        self.dataset = SvmGuideDataset(name=self.name, 
                                     task_type=self.task_type,
                                     aux_feat_prob=self.aux_feat_prob,
                                     use_cuda=self.use_cuda,
                                     seed=self.seed)
    
    def train_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, 
                          shuffle=False, pin_memory=True, 
                          persistent_workers=self.persistent_workers,
                          num_workers=self.num_workers, collate_fn=collate_fn_flat_deal,
                          multiprocessing_context='fork')
    

class Magic04Dataset(Dataset):
    def __init__(self, 
                 name: str, 
                 task_type: str,
                 aux_feat_prob: float,
                 use_cuda: bool,
                 seed: int):
        super().__init__()
        self.name = name
        self.type = task_type
        self.mask = None

        temp_seed(seed)
        
        data_path = os.path.join(os.path.dirname(__file__), 'Datasets', name, 'magic04.data')
        n_feat = 10
        n_aux_feat = 8
        n_base_feat = n_feat - n_aux_feat
        number_of_instances = 19020

        # reading csv files
        data_initial =  pd.read_csv(data_path, sep=",", header=None)
        data_shuffled = data_initial.sample(frac = 1)
        self.label = np.array(data_shuffled[n_feat] == "g")*1
        data = data_shuffled.iloc[: , :n_feat]
        data.insert(0, column='class', value=self.label)

        # Masking
        if self.type == "trapezoidal":
                num_chunks = 10
                chunk_size = int(number_of_instances/10)
                self.aux_mask = np.zeros((number_of_instances, n_aux_feat))
                aux_feat_chunk_list = [round((n_feat/num_chunks)*i) - n_base_feat for i in range(1, num_chunks+1)]
                if aux_feat_chunk_list[0] == -1:
                        aux_feat_chunk_list[0] = 0
                aux_feat_chunk_list
                for i in range(num_chunks):
                        self.aux_mask[chunk_size*i:chunk_size*(i+1), :aux_feat_chunk_list[i]] = 1
        elif self.type == "variable_p":
                self.aux_mask = (np.random.random((number_of_instances, n_aux_feat)) < aux_feat_prob).astype(float)
        else:
                raise Exception("Please choose the type as \"variable_p\" for ", name, " dataset")

        # Data division
        self.n_base_feat = n_base_feat
        self.n_aux_feat = n_aux_feat
        self.n_base_feat = data.shape[1] - 1 - n_aux_feat
        self.Y = np.array(data.iloc[:,:1])
        self.X_base = np.array(data.iloc[:,1:n_base_feat+1])
        self.X_aux = np.array(data.iloc[:,n_base_feat+1:], dtype = float)
        self.X_aux_new = np.where(self.aux_mask, self.X_aux, 0)
        self.n_classes = 2
    
    def __getitem__(self, index):
        item = { 
            "X_base": self.X_base[index], 
            "X_aux_new": self.X_aux_new[index], 
            "aux_mask": self.aux_mask[index], 
            "Y": self.Y[index], 
            "label": self.label[index],
        }
        return item

    def __len__(self):
        return len(self.Y)


class Magic04DataModule(pl.LightningDataModule):
    def __init__(self,
                 name: str = 'magic04',
                 task_type: str = "variable_p",
                 aux_feat_prob: float = 0.68,
                 num_workers: int = -1,
                 persistent_workers: bool = True,
                 use_cuda: bool = False,
                 batch_size: int = 1,
                 seed: int = 0
                ):
        super().__init__()
        self.name = name
        self.task_type = task_type
        self.aux_feat_prob = aux_feat_prob
        self.use_cuda = use_cuda
        self.num_workers = num_workers
        self.persistent_workers = persistent_workers
        self.batch_size = batch_size
        self.n_classes = 2
        self.seed = seed

    def setup(self, stage=None):
        self.dataset = Magic04Dataset(name=self.name, 
                                     task_type=self.task_type,
                                     aux_feat_prob=self.aux_feat_prob,
                                     use_cuda=self.use_cuda,
                                     seed=self.seed)
    
    def train_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, 
                          shuffle=False, pin_memory=True, 
                          persistent_workers=self.persistent_workers,
                          num_workers=self.num_workers, collate_fn=collate_fn_flat_deal,
                          multiprocessing_context='fork')


class a8aDataset(Dataset):
    def __init__(self, 
                 name: str, 
                 task_type: str,
                 aux_feat_prob: float,
                 use_cuda: bool,
                 seed: int):
        super().__init__()
        self.name = name
        self.type = task_type
        self.mask = None

        temp_seed(seed)
        
        data_path = os.path.join(os.path.dirname(__file__), 'Datasets', name, 'a8a.txt')
        n_feat = 123
        n_aux_feat = 121
        n_base_feat = n_feat - n_aux_feat
        number_of_instances = 32561

        data = pd.DataFrame(0, index=range(number_of_instances), columns = list(range(1, n_feat+1)))
        # reading csv files
        data_initial =  pd.read_csv(data_path, sep=" ", header=None)
        data_initial = data_initial.iloc[:, :15]
        for j in range(data_initial.shape[0]):
                l = [int(i.split(":")[0])-1 for i in list(data_initial.iloc[j, 1:]) if not pd.isnull(i)]
                data.iloc[j, l] = 1
        label = np.array(data_initial[0] == -1)*1
        data.insert(0, column='class', value=label)
        data = data.sample(frac = 1)
        self.label = np.array(data["class"])

        # Masking
        if self.type == "trapezoidal":
                num_chunks = 10
                chunk_size = int(number_of_instances/10)
                self.aux_mask = np.zeros((number_of_instances, n_aux_feat))
                aux_feat_chunk_list = [round((n_feat/num_chunks)*i) - n_base_feat for i in range(1, num_chunks+1)]
                if aux_feat_chunk_list[0] == -1:
                        aux_feat_chunk_list[0] = 0
                aux_feat_chunk_list
                for i in range(num_chunks):
                        self.aux_mask[chunk_size*i:chunk_size*(i+1), :aux_feat_chunk_list[i]] = 1
        elif self.type == "variable_p":
                self.aux_mask = (np.random.random((number_of_instances, n_aux_feat)) < aux_feat_prob).astype(float)
        else:
                raise Exception("Please choose the type as \"variable_p\" for ", name, " dataset")

        # Data division
        self.n_base_feat = n_base_feat
        self.n_aux_feat = n_aux_feat
        self.n_base_feat = data.shape[1] - 1 - n_aux_feat
        self.Y = np.array(data.iloc[:,:1])
        self.X_base = np.array(data.iloc[:,1:n_base_feat+1])
        self.X_aux = np.array(data.iloc[:,n_base_feat+1:], dtype = float)
        self.X_aux_new = np.where(self.aux_mask, self.X_aux, 0)
        self.n_classes = 2
    
    def __getitem__(self, index):
        item = { 
            "X_base": self.X_base[index], 
            "X_aux_new": self.X_aux_new[index], 
            "aux_mask": self.aux_mask[index], 
            "Y": self.Y[index], 
            "label": self.label[index],
        }
        return item

    def __len__(self):
        return len(self.Y)


class a8aDataModule(pl.LightningDataModule):
    def __init__(self,
                 name: str = 'a8a',
                 task_type: str = "variable_p",
                 aux_feat_prob: float = 0.75,
                 num_workers: int = -1,
                 persistent_workers: bool = True,
                 use_cuda: bool = False,
                 batch_size: int = 1,
                 seed: int = 0
                ):
        super().__init__()
        self.name = name
        self.task_type = task_type
        self.aux_feat_prob = aux_feat_prob
        self.use_cuda = use_cuda
        self.num_workers = num_workers
        self.persistent_workers = persistent_workers
        self.batch_size = batch_size
        self.n_classes = 2
        self.seed = seed

    def setup(self, stage=None):
        self.dataset = a8aDataset(name=self.name, 
                                     task_type=self.task_type,
                                     aux_feat_prob=self.aux_feat_prob,
                                     use_cuda=self.use_cuda,
                                     seed=self.seed)
    
    def train_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, 
                          shuffle=False, pin_memory=True, 
                          persistent_workers=self.persistent_workers,
                          num_workers=self.num_workers, collate_fn=collate_fn_flat_deal,
                          multiprocessing_context='fork')
    
    
class HiggsDataset(Dataset):
    def __init__(self, 
                 name: str, 
                 task_type: str,
                 aux_feat_prob: float,
                 use_cuda: bool,
                 seed: int):
        super().__init__()
        self.name = name
        self.type = task_type
        self.aux_mask = None

        n_feat = 21
        n_aux_feat = 16
        n_base_feat = n_feat - n_aux_feat
        number_of_instances = 1000000 # 1M
        number_of_instances_name = "1M"
        Start = "50k"
        Gap = "50k"
        Stream = "200k"

        temp_seed(seed)
        
        data_path = os.path.join(os.path.dirname(__file__), 'Datasets', name, 'data', 'HIGGS_1M.csv.gz')
        # Load Data
        data = pd.read_csv(data_path, compression='gzip', nrows=number_of_instances)
        self.label = np.array(data["0"] == 1.0)*1

        # Masking
        if self.type == "variable_p":
                mask_file_name = name + "_" + number_of_instances_name +"_P_" + str(int(aux_feat_prob*100)) + "_AuxFeat_" + str(n_aux_feat) + ".data"
                mask_path = os.path.join(os.path.dirname(__file__), 'Datasets', name, 'mask', mask_file_name)
                with open(mask_path, 'rb') as file:
                        self.aux_mask = pickle.load(file)
        elif self.type == "obsolete_sudden":
                mask_file_name = name + "_" + number_of_instances_name + "_Start" + Start + "_Gap" + Gap + "_Stream" + Stream + "_AuxFeat_" + str(n_aux_feat) + ".data"
                mask_path = os.path.join(os.path.dirname(__file__), 'Datasets', name, 'mask', mask_file_name)
                with open(mask_path, 'rb') as file:
                        self.aux_mask = pickle.load(file)
        else:
                print("Please choose the type as \"variable_p\" or \"obsolete_sudden\" for ", name, " dataset")
                exit()

        # # Data division
        # Y = np.array(data.iloc[:,:1])
        # X_base = np.array(data.iloc[:,1:n_base_feat+1], dtype = float)
        # X_aux = np.array(data.iloc[:,n_base_feat+1:], dtype = float)
        # X_aux_new = np.where(self.aux_mask[:number_of_instances], X_aux, 0)

        # Data division
        self.n_base_feat = n_base_feat
        self.n_aux_feat = n_aux_feat
        self.n_base_feat = data.shape[1] - 1 - n_aux_feat
        self.Y = np.array(data.iloc[:,:1])
        self.X_base = np.array(data.iloc[:,1:n_base_feat+1])
        self.X_aux = np.array(data.iloc[:,n_base_feat+1:], dtype = float)
        self.X_aux_new = np.where(self.aux_mask, self.X_aux, 0)
        self.n_classes = 2
    
    def __getitem__(self, index):
        item = { 
            "X_base": self.X_base[index], 
            "X_aux_new": self.X_aux_new[index], 
            "aux_mask": self.aux_mask[index], 
            "Y": self.Y[index], 
            "label": self.label[index],
        }
        return item

    def __len__(self):
        return len(self.Y)


class HiggsDataModule(pl.LightningDataModule):
    def __init__(self,
                 name: str = 'HIGGS',
                 task_type: str = "variable_p",
                 aux_feat_prob: float = 0,
                 num_workers: int = 0,
                 persistent_workers: bool = True,
                 use_cuda: bool = False,
                 batch_size: int = 1,
                 seed: int = 0
                ):
        super().__init__()
        self.name = name
        self.task_type = task_type
        self.aux_feat_prob = aux_feat_prob
        self.use_cuda = use_cuda
        self.num_workers = num_workers
        self.persistent_workers = persistent_workers
        self.batch_size = batch_size
        self.n_classes = 2
        self.seed = seed

    def setup(self, stage=None):
        self.dataset = HiggsDataset(name=self.name, 
                                     task_type=self.task_type,
                                     aux_feat_prob=self.aux_feat_prob,
                                     use_cuda=self.use_cuda,
                                     seed=self.seed)
    
    def train_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, 
                          shuffle=False, pin_memory=True, 
                          persistent_workers=self.persistent_workers,
                          num_workers=self.num_workers, collate_fn=collate_fn_flat_deal,
                          multiprocessing_context='fork')
    

class SusyDataset(Dataset):
    def __init__(self, 
                 name: str, 
                 task_type: str,
                 aux_feat_prob: float,
                 use_cuda: bool,
                 seed: int):
        super().__init__()
        self.name = name
        self.type = task_type
        self.aux_mask = None

        data_path = os.path.join(os.path.dirname(__file__), 'Datasets', name, 'data', 'SUSY_1M.csv.gz')
        n_feat = 8
        n_aux_feat = 6
        n_base_feat = n_feat - n_aux_feat
        number_of_instances = 1000000 # 1M
        number_of_instances_name = "1M"
        Start = "100k"
        Gap = "100k"
        Stream = "400k"

        temp_seed(seed)
        
        # Load Data
        data = pd.read_csv(data_path, compression='gzip', nrows=number_of_instances)
        self.label = np.array(data["0"] == 1.0)*1

        # Masking
        if self.type == "variable_p":
                mask_file_name = name + "_" + number_of_instances_name +"_P_" + str(int(aux_feat_prob*100)) + "_AuxFeat_" + str(n_aux_feat) + ".data"
                mask_path = os.path.join(os.path.dirname(__file__), 'Datasets', name, 'mask', mask_file_name)
                with open(mask_path, 'rb') as file:
                        self.aux_mask = pickle.load(file)
        elif self.type == "obsolete_sudden":
                mask_file_name = name + "_" + number_of_instances_name + "_Start" + Start + "_Gap" + Gap + "_Stream" + Stream + "_AuxFeat_" + str(n_aux_feat) + ".data"
                mask_path = os.path.join(os.path.dirname(__file__), 'Datasets', name, 'mask', mask_file_name)
                with open(mask_path, 'rb') as file:
                        self.aux_mask = pickle.load(file)
        else:
                print("Please choose the type as \"variable_p\" or \"obsolete_sudden\" for ", name, " dataset")
                exit()

        # Data division
        self.n_base_feat = n_base_feat
        self.n_aux_feat = n_aux_feat
        self.n_base_feat = data.shape[1] - 1 - n_aux_feat
        self.Y = np.array(data.iloc[:,:1])
        self.X_base = np.array(data.iloc[:,1:n_base_feat+1])
        self.X_aux = np.array(data.iloc[:,n_base_feat+1:], dtype = float)
        self.X_aux_new = np.where(self.aux_mask, self.X_aux, 0)
        self.n_classes = 2
    
    def __getitem__(self, index):
        item = { 
            "X_base": self.X_base[index], 
            "X_aux_new": self.X_aux_new[index], 
            "aux_mask": self.aux_mask[index], 
            "Y": self.Y[index], 
            "label": self.label[index],
        }
        return item

    def __len__(self):
        return len(self.Y)

class SusyDataModule(pl.LightningDataModule):
    def __init__(self,
                 name: str = 'SUSY',
                 task_type: str = "variable_p",
                 aux_feat_prob: float = 0,
                 num_workers: int = -1,
                 persistent_workers: bool = True,
                 use_cuda: bool = False,
                 batch_size: int = 1,
                 seed: int = 0
                ):
        super().__init__()
        self.name = name
        self.task_type = task_type
        self.aux_feat_prob = aux_feat_prob
        self.use_cuda = use_cuda
        self.num_workers = num_workers
        self.persistent_workers = persistent_workers
        self.batch_size = batch_size
        self.n_classes = 2
        self.seed = seed

    def setup(self, stage=None):
        self.dataset = SusyDataset(name=self.name, 
                                     task_type=self.task_type,
                                     aux_feat_prob=self.aux_feat_prob,
                                     use_cuda=self.use_cuda,
                                     seed=self.seed)
    
    def train_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, 
                          shuffle=False, pin_memory=True, 
                          persistent_workers=self.persistent_workers,
                          num_workers=self.num_workers, collate_fn=collate_fn_flat_deal,
                          multiprocessing_context='fork')
