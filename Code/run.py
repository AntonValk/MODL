# All libraries
import os
import random
import collections
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import pandas as pd
from tqdm import tqdm
from AuxDrop import AuxDrop_ODL, AuxDrop_OGD, AuxDrop_ODL_AuxLayer1stlayer
from AuxDrop import (
    AuxDrop_ODL_DirectedInAuxLayer_RandomOtherLayer,
    AuxDrop_ODL_RandomAllLayer,
)
from AuxDrop import (
    AuxDrop_ODL_RandomInAuxLayer,
    AuxDrop_ODL_RandomInFirstLayer_AllFeatToFirst,
)
from datasets2 import dataset
from joblib import Parallel, delayed
# from modules.residual import SingleStageResidualNet, SingleStageResidualNetODL, Fast_AuxDrop_ODL, SetSingleStageResidualNet, ODLSetSingleStageResidualNet
from modules.residual import SingleStageResidualNet, SingleStageResidualNetODL, SetSingleStageResidualNet, ODLSetSingleStageResidualNet
from modules.old_residual import Fast_AuxDrop_ODL

from torch.utils.tensorboard import SummaryWriter

# Data description
# "german", "svmguide3", "magic04", "a8a", "ItalyPowerDemand", "SUSY", "HIGGS"
data_name = "HIGGS"

# Choose the type of data unavailability
# type can be - "variable_p", "trapezoidal", "obsolete_sudden"
type = "variable_p"

# Choose a model to run
# "AuxDrop_ODL" - Aux-Drop applied on ODL framework
#  "AuxDrop_OGD" - Aux-Drop applied on OGD framework
# "AuxDrop_ODL_DirectedInAuxLayer_RandomOtherLayer" -  On ODL framework, Aux-Dropout in AuxLayer and Random dropout in all the other layers
# "AuxDrop_ODL_RandomAllLayer" - On ODL framework, Random Dropout applied in all the layers
#  "AuxDrop_ODL_RandomInAuxLayer" - On ODL framework, Random Dropout applied in the AuxLayer
# "AuxDrop_ODL_RandomInFirstLayer_AllFeatToFirst" - On ODL framework, Random Dropout applied in the first layer and all the features (base + auxiliary) are passed to the first layer

# model_to_run = "AuxDrop_ODL"
# model_to_run = "AuxDrop_OGD"
# model_to_run = "ResidualSingleStage"
# model_to_run = "ResidualSingleStage_ODL"
model_to_run = "Fast_AuxDrop_ODL"
# model_to_run = "SetSingleStageResidualNet"
# model_to_run = "ODLSetSingleStageResidualNet"

# Values to change
n = 0.05
aux_feat_prob = 0.7
dropout_p = 0.3
max_num_hidden_layers = 11
qtd_neuron_per_hidden_layer = 50
n_classes = 2
aux_layer = 3
n_neuron_aux_layer = 100
batch_size = 1
b = 0.99
s = 0.2
use_cuda = False
number_of_experiments = 5


n_base_feat, n_aux_feat, X_base, X_aux, X_aux_new, aux_mask, Y, label = dataset(
        data_name, type=type, aux_feat_prob=aux_feat_prob, use_cuda=use_cuda, seed=0
    )

y=Y
X = np.concatenate((X_base, X_aux_new), axis=1)

print(X.shape)
print(y.shape)
print(X.mean(axis=1))
print(y[0])
exit()

# "CD1", "CD2", "HIGGS_5M", "SUSY_5M", "SYN8"
data_name = "SUSY_5M"
X = np.load(f"dataset/Datasets/{data_name}/data/x_train.npy")
y = np.load(f"dataset/Datasets/{data_name}/data/y_train.npy")

y = np.argmax(y, axis=1)

# split X and y into training and testing sets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

lr = []
tree = []
nn = []
for split in [0.1, 0.2, 0.5, 0.8, 0.9]:
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split, random_state=16)
    X_train, X_test, y_train, y_test = X[:int(split*len(X))], X[int(split*len(X)):], y[:int(split*len(X))], y[int(split*len(X)):]
    logreg = LogisticRegression(random_state=16, max_iter=1000)
    logreg.fit(X_train, y_train)
    y_pred = logreg.predict(X_test)
    # import the metrics class
    from sklearn import metrics
    
    cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
    from sklearn.metrics import classification_report
    target_names = ['0', '1']
    # print(classification_report(y_test, y_pred, target_names=target_names))
    # print('norm. cumulative errors:', 5*(cnf_matrix[0,1]+cnf_matrix[1,0]))
    lr.append(metrics.accuracy_score(y_test, y_pred))
    
    from catboost import CatBoostClassifier
    model = CatBoostClassifier(
        iterations=5,
        learning_rate=0.1,
        min_data_in_leaf=20,
        depth=15, 
        grow_policy= "Depthwise"
        # loss_function='CrossEntropy'
    )
    model.fit(
        X_train, y_train,
        cat_features=None,
        verbose=False
    )
    # print('Model is fitted: ' + str(model.is_fitted()))
    # print('Model params:')
    # print('catboost')
    y_pred = model.predict(X_test)
    # import the metrics class
    cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
    # print(cnf_matrix)
    target_names = ['0', '1']
    # print(classification_report(y_test, y_pred, target_names=target_names))
    # print('norm. cumulative errors:', 5*(cnf_matrix[0,1]+cnf_matrix[1,0]))
    tree.append(metrics.accuracy_score(y_test, y_pred))

    from sklearn.neural_network import MLPClassifier
    # print("nn")
    clf = MLPClassifier(solver='sgd', alpha=1e-4, hidden_layer_sizes=(100, 100, 100), random_state=1, batch_size=100, verbose=True, max_iter=2, learning_rate_init=0.001)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
    # print(cnf_matrix)
    target_names = ['0', '1']
    # print(classification_report(y_test, y_pred, target_names=target_names))
    # print('norm. cumulative errors:', 5*(cnf_matrix[0,1]+cnf_matrix[1,0]))
    nn.append(metrics.accuracy_score(y_test, y_pred))
    
df = pd.DataFrame(
    {'train split': [0.1, 0.2, 0.5, 0.8, 0.9],
     'logistic regression': lr,
     'cat boost': tree,
     'mlp': nn
    })
df.to_csv(f"{data_name}_test.csv")
print(df)
exit()



error_list = []
loss_list = []
# for ex in range(number_of_experiments):
def run_trial(ex):
    # default `log_dir` is "runs" - we'll be more specific here
    writer = SummaryWriter(log_dir=f"main/{data_name}/MODEL-{model_to_run}-SEED-{ex}-LR-{str(n)}-L-{max_num_hidden_layers}-EMB_DIM-{qtd_neuron_per_hidden_layer}-p-{aux_feat_prob}")
    trial_stats = 0
    print("Experiment number ", ex + 1)
    seed = ex

    # Please change the value of hyperparameters in the dataset.py file corresponding to the chose data name
    n_base_feat, n_aux_feat, X_base, X_aux, X_aux_new, aux_mask, Y, label = dataset(
        data_name, type=type, aux_feat_prob=aux_feat_prob, use_cuda=use_cuda, seed=seed
    )
    # print(n_base_feat, n_aux_feat)
    # exit()
    # Note: X_aux_new contains the auxiliary data with some data unavailable.
    # X_aux contains the auxiliary features with all the data (even the unavailable ones)

    model = None
    if model_to_run == "AuxDrop_ODL":
        if aux_layer == 1:
            model = AuxDrop_ODL_AuxLayer1stlayer(
                features_size=n_base_feat,
                max_num_hidden_layers=max_num_hidden_layers,
                qtd_neuron_per_hidden_layer=qtd_neuron_per_hidden_layer,
                n_classes=n_classes,
                aux_layer=aux_layer,
                n_neuron_aux_layer=n_neuron_aux_layer,
                batch_size=batch_size,
                b=b,
                n=n,
                s=s,
                dropout_p=dropout_p,
                n_aux_feat=n_aux_feat,
                use_cuda=use_cuda,
            )
        else:
            # Creating the Aux-Drop(ODL) Model
            model = AuxDrop_ODL(
                features_size=n_base_feat,
                max_num_hidden_layers=max_num_hidden_layers,
                qtd_neuron_per_hidden_layer=qtd_neuron_per_hidden_layer,
                n_classes=n_classes,
                aux_layer=aux_layer,
                n_neuron_aux_layer=n_neuron_aux_layer,
                batch_size=batch_size,
                b=b,
                n=n,
                s=s,
                dropout_p=dropout_p,
                n_aux_feat=n_aux_feat,
                use_cuda=use_cuda,
            )
    elif model_to_run == "Fast_AuxDrop_ODL":
        model = Fast_AuxDrop_ODL(
                    features_size=n_base_feat,
                    max_num_hidden_layers=max_num_hidden_layers,
                    qtd_neuron_per_hidden_layer=qtd_neuron_per_hidden_layer,
                    n_classes=n_classes,
                    aux_layer=aux_layer,
                    n_neuron_aux_layer=n_neuron_aux_layer,
                    batch_size=batch_size,
                    b=b,
                    n=n,
                    s=s,
                    dropout_p=dropout_p,
                    n_aux_feat=n_aux_feat,
                    use_cuda=use_cuda,
                )
    elif model_to_run == "AuxDrop_OGD":
        if data_name in ["ItalyPowerDemand"]:
            print(
                "You need to make some changes in the code to support AuxDrop_OGD with ",
                data_name,
                " dataset",
            )
            exit()
        # Creating the Aux-Drop(OGD) use this - The position of AuxLayer cannot be 1 here
        if aux_layer == 1:
            print("Error: Please choose the aux layer position greater than 1")
            exit()
        else:
            model = AuxDrop_OGD(
                features_size=n_base_feat,
                max_num_hidden_layers=max_num_hidden_layers,
                qtd_neuron_per_hidden_layer=qtd_neuron_per_hidden_layer,
                n_classes=n_classes,
                aux_layer=aux_layer,
                n_neuron_aux_layer=n_neuron_aux_layer,
                batch_size=batch_size,
                n_aux_feat=n_aux_feat,
                n=n,
                dropout_p=dropout_p,
            )
    elif model_to_run == "ResidualSingleStage":
        model = SingleStageResidualNet(
                num_blocks_enc=2,
                num_layers_enc=2,
                layer_width_enc=100,
                num_blocks_stage=2, 
                num_layers_stage=2, 
                layer_width_stage=100,
                embedding_dim=0,
                embedding_num=0,
                embedding_size=0,
                size_in=n_base_feat + n_aux_feat,
                size_out=n_classes,
                dropout=dropout_p,
                lr=n,
            )
        
    elif model_to_run == "SetSingleStageResidualNet":
        model = SetSingleStageResidualNet(
                num_blocks_enc=2,
                num_layers_enc=2,
                layer_width_enc=24,
                num_blocks_stage=2, 
                num_layers_stage=2, 
                layer_width_stage=100,
                size_in=1,
                size_out=n_classes,
                dropout=dropout_p,
                lr=n,
            )
        
    elif model_to_run == "ODLSetSingleStageResidualNet":
        model = ODLSetSingleStageResidualNet(
                num_blocks_enc=2,
                num_layers_enc=2,
                layer_width_enc=100,
                num_blocks_stage=2, 
                num_layers_stage=2, 
                layer_width_stage=100,
                size_in=1,
                size_out=n_classes,
                dropout=dropout_p,
                lr=n,
            )
        
    elif model_to_run == "ResidualSingleStage_ODL":
        model = SingleStageResidualNetODL(
                num_blocks_enc=2,
                num_layers_enc=2,
                layer_width_enc=100,
                num_blocks_stage=2, 
                num_layers_stage=2, 
                layer_width_stage=100,
                embedding_dim=0,
                embedding_num=0,
                embedding_size=0,
                size_in=n_base_feat + n_aux_feat,
                size_out=n_classes,
                dropout=dropout_p,
                lr=n,
            )
        

    # Run the model
    N = X_base.shape[0]
    cumulative_error_train = np.array([0])
    cumulative_error_test = np.array([0])
    exp_smoothing = 0.05
    prev_train = prev_test = 0
    for i in tqdm(range(N)):
        model.partial_fit(X_base[i].reshape(1, n_base_feat), X_aux_new[i].reshape(1, n_aux_feat), aux_mask[i].reshape(1, n_aux_feat), Y[i].reshape(1))
        pred = model.prediction[-1]
        cumulative_error_test += torch.argmax(pred).item() != Y[i]
        writer.add_scalar('test/cumulative_error', cumulative_error_test, i)
        # writer.add_scalar('test/exp_smooth_error', exp_smoothing * (torch.argmax(pred).item() != Y[i]) + (1 - exp_smoothing) * prev_test, i)
        # prev_test = exp_smoothing * (torch.argmax(pred).item() != Y[i]) + (1 - exp_smoothing) * prev_test
        writer.add_scalar('test/norm_error', cumulative_error_test/i, i)
        test_loss = model.loss_fn(pred, torch.tensor(Y[i], dtype=torch.long))
        writer.add_scalar('test/test loss', test_loss, i)
        
        with torch.no_grad():
            if hasattr(model, 'alpha_array'):
                pred = model.forward(X_base[i].reshape(1, n_base_feat), X_aux[i].reshape(1, n_aux_feat), aux_mask[i].reshape(1, n_aux_feat))
                if isinstance(model, AuxDrop_ODL):
                    pred = torch.sum(torch.mul(model.alpha.view(model.max_num_hidden_layers - 2, 1).repeat(1, model.batch_size).view(model.max_num_hidden_layers - 2, 
                                                                        model.batch_size, 1), pred), 0)
                elif isinstance(model, SingleStageResidualNetODL) or isinstance(model, ODLSetSingleStageResidualNet):
                    pred = pred[0]
            else:
                pred = model.forward(X_base[i].reshape(1, n_base_feat), X_aux[i].reshape(1, n_aux_feat), aux_mask[i].reshape(1, n_aux_feat))

        cumulative_error_train += torch.argmax(pred).item() != Y[i]
        # cumulative_error_train += torch.argmax(model.prediction[i]).item() != Y[i]
        writer.add_scalar('train/cumulative_error', cumulative_error_train, i)
        # writer.add_scalar('train/exp_smooth_error', exp_smoothing * (torch.argmax(pred).item() != Y[i]) + (1 - exp_smoothing) * prev_train, i)
        # prev_train = exp_smoothing * (torch.argmax(model.prediction[-1]).item() != Y[i]) + (1 - exp_smoothing) * prev_train
        writer.add_scalar('train/norm_error', cumulative_error_train/i, i)
        writer.add_scalar('train/training_loss', model.loss_array[-1], i)
        if hasattr(model, 'alpha_array'):
            for j in range(len(model.alpha_array[-1])):
                writer.add_scalar(f'alphas/{str(j)}', model.alpha_array[-1][j], i)

    # Calculate error or loss
    if data_name == "ItalyPowerDemand":
        loss = np.mean(model.loss_array)
        # print("The loss in the ", data_name, " dataset is ", loss)
        # loss_list.append(loss)
        trial_stats = loss
    else:
        # prediction = []
        # print('len', len(model.prediction))
        # for i in model.prediction:
        #     prediction.append(torch.argmax(i).item())
        # error = len(prediction) - sum(prediction == label)
        # print("The error in the ", data_name, " dataset is ", error)
        # print("Cumulative error is", cumulative_error_test)
        # trial_stats = error
        trial_stats = cumulative_error_test
        # # logging
        # print(error)
        # print(cumulative_error_test)
    return trial_stats

result = Parallel(n_jobs=min(number_of_experiments, os.cpu_count()))(
    delayed(run_trial)(i) for i in range(number_of_experiments)
)

# result = run_trial(1)

if data_name == "ItalyPowerDemand":
    print(
        "The mean loss in the ",
        data_name,
        " dataset for ",
        number_of_experiments,
        " number of experiments is ",
        np.mean(result),
        " and the standard deviation is ",
        np.std(result),
    )
else:
    print(
        "Model:",
        model_to_run,
        "num. layers:",
        max_num_hidden_layers,
        "hidden dim:",
        qtd_neuron_per_hidden_layer,
        "The mean error in the ",
        data_name,
        " dataset for ",
        number_of_experiments,
        " number of experiments is ",
        np.mean(result),
        "$\pm$",
        np.std(result),
        "with lr=",
        n
    )
