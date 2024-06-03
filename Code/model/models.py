import pandas as pd
import numpy as np
import pytorch_lightning as pl

import torch.nn as nn
import torch.nn.functional as F
import torch

from utils.model_factory import instantiate
from metrics import CumulativeError, NormalizedCumulativeError, SmoothedCumulativeError, MovingWindowAccuracy
from modules import ODLSetSingleStageResidualNet, SetDecoder, MLP


class OnlineDelta(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters()
        self.backbone = instantiate(cfg.model.nn.backbone)
        self.init_metrics()
        self.loss = instantiate(cfg.model.loss)
        
    def init_metrics(self):
        self.train_norm_err = NormalizedCumulativeError()
        self.train_err = CumulativeError()

    def shared_forward(self, x): 
        prediction = self.backbone(x)   
        return {'prediction': prediction}

    def forward(self, x):
        out = self.shared_forward(x)
        return out['prediction']

    def training_step(self, batch, batch_idx):
        batch_size=1
        net_output = self.shared_forward(batch)
        y_hat = net_output['prediction'][0]
        y_hat = y_hat.reshape(1,-1)
   
        self.train_err(y_hat, batch['Y'])
        self.log("train/cumulative_error", self.train_err.compute(), on_step=True, on_epoch=True, 
                 prog_bar=True, logger=True, batch_size=batch_size)
        
        self.train_norm_err(y_hat, batch['Y'])
        self.log("train/normalized_error", self.train_norm_err.compute(), on_step=True, on_epoch=False, 
                 prog_bar=True, logger=True, batch_size=batch_size)

        loss = self.loss(
                y_hat.view(batch_size, 2),
                batch['Y'].view(batch_size).long(),
            )
        
        return loss

    def configure_optimizers(self):
        optimizer = instantiate(self.cfg.model.optimizer, self.parameters())
        return optimizer

class OnlineDeltaMix(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters()
        self.backbone = instantiate(cfg.model.nn.backbone)
        self.merge = cfg.model.nn.backbone.merge
        self.init_metrics()
        self.loss = instantiate(cfg.model.loss)
        self.automatic_optimization = False
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.moe = nn.Linear(2*3, 2)
        
    def init_metrics(self):
        self.train_norm_err = NormalizedCumulativeError()
        self.train_err = CumulativeError()
        self.train_norm_err_MLP = NormalizedCumulativeError()
        self.train_norm_err_LR = NormalizedCumulativeError()
        self.train_norm_err_Proto = NormalizedCumulativeError()

    def shared_forward(self, x): 
        prediction = self.backbone(x)   
        return {'prediction': prediction}

    def forward(self, x):
        out = self.shared_forward(x)
        return out['prediction']

    def training_step(self, batch, batch_idx):
        opt = self.optimizers()
        opt.zero_grad()
        
        batch_size=1
        net_output = self.shared_forward(batch)
        y_hat_lr = net_output['prediction'][0]
        y_hat_lr = y_hat_lr.reshape(1,-1)

        y_hat_MLP = net_output['prediction'][1]
        y_hat_MLP = y_hat_MLP.reshape(1,-1)

        y_hat_proto = net_output['prediction'][2]
        y_hat_proto = y_hat_proto.reshape(1,-1)

        self.train_norm_err_MLP(y_hat_MLP, batch['Y'])
        self.log("train/normalized_error_MLP", self.train_norm_err_MLP.compute(), on_step=True, on_epoch=False, 
                 prog_bar=True, logger=True, batch_size=batch_size)

        self.train_norm_err_LR(y_hat_lr, batch['Y'])
        self.log("train/normalized_error_LR", self.train_norm_err_LR.compute(), on_step=True, on_epoch=False, 
                     prog_bar=True, logger=True, batch_size=batch_size)

        self.train_norm_err_Proto(y_hat_proto, batch['Y'])
        self.log("train/train_norm_err_Proto", self.train_norm_err_Proto.compute(), on_step=True, on_epoch=False, 
                     prog_bar=True, logger=True, batch_size=batch_size)
        
        weights = np.array([1-self.train_norm_err_LR.compute(), 1-self.train_norm_err_MLP.compute(), 1-self.train_norm_err_Proto.compute()])/0.1
        weights = torch.Tensor(weights).view(1, -1)
        weights = torch.softmax(weights, dim=1)[0]

        if self.merge == "sum":
            y_hat = y_hat_lr + y_hat_MLP + 0.25*y_hat_proto  
        if self.merge == "mul":
            y_hat = self.logsoftmax(y_hat_lr) + self.logsoftmax(y_hat_MLP) + self.logsoftmax(y_hat_proto)
        if self.merge == "soft":
            y_hat = y_hat_lr * weights[0] + y_hat_MLP * weights[1] + y_hat_proto * weights[2]
        if self.merge == "ens":
            y_hat = torch.softmax(y_hat_lr, dim=1) + torch.softmax(y_hat_MLP, dim=1) + torch.softmax(y_hat_proto, dim=1)
        if self.merge == "moe":
            y_hat = self.moe(torch.cat((torch.softmax(y_hat_lr, dim=1), torch.softmax(y_hat_MLP, dim=1), torch.softmax(y_hat_proto, dim=1)), axis=1))
        
        self.train_err(y_hat, batch['Y'])
        self.log("train/cumulative_error", self.train_err.compute(), on_step=True, on_epoch=False, 
                 prog_bar=True, logger=True, batch_size=batch_size)
        
        self.train_norm_err(y_hat, batch['Y'])
        self.log("train/normalized_error", self.train_norm_err.compute(), on_step=True, on_epoch=False, 
                 prog_bar=True, logger=True, batch_size=batch_size)

        loss = self.loss(
                y_hat.view(batch_size, 2),
                batch['Y'].view(batch_size).long(),
            )

        self.manual_backward(loss)
        opt.step()
        return loss

    def configure_optimizers(self):
        optimizer = instantiate(self.cfg.model.optimizer, self.parameters())
        return optimizer

class OnlineDeltaU(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters()
        self.backbone = instantiate(cfg.model.nn.backbone)
        self.init_metrics()
        self.loss = instantiate(cfg.model.loss)
        self.automatic_optimization = False

        
    def init_metrics(self):
        self.train_norm_err = NormalizedCumulativeError()
        self.train_err = CumulativeError()
        self.train_norm_err_MLP = NormalizedCumulativeError()
        self.train_norm_err_LR = NormalizedCumulativeError()

    def shared_forward(self, x): 
        prediction = self.backbone(x)   
        return {'prediction': prediction}

    def forward(self, x):
        out = self.shared_forward(x)
        return out['prediction']

    def training_step(self, batch, batch_idx):
        opt = self.optimizers()
        opt.zero_grad()
        
        batch_size=1
        net_output = self.shared_forward(batch)
        y_hat_lr = net_output['prediction'][0]
        y_hat_lr = y_hat_lr.reshape(1,-1)

        y_hat_MLP = net_output['prediction'][1]
        y_hat_MLP = y_hat_MLP.reshape(1,-1)


        self.train_norm_err_MLP(y_hat_MLP, batch['Y'])
        self.log("train/normalized_error_MLP", self.train_norm_err_MLP.compute(), on_step=True, on_epoch=False, 
                 prog_bar=True, logger=True, batch_size=batch_size)

        self.train_norm_err_LR(y_hat_lr, batch['Y'])
        self.log("train/normalized_error_LR", self.train_norm_err_LR.compute(), on_step=True, on_epoch=False, 
                     prog_bar=True, logger=True, batch_size=batch_size)
        weights = np.array([1-self.train_norm_err_LR.compute(), 1-self.train_norm_err_MLP.compute()])/0.1
        weights = torch.Tensor(weights).view(1, -1)
        weights = torch.softmax(weights, dim=1)[0]

        y_hat = y_hat_lr * weights[0] + y_hat_MLP * weights[1]
        
        self.train_err(y_hat, batch['Y'])
        self.log("train/cumulative_error", self.train_err.compute(), on_step=True, on_epoch=True, 
                 prog_bar=True, logger=True, batch_size=batch_size)
        
        self.train_norm_err(y_hat, batch['Y'])
        self.log("train/normalized_error", self.train_norm_err.compute(), on_step=True, on_epoch=False, 
                 prog_bar=True, logger=True, batch_size=batch_size)

        loss = self.loss(
                y_hat.view(batch_size, 2),
                batch['Y'].view(batch_size).long(),
            )

        self.manual_backward(loss)
        opt.step()
        return loss

    def configure_optimizers(self):
        optimizer = instantiate(self.cfg.model.optimizer, self.parameters())
        return optimizer

class OnlineMoE(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters()
        self.backbone = instantiate(cfg.model.nn.backbone)
        self.init_metrics()
        self.loss = instantiate(cfg.model.loss)
        self.theta = torch.zeros(cfg.model.nn.backbone.size_in + 1)
        self.Hessian = cfg.model.variance*torch.eye(cfg.model.nn.backbone.size_in + 1)
        self.temperature = cfg.model.temperature
        
    def init_metrics(self):
        self.train_norm_err = NormalizedCumulativeError()
        self.train_norm_err_MLP = NormalizedCumulativeError()
        self.train_norm_err_LR = NormalizedCumulativeError()
        self.train_err = CumulativeError()

    def shared_forward(self, x): 
        prediction = self.backbone(x)   
        return {'prediction': prediction}

    def forward(self, x):
        out = self.shared_forward(x)
        return out['prediction']

    def training_step(self, batch, batch_idx):
        batch_size=1
        net_output = self.shared_forward(batch)
        y_hat_MLP = net_output['prediction'][0]
        y_hat_MLP = y_hat_MLP.reshape(1,-1)

        loss = self.loss(
                y_hat_MLP.view(batch_size, 2),
                batch['Y'].view(batch_size).long(),
            )

        self.train_norm_err_MLP(y_hat_MLP, batch['Y'])
        self.log("train/normalized_error_MLP", self.train_norm_err_MLP.compute(), on_step=True, on_epoch=False, 
                 prog_bar=True, logger=True, batch_size=batch_size)
        
        with torch.no_grad():
            x = batch
            X = x['X_base']
            aux_feat = x['X_aux_new']
            aux_mask = x['aux_mask']
            Y = x['Y']
            x = torch.cat([X, aux_feat * aux_mask], axis=1)[0]
            x = torch.cat([x, torch.ones(1)])
            
            def sigmoid(x):
                x = torch.clamp(x, min=-10, max=10)
                return 1/(1+torch.exp(-x))
    
            y_hat_lr = np.array([1-sigmoid(x @ self.theta).detach(), sigmoid(x @ self.theta).detach()])
            y_hat_lr = y_hat_lr.T
            y_hat_lr = torch.Tensor(y_hat_lr)
            y_hat_lr = y_hat_lr.reshape(1,-1)
    
            def rirls(X, y, theta0, Lambda0, theta):
                H_k = X#.numpy().astype(np.float32)
                P_k_old = Lambda0#.numpy().astype(np.float32)
                S_k = H_k @ P_k_old @ H_k.T + sigmoid(H_k @ theta0).detach().item() * (1-sigmoid(H_k @ theta0).detach().item())
                K_k = P_k_old @ H_k.T * 1/S_k
                theta = theta0 + K_k * (y - sigmoid(H_k @ theta0))
                Hessian = P_k_old - torch.outer(K_k, K_k) * S_k
                return theta, Hessian
    
            self.theta, self.Hessian = rirls(x, Y, theta0=self.theta, Lambda0=self.Hessian, theta=self.theta)
            
            self.train_norm_err_LR(y_hat_lr, batch['Y'])
            self.log("train/normalized_error_LR", self.train_norm_err_LR.compute(), on_step=True, on_epoch=False, 
                     prog_bar=True, logger=True, batch_size=batch_size)
            # weights = np.array([1-self.train_norm_err_LR.compute(), 1-self.train_norm_err_MLP.compute()])/self.temperature
            # weights = torch.Tensor(weights).view(1, -1)
            # weights = torch.softmax(weights, dim=1)[0]
            # y_hat = y_hat_lr * weights[0] + y_hat_MLP * weights[1]
            y_hat = y_hat_lr + y_hat_MLP
            # y_hat = F.log_softmax(y_hat_lr, dim=1) + F.log_softmax(y_hat_MLP, dim=1) 
            
            
            self.train_err(y_hat, batch['Y'])
            self.log("train/cumulative_error", self.train_err.compute(), on_step=True, on_epoch=True, 
                     prog_bar=True, logger=True, batch_size=batch_size)
            
            self.train_norm_err(y_hat, batch['Y'])
            self.log("train/normalized_error", self.train_norm_err.compute(), on_step=True, on_epoch=False, 
                     prog_bar=True, logger=True, batch_size=batch_size)
        return loss

    def configure_optimizers(self):
        optimizer = instantiate(self.cfg.model.optimizer, self.parameters())
        return optimizer
    
class OnlineLearner(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters()
        self.backbone = instantiate(cfg.model.nn.backbone)
        self.init_metrics()
        self.loss = instantiate(cfg.model.loss)
        self.automatic_optimization = False
        self.theta = torch.zeros(cfg.model.nn.backbone.features_size + cfg.model.nn.backbone.n_aux_feat + 1)
        self.Hessian = 0.01*torch.eye(cfg.model.nn.backbone.features_size + cfg.model.nn.backbone.n_aux_feat + 1)
        
    def init_metrics(self):
        self.train_norm_err = NormalizedCumulativeError()
        self.train_exp_err = SmoothedCumulativeError()
        self.train_err = CumulativeError()

    def shared_forward(self, x): 
        prediction = self.backbone(x)   
        # self.backbone.update_alpha(prediction[1], x['Y'])
        return {'prediction': prediction}

    def forward(self, x):
        out = self.shared_forward(x)
        return out['prediction']

    def training_step(self, batch, batch_idx):
        
        opt = self.optimizers()
        opt.zero_grad()
        batch_size=1
        net_output = self.shared_forward(batch)
        # print(batch)
        # exit()
        # print(net_output['prediction'][0])
        # print(net_output['prediction'][1])
        # print(net_output['prediction'][2])
        # exit()
        
        y_hat = net_output['prediction'][0]
        # loss = self.loss(y_hat, batch['Y']) 
        losses_per_layer = []
        for out in net_output['prediction'][1]:
            criterion = self.loss
            loss = criterion(
                out.view(batch_size, 2),
                batch['Y'].view(batch_size).long(),
            )
            losses_per_layer.append(loss)

        total_loss = torch.dot(self.backbone.alpha, torch.stack(losses_per_layer))
        self.manual_backward(total_loss)
        opt.step()
        
        self.backbone.update_alpha(losses_per_layer, batch['Y'])
        # self.manual_backward(loss)
        # opt.step()
        
        # self.log("train/loss", loss, on_step=False, on_epoch=True, 
        #          prog_bar=True, logger=True, batch_size=batch_size)

        # # Skeleton code for Online Bayesian Logistic Regression
        # x = batch
        # X = x['X_base']
        # aux_feat = x['X_aux_new']
        # aux_mask = x['aux_mask']
        # Y = x['Y']
        # x = torch.cat([X, aux_feat * aux_mask], axis=1)[0]
        # x = torch.cat([x, torch.ones(1)])
        
        # def sigmoid(x):
        #     x = torch.clamp(x, min=-10, max=10)
        #     return 1/(1+torch.exp(-x))

        # y_hat = np.array([1-sigmoid(x @ self.theta).detach(), sigmoid(x @ self.theta).detach()])
        # # print('preds', sigmoid(x @ self.theta).detach())
        # y_hat = y_hat.T
        # # print(y_hat)
        # # print(y_hat.shape)
        # # exit()
        # y_hat = torch.Tensor(y_hat)
        # y_hat = y_hat.reshape(1,-1)
        # # print(y_hat)
        # # print(sigmoid(x @ self.theta))
        # # exit()
        
        # def rirls(X, y, theta0, Lambda0, theta):
        #     H_k = X#.numpy().astype(np.float32)
        #     P_k_old = Lambda0#.numpy().astype(np.float32)
        #     S_k = H_k @ P_k_old @ H_k.T + sigmoid(H_k @ theta0).detach().item() * (1-sigmoid(H_k @ theta0).detach().item())
        #     K_k = P_k_old @ H_k.T * 1/S_k
        #     theta = theta0 + K_k * (y - sigmoid(H_k @ theta0))
        #     Hessian = P_k_old - torch.outer(K_k, K_k) * S_k
        #     return theta, Hessian

        # self.theta, self.Hessian = rirls(x, Y, theta0=self.theta, Lambda0=self.Hessian, theta=self.theta)
        # # End of online regression code
                
        self.train_err(y_hat, batch['Y'])
        self.log("train/cumulative_error", self.train_err.compute(), on_step=True, on_epoch=True, 
                 prog_bar=True, logger=True, batch_size=batch_size)
        
        self.train_norm_err(y_hat, batch['Y'])
        self.log("train/normalized_error", self.train_norm_err.compute(), on_step=True, on_epoch=False, 
                 prog_bar=True, logger=True, batch_size=batch_size)

        # self.train_exp_err(y_hat, batch['Y'])
        # self.log("train/smoothed_error", self.train_exp_err.compute(), on_step=True, on_epoch=False, 
        #          prog_bar=True, logger=True, batch_size=batch_size)
        # for i, a in enumerate(self.backbone.alpha_array[0]):
        #     self.log(f"train/alpha_{i}", a, on_step=True, on_epoch=False, prog_bar=True,
        #             logger=True, batch_size=batch_size)
        return loss
    
    # def validation_step(self, batch, batch_idx):
    #     pass
        
    # def test_step(self, batch, batch_idx):
    #     pass

    # def on_after_optimizer_step(self, batch, batch_idx):
    #     print("here")
    #     prediction = self.shared_forward(batch)
    #     self.backbone.update_alpha(prediction[1], batch['Y'])

    def configure_optimizers(self):
        optimizer = instantiate(self.cfg.model.optimizer, self.parameters())
        # scheduler = instantiate(self.cfg.model.scheduler, optimizer)
        # if scheduler is not None:
        #     optimizer = {"optimizer": optimizer, 
        #                  "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}
        return optimizer


class AlphaExperiment(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters()
        self.backbone = instantiate(cfg.model.nn.backbone)
        self.init_metrics()
        self.loss = instantiate(cfg.model.loss)
        self.automatic_optimization = False
        
    def init_metrics(self):
        self.train_norm_err = NormalizedCumulativeError()
        self.train_exp_err = SmoothedCumulativeError()
        self.train_err = CumulativeError()
        self.acc = [MovingWindowAccuracy() for _ in range(len(self.backbone.alpha))]

    def shared_forward(self, x): 
        prediction = self.backbone(x)   
        # self.backbone.update_alpha(prediction[1], x['Y'])
        return {'prediction': prediction}

    def forward(self, x):
        out = self.shared_forward(x)
        return out['prediction']

    def training_step(self, batch, batch_idx):
        
        opt = self.optimizers()
        opt.zero_grad()
        batch_size=1
        net_output = self.shared_forward(batch)
        # print(batch)
        # print(net_output)
        # exit()
        
        y_hat = net_output['prediction'][0]
        # loss = self.loss(y_hat, batch['Y']) 
        losses_per_layer = []
        for out in net_output['prediction'][1]:
            criterion = self.loss
            loss = criterion(
                out.view(batch_size, 2),
                batch['Y'].view(batch_size).long(),
            )
            losses_per_layer.append(loss)

        total_loss = torch.dot(self.backbone.alpha, torch.stack(losses_per_layer))
        self.manual_backward(total_loss)
        opt.step()
        
        # self.backbone.update_alpha(losses_per_layer, batch['Y'])
        # self.manual_backward(loss)
        # opt.step()
        
        # self.log("train/loss", loss, on_step=False, on_epoch=True, 
        #          prog_bar=True, logger=True, batch_size=batch_size)
        
        self.train_err(y_hat, batch['Y'])
        self.log("train/cumulative_error", self.train_err.compute(), on_step=True, on_epoch=False, 
                 prog_bar=True, logger=True, batch_size=batch_size)
        
        self.train_norm_err(y_hat, batch['Y'])
        self.log("train/normalized_error", self.train_norm_err.compute(), on_step=True, on_epoch=False, 
                 prog_bar=True, logger=True, batch_size=batch_size)

        # self.train_exp_err(y_hat, batch['Y'])
        # self.log("train/smoothed_error", self.train_exp_err.compute(), on_step=True, on_epoch=False, 
        #          prog_bar=True, logger=True, batch_size=batch_size)

        for i, a in enumerate(self.backbone.alpha):
            self.log(f"train/alpha_{i}", a, on_step=True, on_epoch=False, prog_bar=False,
                    logger=True, batch_size=batch_size)
            self.acc[i](net_output['prediction'][1][i], batch['Y'])
            self.log(f"train/clf-{i}-accuracy", self.acc[i].compute(), on_step=True, on_epoch=False, 
                 prog_bar=False, logger=True, batch_size=batch_size)
        return loss
    
    # def validation_step(self, batch, batch_idx):
    #     pass
        
    # def test_step(self, batch, batch_idx):
    #     pass

    # def on_after_optimizer_step(self, batch, batch_idx):
    #     print("here")
    #     prediction = self.shared_forward(batch)
    #     self.backbone.update_alpha(prediction[1], batch['Y'])

    def configure_optimizers(self):
        optimizer = instantiate(self.cfg.model.optimizer, self.parameters())
        # scheduler = instantiate(self.cfg.model.scheduler, optimizer)
        # if scheduler is not None:
        #     optimizer = {"optimizer": optimizer, 
        #                  "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}
        return optimizer


class VariableAlphaExperiment(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters()
        self.backbone = instantiate(cfg.model.nn.backbone)
        self.init_metrics()
        self.loss = instantiate(cfg.model.loss)
        self.automatic_optimization = False
        
    def init_metrics(self):
        self.train_norm_err = NormalizedCumulativeError()
        self.train_exp_err = SmoothedCumulativeError()
        self.train_err = CumulativeError()
        self.acc = [MovingWindowAccuracy() for _ in range(len(self.backbone.alpha))]

    def shared_forward(self, x): 
        prediction = self.backbone(x)   
        # self.backbone.update_alpha(prediction[1], x['Y'])
        return {'prediction': prediction}

    def forward(self, x):
        out = self.shared_forward(x)
        return out['prediction']

    def training_step(self, batch, batch_idx):
        
        opt = self.optimizers()
        opt.zero_grad()
        batch_size=1
        net_output = self.shared_forward(batch)
        # print(batch)
        # print(net_output)
        # exit()
        
        y_hat = net_output['prediction'][0]
        # loss = self.loss(y_hat, batch['Y']) 
        losses_per_layer = []
        for out in net_output['prediction'][1]:
            criterion = self.loss
            loss = criterion(
                out.view(batch_size, 2),
                batch['Y'].view(batch_size).long(),
            )
            losses_per_layer.append(loss)

        total_loss = torch.dot(self.backbone.alpha, torch.stack(losses_per_layer))
        self.manual_backward(total_loss)
        opt.step()
        
        self.backbone.update_alpha(losses_per_layer, batch['Y'])
        # self.manual_backward(loss)
        # opt.step()
        
        # self.log("train/loss", loss, on_step=False, on_epoch=True, 
        #          prog_bar=True, logger=True, batch_size=batch_size)
        
        self.train_err(y_hat, batch['Y'])
        self.log("train/cumulative_error", self.train_err.compute(), on_step=True, on_epoch=True, 
                 prog_bar=True, logger=True, batch_size=batch_size)
        
        self.train_norm_err(y_hat, batch['Y'])
        self.log("train/normalized_error", self.train_norm_err.compute(), on_step=True, on_epoch=False, 
                 prog_bar=True, logger=True, batch_size=batch_size)

        self.train_exp_err(y_hat, batch['Y'])
        self.log("train/smoothed_error", self.train_exp_err.compute(), on_step=True, on_epoch=False, 
                 prog_bar=True, logger=True, batch_size=batch_size)

        for i, a in enumerate(self.backbone.alpha):
            self.log(f"train/alpha_{i}", a, on_step=True, on_epoch=False, prog_bar=True,
                    logger=True, batch_size=batch_size)
            self.acc[i](net_output['prediction'][1][i], batch['Y'])
            self.log(f"train/clf-{i}-accuracy", self.acc[i].compute(), on_step=True, on_epoch=False, 
                 prog_bar=False, logger=True, batch_size=batch_size)
        return loss
    
    # def validation_step(self, batch, batch_idx):
    #     pass
        
    # def test_step(self, batch, batch_idx):
    #     pass

    # def on_after_optimizer_step(self, batch, batch_idx):
    #     print("here")
    #     prediction = self.shared_forward(batch)
    #     self.backbone.update_alpha(prediction[1], batch['Y'])

    def configure_optimizers(self):
        optimizer = instantiate(self.cfg.model.optimizer, self.parameters())
        # scheduler = instantiate(self.cfg.model.scheduler, optimizer)
        # if scheduler is not None:
        #     optimizer = {"optimizer": optimizer, 
        #                  "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}
        return optimizer


class OnlineMLP(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters()
        self.backbone = instantiate(cfg.model.nn.backbone)
        self.init_metrics()
        self.loss = instantiate(cfg.model.loss)
        
    def init_metrics(self):
        self.train_norm_err = NormalizedCumulativeError()
        self.train_exp_err = SmoothedCumulativeError()
        self.train_err = CumulativeError()

    def shared_forward(self, x): 
        prediction = self.backbone(x)   
        return {'prediction': prediction}

    def forward(self, x):
        out = self.shared_forward(x)
        return out['prediction']

    def training_step(self, batch, batch_idx):
        batch_size=1
        net_output = self.shared_forward(batch)
        y_hat = net_output['prediction'][0]
        y_hat = y_hat.reshape(1,-1)
   
        self.train_err(y_hat, batch['Y'])
        self.log("train/cumulative_error", self.train_err.compute(), on_step=True, on_epoch=True, 
                 prog_bar=True, logger=True, batch_size=batch_size)
        
        self.train_norm_err(y_hat, batch['Y'])
        self.log("train/normalized_error", self.train_norm_err.compute(), on_step=True, on_epoch=False, 
                 prog_bar=True, logger=True, batch_size=batch_size)

        loss = self.loss(
                y_hat.view(batch_size, 2),
                batch['Y'].view(batch_size).long(),
            )

        return loss

    def configure_optimizers(self):
        optimizer = instantiate(self.cfg.model.optimizer, self.parameters())
        # scheduler = instantiate(self.cfg.model.scheduler, optimizer)
        # if scheduler is not None:
        #     optimizer = {"optimizer": optimizer, 
        #                  "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}
        return optimizer


class OnlineLogisticRegression(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters()
        self.backbone = instantiate(cfg.model.nn.backbone)
        self.init_metrics()
        self.loss = instantiate(cfg.model.loss)
        # self.automatic_optimization = False
        self.theta = torch.zeros(cfg.model.nn.backbone.size_in + 1)
        self.Hessian = 0.01*torch.eye(cfg.model.nn.backbone.size_in + 1)
        
    def init_metrics(self):
        self.train_norm_err = NormalizedCumulativeError()
        self.train_exp_err = SmoothedCumulativeError()
        self.train_err = CumulativeError()

    def shared_forward(self, x): 
        prediction = self.backbone(x)   
        return {'prediction': prediction}

    def forward(self, x):
        out = self.shared_forward(x)
        return out['prediction']

    def training_step(self, batch, batch_idx):
        batch_size=1
        net_output = self.shared_forward(batch)
        y_hat = net_output['prediction'][0]
        y_hat = y_hat.reshape(1,-1)
   
        self.train_err(y_hat, batch['Y'])
        self.log("train/cumulative_error", self.train_err.compute(), on_step=True, on_epoch=True, 
                 prog_bar=True, logger=True, batch_size=batch_size)
        
        self.train_norm_err(y_hat, batch['Y'])
        self.log("train/normalized_error", self.train_norm_err.compute(), on_step=True, on_epoch=False, 
                 prog_bar=True, logger=True, batch_size=batch_size)

        loss = self.loss(
                y_hat.view(batch_size, 2),
                batch['Y'].view(batch_size).long(),
            )

        return loss

    def training_step(self, batch, batch_idx):
        batch_size=1
        net_output = self.shared_forward(batch)
        y_hat = net_output['prediction'][0]
        y_hat = y_hat.reshape(1,-1)
        loss = self.loss(
                y_hat.view(batch_size, 2),
                batch['Y'].view(batch_size).long(),
            )
        
        x = batch
        X = x['X_base']
        aux_feat = x['X_aux_new']
        aux_mask = x['aux_mask']
        Y = x['Y']
        x = torch.cat([X, aux_feat * aux_mask], axis=1)[0]
        x = torch.cat([x, torch.ones(1)])
        
        def sigmoid(x):
            x = torch.clamp(x, min=-10, max=10)
            return 1/(1+torch.exp(-x))

        y_hat = np.array([1-sigmoid(x @ self.theta).detach(), sigmoid(x @ self.theta).detach()])
        y_hat = y_hat.T
        y_hat = torch.Tensor(y_hat)
        y_hat = y_hat.reshape(1,-1)

        def rirls(X, y, theta0, Lambda0, theta):
            H_k = X#.numpy().astype(np.float32)
            P_k_old = Lambda0#.numpy().astype(np.float32)
            S_k = H_k @ P_k_old @ H_k.T + sigmoid(H_k @ theta0).detach().item() * (1-sigmoid(H_k @ theta0).detach().item())
            K_k = P_k_old @ H_k.T * 1/S_k
            theta = theta0 + K_k * (y - sigmoid(H_k @ theta0))
            Hessian = P_k_old - torch.outer(K_k, K_k) * S_k
            return theta, Hessian

        self.theta, self.Hessian = rirls(x, Y, theta0=self.theta, Lambda0=self.Hessian, theta=self.theta)
                
        self.train_err(y_hat, batch['Y'])
        self.log("train/cumulative_error", self.train_err.compute(), on_step=True, on_epoch=True, 
                 prog_bar=True, logger=True, batch_size=batch_size)
        
        self.train_norm_err(y_hat, batch['Y'])
        self.log("train/normalized_error", self.train_norm_err.compute(), on_step=True, on_epoch=False, 
                 prog_bar=True, logger=True, batch_size=batch_size)

        return loss
        
    def configure_optimizers(self):
        optimizer = instantiate(self.cfg.model.optimizer, self.parameters())
        # scheduler = instantiate(self.cfg.model.scheduler, optimizer)
        # if scheduler is not None:
        #     optimizer = {"optimizer": optimizer, 
        #                  "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}
        return optimizer