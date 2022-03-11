# General imports
import os
import wandb
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import copy
import pathlib
from metrics import auc_roc_as_dict
from utils import get_classification_thresholds

# Turn off wandb logging by default
os.environ['WANDB_MODE'] = 'offline'



class ModelTrainer:

    def __init__(self, model, dataloaders, criterion, optimizer, config_dict, scheduler=None, cuda_device="cuda:0",  
                 num_epochs=20, save_last_model=True, scheduler_step_per_epoch=True, eval_freq=2000, wandb_logging=False):

        self.model = model
        self.train_loader = dataloaders['train']
        self.val_loader = dataloaders['val']
        self.small_val_loader = dataloaders['small_val']
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.num_epochs = num_epochs
        self.device = torch.device(cuda_device if torch.cuda.is_available() else "cpu")
        self.save_last_model = save_last_model
        self.scheduler_step_per_epoch = scheduler_step_per_epoch
        self.eval_freq = eval_freq

        # Summary: Best epoch, loss, metric and best model weights
        self.best_val_step = 0
        self.best_val_metric = -float('inf')
        self.best_model_weights = None
        self.checkpoint = None  # last model and optimizer weights

        # Logging to wandb
        if wandb_logging:
            os.environ['WANDB_MODE'] = 'online'
        wandb.init(project="chexpert", entity="obatek")
        wandb.config.update(config_dict)


    def train_model(self):
        self.model.train()
        step = 0
        for epoch in range(self.num_epochs):
            print(f"[Epoch {epoch} / {self.num_epochs}]")

            for image_data, label in tqdm(self.train_loader):
                image_data = image_data.to(self.device)        
                label = label.to(self.device)

                self.optimizer.zero_grad()
                output = self.model(image_data)
                loss = self.criterion(output, label)
                loss.backward()
                self.optimizer.step()

                wandb.log({"train loss": loss})
                step += 1

                if step % self.eval_freq == 0:
                    self.model.eval()
                    
                    out_pred = torch.FloatTensor().to('cpu')
                    out_gt = torch.FloatTensor().to('cpu')

                    for image_data, label in tqdm(self.small_val_loader):
                        image_data = image_data.to(self.device)
                        label = label.detach().cpu()

                        val_output = self.model(image_data)
                        probabs = val_output.sigmoid().detach().cpu()
                        
                        out_gt = torch.cat((out_gt,  label), 0)
                        out_pred = torch.cat((out_pred, probabs), 0)

                    roc_auc_dict = auc_roc_as_dict(out_gt, out_pred)
                    
                    if roc_auc_dict['auc_average'] < self.best_val_metric:
                        print('Updating best result')
                        self.best_step = step
                        self.best_val_metric = roc_auc_dict['auc_average']
                        self.best_model_weights = copy.deepcopy(self.model.state_dict())
                    
                    wandb.log(roc_auc_dict)
                    self.model.train()
        
        if self.save_last_model:
            self.checkpoint = {'model_state_dict': copy.deepcopy(self.model.state_dict()),
                               'optimizer_state_dict': copy.deepcopy(self.optimizer.state_dict())}


    def eval_model(self):
        print('===> Evaluating')
        self.model.eval()       
        out_pred = torch.FloatTensor().to('cpu')
        out_gt = torch.FloatTensor().to('cpu')

        for image_data, label in tqdm(self.small_val_loader):
            image_data = image_data.to(self.device)
            label = label.detach().cpu()

            val_output = self.model(image_data)
            probabs = val_output.sigmoid().detach().cpu()
            
            out_gt = torch.cat((out_gt,  label), 0)
            out_pred = torch.cat((out_pred, probabs), 0)

        roc_auc_dict = auc_roc_as_dict(out_gt, out_pred)
        roc_auc_dict = {f'{k}@validation': v for k, v in roc_auc_dict.items()}
        wandb.log(roc_auc_dict)

        # NEED TO SAVE IT IF NEEDED
        #class_thresholds = get_classification_thresholds(out_gt, out_pred)


    def save_results(self, path_to_dir):
        print('===> Saving')
        path_to_dir = pathlib.Path(path_to_dir)

        # Check if the directory exists:
        if not os.path.exists(path_to_dir):
            os.makedirs(path_to_dir)

        # Save best model weights:
        torch.save(self.best_model_weights, path_to_dir / 'best_model_weights.pt')

        # Save last model weights (checkpoint):
        if self.save_last_model:
            torch.save(self.checkpoint, path_to_dir / 'last_model_checkpoint.tar')


    def load_model_weights(self, path_to_checkpoint):
        checkpoint = torch.load(path_to_checkpoint)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
