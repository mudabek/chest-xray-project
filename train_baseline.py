import argparse
from xml.etree.ElementTree import C14NWriterTarget
import yaml
import pathlib
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

import dataset
import transforms as custom_transforms
import trainer
import models
import utils
from convnext.convnext import convnext_small 

torch.manual_seed(7575)


def main(args):
    path_to_config = pathlib.Path(args.path)
    with open(path_to_config) as f:
        config = yaml.safe_load(f)

    # Read config:
    path_root_to_data = pathlib.Path(config['path_to_data'])
    path_to_save_dir = pathlib.Path(config['path_to_save_dir'])

    train_batch_size = int(config['train_batch_size'])
    val_batch_size = int(config['val_batch_size'])
    num_workers = int(config['num_workers'])
    lr = float(config['lr'])
    n_epochs = int(config['n_epochs'])
    small_eval_size = int(config['small_eval_size'])
    eval_freq = int(config['eval_freq'])
    image_size = int(config['image_size'])
    num_classes = int(config['num_classes'])
    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

    # Train and val data transforms:
    train_transforms = transforms.Compose([
        transforms.ToTensor(),
        custom_transforms.XRayCenterCrop(),
        custom_transforms.NormalizeIntensity(),
        transforms.Resize((image_size, image_size)),
    ])

    val_transforms = transforms.Compose([
        transforms.ToTensor(),
        custom_transforms.XRayCenterCrop(),
        custom_transforms.NormalizeIntensity(),
        transforms.Resize((image_size, image_size)),
    ])

    # Datasets:
    train_dataset = dataset.ChexpertDataset(path_root_to_data, mode='train', transforms=train_transforms)
    val_dataset = dataset.ChexpertDataset(path_root_to_data, mode='valid', transforms=val_transforms)
    small_val_dataset = Subset(val_dataset, np.arange(0, small_eval_size))

    # Dataloaders:
    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False, num_workers=num_workers)
    small_val_loader = DataLoader(small_val_dataset, batch_size=val_batch_size, shuffle=False, num_workers=num_workers)

    dataloaders = {
        'train': train_loader,
        'val': val_loader,
        'small_val': small_val_loader,
    }

    # Model
    model = convnext_small()
    head_in_features = model.head.in_features
    model.head = nn.Linear(head_in_features, num_classes)
    model.to(device)

    # Training things
    pos_weights = utils.get_class_weights(path_root_to_data).to(device)
    # criterion = nn.BCEWithLogitsLoss(reduction='sum', pos_weight=pos_weights)
    criterion = utils.FocalLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.99))

    model_trainer = trainer.ModelTrainer(
        model=model,
        dataloaders=dataloaders,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=n_epochs,
        config_dict=config,
        eval_freq=eval_freq,
    )

    model_trainer.train_model()
    model_trainer.eval_model()
    model_trainer.save_results(path_to_save_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Model training script')
    parser.add_argument("-p", "--path", type=str, required=True, help="path to the config file")
    args = parser.parse_args()
    main(args)
