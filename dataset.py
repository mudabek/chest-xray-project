import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset


class ChexpertDataset(Dataset):
    
    def __init__(self, root_dir, mode='train', transforms=None):
        self.image_paths = [] # List of image paths
        self.image_labels = [] # List of image labels
        
        # Image transformations
        self.transforms = transforms

        # Read the dataframe
        path_df = os.path.join(root_dir, mode + '.csv')
        main_df = pd.read_csv(path_df)        
        
        # Get all image paths and image labels from dataframe
        for index, row in main_df.iterrows():
            image_path = os.path.join(root_dir[:-20], row.Path) # 20 to remove ChexPert index
            self.image_paths.append(image_path)
            labels = []
            for col in row[5:]:
                if col == 1:
                    labels.append(1)
                else:
                    labels.append(0)
            self.image_labels.append(labels)
    

    def __len__(self):
        return len(self.image_paths)
    

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image_data = Image.open(image_path).convert("RGB") # Convert image to RGB channels
        
        if self.transforms:
            image_data = self.transforms(image_data)
        
        return image_data, torch.FloatTensor(self.image_labels[index])


class DummyDataset():
    def __init__(self):
        pass


    def __len__(self):
        return 20
    

    def __getitem__(self, index):
        dummy_image_data = torch.rand(1, 224, 224)
        dummy_labels = torch.rand(14)

        return dummy_image_data, dummy_labels


if __name__ == "__main__":
    root_dir = '/home/otabek.nazarov/Downloads/research/chexpert/CheXpert-v1.0-small/'
    test_dataset = ChexpertDataset(root_dir=root_dir, mode='train')
    for i in range(0, 2):
        print(f'image tensor {test_dataset[i][0]}')
        print(f'labels tensor {test_dataset[i][1]}')
