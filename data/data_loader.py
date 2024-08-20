import os
import pickle
import re
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import ImageFile,Image
ImageFile.LOAD_TRUNCATED_IMAGES = True

import torch
from torchvision import datasets
from torch.utils.data import Dataset
from torch.utils.data import DataLoader as DL
import torchvision.transforms as T

from .constants import *
from .utils import get_imgs, read_from_dicom,resize_img

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
np.random.seed(42)


###################################################################################################
# CLASSIFICATION DATA
###################################################################################################

class BaseImageDataset(Dataset):
    def __init__(self, split="train", transform=None) -> None:
        super().__init__()

        self.split = split
        self.transform = transform

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError
        

class NIHImageDataset(BaseImageDataset):
    def __init__(self, split="train", transform=None,
                  data_pct=0.01, imsize=256, task = NIH_TASKS):
        super().__init__(split=split, transform=transform)

        if not os.path.exists(NIH_CXR_DATA_DIR):
            raise RuntimeError(f"{NIH_CXR_DATA_DIR} does not exist!")

        self.imsize = imsize

        # read in csv file
        if split == "train":
            self.df = pd.read_csv(NIH_TRAIN_CSV)
        elif split == "valid":
            self.df = pd.read_csv(NIH_TEST_CSV)
        else:
            raise NotImplementedError(f"split {split} is not implemented!")

        # sample data
        if data_pct != 1 and self.split == "train":
            self.df = self.df.sample(frac=data_pct, random_state=42)
        
        #get path
        self.df[NIH_PATH_COL] = self.df[NIH_PATH_COL].apply(lambda x: os.path.join(
                                    NIH_CXR_DATA_DIR, "/".join(x.split("/")[:])))

        # fill na with 0s
        self.df = self.df.fillna(0)

        self.path = self.df[NIH_PATH_COL].values
        self.labels = self.df.loc[:, task].values

    def __getitem__(self, index):
        # get image
        img_path = self.path[index]
        x = get_imgs(img_path, self.imsize, self.transform)

        # get labels
        y = self.labels[index]
        y = torch.tensor(y, dtype=torch.float)

        return x, y

    def __len__(self):
        return len(self.df)    


class NIHPertubedDataset(BaseImageDataset):
    def __init__(self, split="train", transform=None,
                  data_pct=0.01,pertb_pct=10, imsize=256, task = NIH_TASKS):
        super().__init__(split=split, transform=transform)

        if not os.path.exists(NIH_CXR_DATA_DIR):
            raise RuntimeError(f"{NIH_CXR_DATA_DIR} does not exist!")

        self.imsize = imsize
        
        if pertb_pct == 10:
            NIH_PERT_TRAIN_CSV = NIH_PERT_TRAIN_CSV10 
        elif pertb_pct == 20:
            NIH_PERT_TRAIN_CSV = NIH_PERT_TRAIN_CSV20 
        elif pertb_pct == 30:
            NIH_PERT_TRAIN_CSV = NIH_PERT_TRAIN_CSV30 
        elif pertb_pct == 40:
            NIH_PERT_TRAIN_CSV = NIH_PERT_TRAIN_CSV40 
        elif pertb_pct == 50:
            NIH_PERT_TRAIN_CSV = NIH_PERT_TRAIN_CSV50  
        
        
        # read in csv file
        if split == "train":
            self.df = pd.read_csv(NIH_PERT_TRAIN_CSV)
        elif split == "valid":
            self.df = pd.read_csv(NIH_PERT_TEST_CSV)
        else:
            raise NotImplementedError(f"split {split} is not implemented!")

        # sample data
        if data_pct != 1 and self.split == "train":
            self.df = self.df.sample(frac=data_pct, random_state=42)
        
        #get path
        self.df[NIH_PATH_COL] = self.df[NIH_PATH_COL].apply(lambda x: os.path.join(
                                    NIH_CXR_DATA_DIR, "/".join(x.split("/")[:])))

        # fill na with 0s
        self.df = self.df.fillna(0)

        self.path = self.df[NIH_PATH_COL].values
        self.labels = self.df.loc[:, task].values

    def __getitem__(self, index):
        # get image
        img_path = self.path[index]
        x = get_imgs(img_path, self.imsize, self.transform)

        # get labels
        y = self.labels[index]
        y = torch.tensor(y, dtype=torch.float)
        return x, y
    
    def __len__(self):
        return len(self.df)       

        

class ISICImageDataset(Dataset):
    def __init__(self, split="train", transform=None, data_pct=1.0, imsize=256):
        if split == "train":
            self.df = pd.read_csv(ISIC_TRAIN_CSV)
            self.img_dir = ISIC_TRAIN_DIR
        elif split == "valid":
            self.df = pd.read_csv(ISIC_VALID_CSV)
            self.img_dir = ISIC_VALID_DIR
        elif split == "test":
            self.df = pd.read_csv(ISIC_TEST_CSV)
            self.img_dir = ISIC_TEST_DIR
        else:
            raise NotImplementedError(f"split {split} is not implemented!")
        
        if not os.path.exists(self.img_dir):
            raise RuntimeError(f"{self.img_dir} does not exist!")
        
        self.imsize = imsize
        self.transform = transform
        
        # Sample data
        if data_pct != 1 and split == "train":
            self.df = self.df.sample(frac=data_pct, random_state=42)
        
        # Ensure column names are correct
        if 'image' not in self.df.columns or 'class_label' not in self.df.columns:
            raise ValueError("CSV file must contain 'image' and 'class_label' columns")
        
        # Convert class_label to numeric type
        self.df['class_label'] = pd.to_numeric(self.df['class_label'], errors='coerce')
                
        # Get the number of unique classes
        self.num_classes = len(self.df['class_label'].unique())

    def __getitem__(self, index):
        # Get image
        img_name = self.df.iloc[index]['image']
        img_path = os.path.join(self.img_dir, img_name + '.jpg')
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        # Get label
        label = self.df.iloc[index]['class_label']
        label = torch.tensor(label, dtype=torch.long)  # Use long for class indices
        
        return image, label

    def __len__(self):
        return len(self.df)

class DataLoader():
    def __init__(self, config=None):
        self.config = config
        self.train_bs        = config['train_bs']
        self.val_bs          = config['val_bs']
        self.data_workers    = config['data_workers']
        self.data_pct        = config['data_pct']/100.0
        self.pertb_data      = config['pertb_data']
        self.img_size        = config['img_size']
                
        if config['dataset'] == 'nih' or config['dataset'] == 'pertb_nih':
            if config['task'] == 'nih_tasks':
                self.task = NIH_TASKS
            if config['task'] == 'nih_cxr8_tasks':
                self.task = NIH_CXR8_TASKS        
                                      
        self.train_transform = T.Compose([T.Resize((self.img_size, self.img_size)), 
                                          T.RandomHorizontalFlip(),
                                          T.RandomRotation(10),
                                          T.CenterCrop(self.img_size),
                                          T.ToTensor(),           
                                          T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  
                                        ])
            
        self.valid_augmentations = T.Compose([
                                              T.Resize((self.img_size, self.img_size)),
                                              T.CenterCrop(self.img_size),
                                              T.ToTensor(),           
                                              T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  
                                            ])
    
    #####################################################################
    # CLASSIFICATION LOADERS (For Downstream Tasks)
    #####################################################################
    def GetIsicDataset(self):                
        train_transform = self.train_transform
        valid_transform = self.valid_augmentations        
        train_set = ISICImageDataset(split="train",
                                      transform = train_transform,
                                      data_pct=self.data_pct,
                               )
        valid_set = ISICImageDataset(split="valid",
                                 transform =valid_transform,
                               )
        test_set = ISICImageDataset(split="test",
                                 transform =valid_transform,
                               )
        
        
        train_loader = DL(dataset=train_set,
                         batch_size=self.train_bs,
                         shuffle= True,
                         num_workers=self.data_workers,
                         pin_memory=True,
                         drop_last=True)
        
        valid_loader = DL(dataset=valid_set,
                         batch_size=self.val_bs,
                         shuffle= False,
                         num_workers=self.data_workers,
                         pin_memory=True,
                         drop_last=True)
        
        test_loader = DL(dataset=test_set,
                         batch_size=self.val_bs,
                         shuffle= False,
                         num_workers=self.data_workers,
                         pin_memory=True,
                         drop_last=True)
        
        print(f'{len(train_set)} images have loaded for training')
        print(f'{len(valid_set)} images have loaded for validation')
        print(f'{len(test_set)} images have loaded for testing')
                
        return train_loader, valid_loader, test_loader
    
    
    
    def GetNihDataset(self):                
        train_transform = self.train_transform
        valid_transform = self.valid_augmentations        
        train_set = NIHImageDataset(split="train",
                                      transform = train_transform,
                                      data_pct=self.data_pct,
                                      task = self.task
                               )
        valid_set = NIHImageDataset(split="valid",
                                 transform =valid_transform,
                                 task = self.task
                               )
        
        
        train_loader = DL(dataset=train_set,
                         batch_size=self.train_bs,
                         shuffle= True,
                         num_workers=self.data_workers,
                         pin_memory=True,
                         drop_last=True)
        
        valid_loader = DL(dataset=valid_set,
                         batch_size=self.val_bs,
                         shuffle= True,
                         num_workers=self.data_workers,
                         pin_memory=True,
                         drop_last=True)
        
        print(f'{len(train_set)} images have loaded for training')
        print(f'{len(valid_set)} images have loaded for validation')
                
        return train_loader, valid_loader, valid_loader
    
    
    
    def GetPertbNihDataset(self):                
        train_transform = self.train_transform
        valid_transform = self.valid_augmentations        
        train_set = NIHPertubedDataset(split="train",
                                      transform = train_transform,
                                      data_pct=self.data_pct,
                                      pertb_pct = self.pertb_data,
                                      task = self.task
                                      
                               )
        valid_set = NIHPertubedDataset(split="valid",
                                 transform =valid_transform,
                                 task = self.task
                               )
        
        
        train_loader = DL(dataset=train_set,
                         batch_size=self.train_bs,
                         shuffle= True,
                         num_workers=self.data_workers,
                         pin_memory=True,
                         drop_last=True)
        
        valid_loader = DL(dataset=valid_set,
                         batch_size=self.val_bs,
                         shuffle= False,
                         num_workers=self.data_workers,
                         pin_memory=True,
                         drop_last=True)
        
        print(f'{len(train_set)} images have loaded for training')
        print(f'{len(valid_set)} images have loaded for validation')
                
        return train_loader, valid_loader, valid_loader
    