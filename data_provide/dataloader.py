import torch
import lightning as pl
from torch.utils.data import Dataset,DataLoader
from data_provide.dataset import MyData


class MyDataLoader(pl.LightningDataModule):
    def __init__(self,args):
        super(MyDataLoader,self).__init__()
        self.args=args
        self.num_workers=self.args.data_config.num_workers
        self.batch_size=self.args.data_config.batch_size

           
    def setup(self, stage=None):
        self.train_dataset=MyData(self.args,self.args.data_config.train_dataset_path)
        self.vali_dataset=MyData(self.args,self.args.data_config.vali_dataset_path)
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False, 
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.vali_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )