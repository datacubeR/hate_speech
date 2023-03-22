import lightning as L
import pandas as pd
from .dataset import HateDataset
from torch.utils.data import DataLoader

class HateDataModule(L.LightningDataModule):
    def __init__(
        self, 
        train: pd.DataFrame,
        labels: list,
        fold: int,
        batch_size: list,
        tokenizer: str,
        max_token_len: int = 128,
        test: pd.DataFrame = None):
        
        super().__init__()
        self.train = train
        self.test = test
        self.tokenizer = tokenizer
        self.labels = labels
        self.fold = fold
        self.batch_size = batch_size
        self.max_token_len = max_token_len
        
    def setup(self, stage=None):
        self.train_dataset = HateDataset(
            self.train,
            self.tokenizer,
            self.max_token_len,
            self.labels,
            fold=self.fold,
        )
        
        self.val_dataset = HateDataset(
            self.train, 
            self.tokenizer,
            self.max_token_len,
            self.labels,
            fold = self.fold,
            validation=True,
        )
        
        self.test_dataset = HateDataset(
            self.test, 
            self.tokenizer,
            self.max_token_len,
            self.labels,
        )
        
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=6,
            shuffle=True,
            drop_last=True,
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=6,
            shuffle=False,
            drop_last=True,
        )
    
    def predict_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=6,
            shuffle=False,
        )
        
        
    
    
    
    