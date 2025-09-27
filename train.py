import lightning as pl
from lightning import Trainer
import torch
import transformers
from lightning.pytorch.strategies import FSDPStrategy
from config.config import Config
from data_provide.dataloader import MyDataLoader
from lightning.pytorch.callbacks import ModelCheckpoint
from module.module import SAETrain

config=Config()

fsdp_strategy = FSDPStrategy()

dataloader=MyDataLoader(config)
dataloader.setup()
train_dataloader=dataloader.train_dataloader()
vali_dataloader=dataloader.val_dataloader()

checkpoint_callback = ModelCheckpoint(
    dirpath=Config.call_back_config.save_path,          
    filename=Config.call_back_config.save_path,    
    every_n_train_steps=Config.call_back_config.save_path,        
    save_top_k=Config.call_back_config.save_top_k,                   
    save_last=Config.call_back_config.save_path  
)

model=SAETrain(config)
trainer = Trainer(
    callbacks=[checkpoint_callback],
    accelerator=Config.trainer_config.accelerate,
    devices=Config.trainer_config.devices,
    strategy=fsdp_strategy,
    precision=Config.trainer_config.precision
)

trainer.fit(model,train_dataloader,vali_dataloader)


