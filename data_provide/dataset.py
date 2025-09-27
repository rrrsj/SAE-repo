import torch
import lightning as pl
from torch.utils.data import Dataset
import json
import pandas as pd 
import pyarrow
import os
from torch.utils.data import IterableDataset, get_worker_info
from pathlib import Path
import random
from transformers import AutoModelForCausalLM, AutoTokenizer

class MyData(Dataset):
    def __init__(self,args,dataset_path):
        super(MyData,self).__init__()
        self.args=args
        self.dataset_path=dataset_path
        self.now_read_number=0#now read data_id
        self.all_read_number=0#all data number
        self.now_file_number=0#next file_id need to read 
        self.file_list=[]#all file in now work_process
        self.device_number=self.args.trainer_config.devices
        self.dataset=[]#data
        self.max_length=self.args.data_config.max_length
        self.tokenizer=AutoTokenizer.from_pretrained(self.args.llm_model_config.model_path)
        self.bos_token_id=self.tokenizer.bos_token_id
        self.eos_token_id=self.tokenizer.eos_token_id
        self.get_all_dataset_path()

    def read_data(self):
        all_data=pd.read_parquet(self.file_list[self.now_file_number])
        for i in range(len(all_data['text'])):
            self.dataset.append(all_data['text'][i])
        random.shuffle(self.dataset)
        self.now_read_number=0
        self.all_read_number=len(self.dataset)
        self.now_file_numebr=self.now_file_number+1
        self.now_file_number=self.now_file_number%len(self.file_list)

    def get_all_dataset_path(self):
        worker_info = get_worker_info()
        files=[self.dataset_path+f.name for f in Path(self.dataset_path).iterdir() if f.is_file()]
        random.shuffle(files)
        if worker_info==None:
            self.file_list=files
        else:
            self.file_list=[files[i+worker_info] for i in range(0,len(files),self.device_number)]
    
    def get_data(self):
        if self.all_read_number==self.now_read_number:
            self.read_data()
        return_data=self.dataset[self.now_read_number]
        self.now_read_number=self.now_read_number+1
        return return_data

    def __getitem__(self,index):
        now_data=[]
        while len(now_data)<self.max_length:
            data_temp=self.get_data()
            word_2_token=self.tokenizer(data_temp)['input_ids']
            if not self.bos_token_id==None:
                now_data.append(self.bos_token_id)    
            now_data=now_data+word_2_token
            if not self.eos_token_id==None:
                now_data.append(self.eos_token_id)

        return torch.tensor(now_data[:self.max_length])
        
    def __len__(self):
        return self.args.trainer_config.epoch




    

    