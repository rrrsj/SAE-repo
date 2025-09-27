import torch
import torch.nn as nn
import transformers
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from accelerate import infer_auto_device_map

class Token_2_Embedding(nn.Module):
    def __init__(self,args):
        super(Token_2_Embedding,self).__init__()
        self.args=args
        self.llm_model_path=self.args.llm_model_config.model_path
        self.llm_model=AutoModelForCausalLM.from_pretrained(
            self.llm_model_path,
            torch_dtype="auto")
        self.embedding_data=0
        self.register_model_hook()
    
    def register_model_hook(self):
        def hook_fn(module, input, output):
            self.embedding_data=output
        hook_handle =self.llm_model.model.layers[12].register_forward_hook(hook_fn)


    def get_token_2_embedding(self,inputs):
        ans=self.llm_model(inputs)
        return self.embedding_data    

