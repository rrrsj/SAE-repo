import torch
import lightning as pl
from model.sae import SAE
from model.token_2_embedding import Token_2_Embedding
import torch.optim as optim
class SAETrain(pl.LightningModule):
    def __init__(self,args):
        super(SAETrain,self).__init__()
        self.args=args
        self.sae=SAE(args)
        self.token_2_embedding=Token_2_Embedding(args)

    def training_step(self,batch,batch_idx):
        token_data=self.token_2_embedding.get_token_2_embedding(batch)
        recover,activate=self.sae(token_data)
        print(recover.shape)

    def validation_step(self,batch,batch_idx):
        return None

    def configure_optimizers(self):
        self.token_2_embedding.requires_grad_(False)
        optimizer = optim.Adam(self.sae.parameters(), lr=1e-3)
        return optimizer