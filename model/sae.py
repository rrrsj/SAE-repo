import torch
import lightning as pl
import torch.nn as nn

class SAE(nn.Module):
    def __init__(self,args):
        super(SAE,self).__init__()
        self.args=args
        self.in_dim=self.args.sae_config.in_dim
        self.latent_dim=self.args.sae_config.latent_dim
        self.norm_type=self.args.sae_config.norm_type
        self.sae_type=self.args.sae_config.sae_type
        self.out_dim=self.in_dim
        self.topk=self.args.sae_config.topk
        self.aux_topk=self.args.sae_config.aux_topk
        parameter_temp=torch.randn(self.in_dim,self.latent_dim)
        self.encoder_linear=nn.Parameter(parameter_temp)
        self.encoder_bias=nn.Parameter(torch.zeros(1,self.in_dim))
        self.activate_bias=nn.Parameter(torch.zeros(1,self.latent_dim))
        self.decoder_linear=nn.Parameter(parameter_temp.T)
        self.activate=nn.ReLU()
        self.dead_concept=torch.zeros((self.latent_dim))
        self.get_norm_function()
        self.get_forward_function()


    def get_norm_function(self):
        if self.norm_type=='layer_norm':
            def norm_function(x):
                self.x_mean=x.mean(dim=-1,keepdim=True).expand(-1,x.shape[-1])
                self.x_std=x.std(dim=-1,keepdim=True).expand(-1,x.shape[-1])
                return (x-self.x_mean)/(self.x_std+1e-6)
            def de_norm_function(x):
                return x*self.x_std+self.x_mean

            self.norm=norm_function
            self.denorm=de_norm_function

        elif self.norm_type=='z_norm':
            def norm_function(x):
                self.x_mean=x.mean(dim=-2,keepdim=True).expand(x.shape[0],-1)
                self.x_std=x.std(dim=-2,keepdim=True).expand(x.shape[0],-1)
                return (x-self.x_mean)/(self.x_std+1e-6)
            def de_norm_function(x):
                return x*self.x_std+self.x_mean
            self.norm=norm_function
            self.denorm=de_norm_function
        
        elif self.norm_type=='rms_norm':
            def norm_function(x):
                self.x_mean=torch.zeros_like(x)
                self.x_std=(((x**2).sum(dim=-1,keepdim=True))**0.5).expand(-1,x.shape[1])
                return (x-self.x_mean)/(self.x_std+1e-6)
            def de_norm_function(x):
                return x*self.x_std+self.x_mean

            self.norm=norm_function
            self.denorm=de_norm_function
    

    def get_forward_function(self):
        if self.sae_type=='topk':
            def get_latent_activate(x):
                index=torch.topk(x,k=self.topk,dim=-1)[1]#index shape batch,token,latent_dim
                mask_index=torch.zeros_like(x)
                mask_index.scatter_(dim=-1,index=index,value=1)
                return x*mask_index,index
            self.get_latent_activate=get_latent_activate
                
        elif self.sae_type=='standard':
            def get_latent_activate(x):
                return self.activate(x),None
            self.get_latent_activate=get_latent_activate
        
        elif self.sae_type=='batch_topk':
            def get_latent_activate(x):
                x_temp=x.reshape(-1)
                index=torch.topk(x_temp,k=self.topk*x.shape[0],dim=-1)[1]
                mask_index=torch.zeros_like(x_temp)
                mask_index[index]=1
                mask_index=mask_index.reshape(**x.shape)
                return x*mask_index,index
            self.get_latent_activate=get_latent_activate

    def aux_recover_activate(self):
        if self.sae_type=='topk':
            def get_aux_activate(x):
                index=torch.topk(x,k=self.aux_topk,dim=-1)[1]#index shape batch,token,latent_dim
                mask_index=torch.zeros_like(x)
                mask_index.scatter_(dim=-1,index=index,value=1)
                dead_mask=torch.where(dead_concept>1e6,1,0)
                dead_mask=dead_mask.reshape(1,-1).expand(x.shape[0],-1)
                return x*mask_index*dead_mask
            self.get_aux_activate=get_aux_activate
                
        elif self.sae_type=='batch_topk':
            def get_aux_activate(x):
                x_temp=x.reshape(-1)
                index=torch.topk(x_temp,k=self.aux_topk*x.shape[0],dim=-1)[1]
                mask_index=torch.zeros_like(x_temp)
                mask_index[index]=1
                mask_index=mask_index.reshape(**x.shape)
                dead_mask=torch.where(dead_concept>1e6,1,0)
                dead_mask=dead_mask.reshape(1,-1).expand(x.shape[0],-1)
                return x*mask_index*dead_mask
            self.get_aux_activate=get_aux_activate

    def forward(self,inputs):
        inputs=inputs.reshape(-1,inputs.shape[-1])
        inputs=self.norm(inputs)
        inputs=inputs-self.encoder_bias
        inputs=inputs@self.encoder_linear.T 
        inputs=inputs+self.activate_bias
        recover_activate,index=self.get_latent_activate(inputs)
        aux_recover_activate=None
        if not index==None:
            self.dead_concept=self.dead_concept+inputs.shape[0]
            self.dead_concept[index%self.latent_dim]=0
            aux_recover_activate=self.get_aux_activate(inputs)
            aux_recover_output=aux_recover_activate@self.decoder_linear.T
            aux_recover_output=aux_recover_output+self.encoder_bias

        recover_output=recover_activate@self.decoder_linear.T
        recover_output=recover_output+self.encoder_bias
        return recover_output,activate,aux_recover_activate

        