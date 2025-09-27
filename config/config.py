import torch



class Data_config:
    train_dataset_path='/data/coding/sae/dataset/plain_text/partial-train/'
    vali_dataset_path='/data/coding/sae/dataset/plain_text/partial-train/'
    num_workers=1
    batch_size=1
    max_length=1024

class LLM_Model_config:
    model_path='/data/coding/sae/model_checkpoint'
    hugging_face_path=''


class SAE_config:
    norm_type='z_norm'
    in_dim=0
    latent_dim=1560
    topk=96
    sae_type='batch_topk'

class Trainer_config:
    accelerate='gpu'
    devices=1
    #precision='bf16-mixed'
    precision='32'
    epoch=1000000

class Call_back_config:
    save_path=''
    file_name=''
    every_n_train_steps=1000
    save_last=True
    save_top_k=-1

class Config:
    trainer_config=Trainer_config()
    sae_config=SAE_config()
    data_config=Data_config()
    llm_model_config=LLM_Model_config()
    call_back_config=Call_back_config()