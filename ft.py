import lightning.pytorch as pl

from configs import Config

from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks import LearningRateMonitor

from models.gpt2_rope_ft import GPTModel
from data.data_loader_ft import IndicGPTDataModule

from keys import WANDB_KEY
import wandb
wandb.login(key=WANDB_KEY)

import sentencepiece as spm

import torch
import os

import warnings
warnings.simplefilter('ignore')


if __name__ == "__main__":
    # Init config
    config = Config()
    
    train_file_path = config.train_file_path
    val_file_path = config.val_file_path
    tokenizer_path = config.tokenizer_path
    checkpoint_dir = config.checkpoint_dir
    ft_dir = config.ft_dir

    files = os.listdir(val_file_path)
    val_file_path = [val_file_path+i for i in files if '.mask' not in i]
    val_id_to_name = {i:files[i].split('.')[0] for i in range(len(files))}

    files = os.listdir(train_file_path)
    train_file_path = [train_file_path+i for i in files  if '.mask' not in i]
    train_id_to_name = {i:files[i].split('.')[0] for i in range(len(files))}


    print('Val id',val_id_to_name)
    print('Train id',train_id_to_name)
    Config.val_id_to_name = val_id_to_name
    Config.train_id_to_name = train_id_to_name


    #Wandb Logger
    wandb_logger = WandbLogger(name=config.wandb_name,project=config.wandb_project, job_type='train', offline=True, dir=f'./wandb-{config.wandb_name}/')
    
    #Turn on SDP kernels for flash attention
    if config.use_flashattn:
        torch.backends.cuda.enable_flash_sdp(True) #Enable flash scaled dot product attention
        torch.backends.cuda.enable_mem_efficient_sdp(False) #Enable mem efficient SDP
        torch.backends.cuda.enable_math_sdp(False) #Math sdp
        #Print status
        print(torch.backends.cuda.flash_sdp_enabled())
        print(torch.backends.cuda.mem_efficient_sdp_enabled())
        print(torch.backends.cuda.math_sdp_enabled())


    # Tokenizer
    tokenizer = spm.SentencePieceProcessor(model_file=tokenizer_path)
    
    # Define the model and dataset
    #model = GPTModel(config,tokenizer)
    model = GPTModel.load_from_checkpoint(config.checkpoint_dir)
    model.config = config
    data = IndicGPTDataModule(
        config=config,
        train_file=train_file_path,
        val_file=val_file_path,
        tokenizer=tokenizer,
        batch_size=config.bs
    )
    total_params = sum(p.numel() for name,p in model.named_parameters() if p.requires_grad and ('embedding' not in name and 'lm_head' not in name))
    for name,p in model.named_parameters():
        print(name, p.numel() if p.requires_grad else "None")
    print(f"Total parameters in the model excluding embedding layer: {total_params}")
    
    #Callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=ft_dir, 
        filename="ft-minilm-{epoch:02d}-{val_loss:.7f}",
        save_top_k=config.save_top_k, 
        save_last=True,
        monitor="train_loss",
        every_n_train_steps=config.checkpoint_every_n_steps,
        mode="min",
        save_on_train_epoch_end=True
    )
    lr_monitor = LearningRateMonitor(logging_interval='step')
    
    trainer = pl.Trainer(
        accelerator=config.accelerator, 
        devices=config.NUM_DEVICES, 
        strategy=config.strategy, 
        num_nodes=config.NUM__NODES,
        max_epochs=config.epochs,
        detect_anomaly=True,
        enable_checkpointing=True,
        val_check_interval=config.val_every,
        logger=wandb_logger,
        log_every_n_steps=config.log_every_n_steps,
        callbacks=[checkpoint_callback, lr_monitor],
        precision=config.precision,
        gradient_clip_val=config.gradient_clip_val,
        accumulate_grad_batches=config.accumulate_grad_batches
    )
    
    trainer.fit(model, data)

    print('Best model path', checkpoint_callback.best_model_path)
    wandb.finish()