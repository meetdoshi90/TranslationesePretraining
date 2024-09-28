class Config:
    #Model config
    #batches_to_train = 8
    vocab_size = 56000
    val_every = 0.25
    bs = 48
    #bs = 1
    n_embed = 768
    num_blocks = 12
    num_heads = 12
    head_size = n_embed // num_heads
    context_len = 1024
    block_size = context_len
    attn_drop_value = 0.1
    dropout = 0.1
    ffn_drop_value = 0.1
    use_flashattn=True
    ffn_scaling = 4
    positional_embedding='rope'
    rotatory_embedding_dim=head_size//2

    #Optimizer config
    lr = 6e-4
    wd = 1e-1 #originally 1e-5
    beta_1 = 0.9
    beta_2 = 0.95
    eps = 1e-5



    #Wandb config
    wandb = True
    data_name = 'ft-hi-IndicWikiBio'
    wandb_name = 'ft-ft-syn-hi-small-extended-syn-hi-IndicWikiBio'
    wandb_project = 'ACL'

    #Trainer config
    epochs = 2
    precision="bf16"
    checkpoint_every_n_steps=250
    save_top_k=-1
    accumulate_grad_batches=8
    gradient_clip_val=1.0
    log_every_n_steps=3
    strategy='ddp'
    accelerator='gpu'
    #warmup_steps=5000

    #Dataloader config
    val_id_to_name=None
    train_id_to_name=None
    num_workers=16
    SHUFFLE_SEED=42
    PIN_MEMORY=True

    #Tokenizer config
    UNK_TOKEN_ID=0
    BOS_TOKEN_ID=1
    EOS_TOKEN_ID=2
    PAD_TOKEN_ID=3

    #GPU config
    NUM__NODES=1
    NUM_DEVICES=2
    #NUM_DEVICES=1

    #Paths
    train_file_path = f'/raid/nlp/pranavg/meet/IndicLLM/IndicGPT/data/train-data-{data_name}/'
    val_file_path = f'/raid/nlp/pranavg/meet/IndicLLM/IndicGPT/data/val-data-{data_name}/'
    tokenizer_path = "/raid/nlp/pranavg/meet/IndicLLM/IndicGPT/models/ACL24_Multi_8.model"
    #checkpoint_dir = f'./checkpoints-{wandb_name}/'
    checkpoint_dir = f'./checkpoints-ft-syn-hi-small-extended-syn-hi/last.ckpt'
    ft_dir = f'./checkpoints-{wandb_name}/'

