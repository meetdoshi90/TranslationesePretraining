# from keys import WANDB_KEY

# import wandb

# def wandb_log(**kwargs):
#     for k, v in kwargs.items():
#         wandb.log({k: v})

# def wandb_setup(Config):
#     # W&B Tracking
#     config_dict = dict(vars(Config))
#     # del[config_dict['__module__']]
#     # del[config_dict['__dict__']]
#     # del[config_dict['__weakref__']]
#     # del[config_dict['__doc__']]
#     if Config.wandb:
#         wandb.login(key=WANDB_KEY)

#         run = wandb.init(
#             project='IndicGPT',
#             config=config_dict,
#             group='train_merged.indic',
#             job_type='train',
#         )