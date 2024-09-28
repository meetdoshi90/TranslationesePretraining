# Pretraining Language Models Using Translationese

This repository contains the code for the paper published at EMNLP 2024

The dataset can be found on [huggingface.co/datasets/cfilt/IITB-IndicMonoDoc](https://huggingface.co/datasets/cfilt/IITB-IndicMonoDoc)

## Install required libraries:
```
pip3 install -r requirements.txt
```

## Getting started

> In the models/ directory you will find the code for the decoder architecture.

> In the data/ directory you will find the scripts to convert the data for pretraining.

> In the eval/ directory you will find the code for evaluating various benchmarks.

> In the hf/ directory you will find scripts for converting models to HF format.

> configs_pt.py and configs_ft.py are configurations that change the architecture according to needs.

> train.py and ft.py are the main files to run after configurations are set

> ppl_scorer.py gives code for calculating perplexity using TinyLMs

> regression.py and classification_*.py are codes to evaluate classification and regression tasks.

> demo.py is a gradio demo file to evaluate the model, it supports pre and post-processing in all 22 languages.

## Citation 

```
@misc{doshi2024worrydatabuildingpretrained,
      title={Pretraining Language Models Using Translationese}, 
      author={Meet Doshi and Raj Dabre and Pushpak Bhattacharyya},
      year={2024},
      eprint={2403.13638},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2403.13638}, 
}
```
