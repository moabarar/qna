# Learned Queries for Efficient Local Attention (CVPR 2022 - Oral)

---


[ [Arxiv](https://arxiv.org/abs/2112.11435) ]

![QnA-Overview](qna_github.png)

---
### Updates  (April 19th):
- QnA was accepted for **Oral Presentation at CVPR 2022**
- Implementation of QnA layer and other components are available
- QnA-ViT and training code will be released later this month
  - Code went refactoring - under testing and reproducing results
---
### Models
Pretrained models can be downloaded from this [link](https://drive.google.com/drive/folders/1o2npwntBlF8lkA_j9ZaaVIZ87b0Abg6R?usp=sharing). 

| Model            | Params | GFLOPs | Top-1 | Warmup |
|------------------|:------:|:------:|:-----:|:------:|
| QnA_ViT_tiny     |   16M  |   2.5  |  81.7 |    5   |
| QnA_ViT_tiny_7x7 |   16M  |   2.6  |  82.0 |    5   |
| QnA_ViT_small    |   25M  |   4.4  |  83.2 |    5   |
| QnA_ViT_base     |   56M  |   9.7  |  83.9 |   20   |

---
### Evaluation
Download the model parameters and copy 
```commandline
CUDA_VISIBLE_DEVICES=0 python3 main.py --eval_only \
    --workdir <MODEL_DIR> \ 
    --config configs/imagenet_qna.py \
    --config.model_name <MODEL_DIR> \ 
    --config.dataset_version 5.1.0  \
    --config.data_dir <DATA_DIR> \
    --config.batch_size <BATCH_SIZE> \
    --config.half_precision=False
```
Flags:
```buildoutcfg
- workdir : location to the checkpoints directory
- model_name : the model name, e.g., qna_vit_tiny (see table above for model names - use lowercase names only).
- dataset_version : Tensorflow datasets ImageNet dataset version. Mine was (5.1.0),
                     you can change according to your installed version.
- data_dir : the location of the ImageNet directory (need to have the validation set)
- batch_size : the evaluation batch size
```

---
### Citation
Please cite our paper if you find this repo helpful:
```
@InProceedings{Arar_2022_CVPR,
author = {Arar, Moab and Shamir, Ariel and Bermano, Amit H.},
title = {Learned Queries for Efficient Local Attention},
booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
month = {June},
year = {2022}
}
```
