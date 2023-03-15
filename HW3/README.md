# Approach

public: 0.90733

## Model

Tried Resnet18, Resnet34, Resnet50

Choose Resnet50

efficientnet_v2_s or efficientnet_v2_m may be better

## Data Augmentation

Ref: 
https://zhuanlan.zhihu.com/p/430563265

RandomResizedCrop

RandomHorizontalFlip

TrivialAugmentWide

Normalize

RandomErasing

## Mix

Ref: 
https://www.kaggle.com/code/ar2017/pytorch-efficientnet-train-aug-cutmix-fmix

https://github.com/ecs-vlc/FMix/blob/master/fmix.py

https://github.com/CoinCheung/pytorch-loss/blob/master/focal_loss.py

cutmix + fmix + focal loss

## SGD v.s. Adam

Ref: 

https://opt-ml.org/papers/2021/paper53.pdf

choose SGD

## Finetune

1st round: lr = 0.1   -> valid acc: 0.8

2st round: lr = 0.01  -> valid acc: 0.89

3st round: lr = 0.001 -> valid acc: 0.9

## Train Time Augmentation

final = test*0.5 + (train*0.1) * 5

## Five Fold Cross Validation Ensemble

train 5 models with 5-fold validation, and ensemble 5 results with voting.