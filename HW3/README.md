# Train

```
# set hyperparameters in line 65~82
# set do_train = True
# set save_model = model path
# set pretrained = True when finetuning
# set pretrained_model = pretrained model path when finetuning
# set cross_valid_num = 1,2,3,4,5 with 5-fold cross validation
# run
python main.py
```

# Test
```
# set do_train = False
# set do_test = True
# set test_model = model path
# run
python main.py

# voting
python voting --csv 1.csv 2.csv 3.csv ...etc
```

# Approach

public: 0.92866

## Model

Tried Resnet18, Resnet34, Resnet50

Choose Resnet50

## Data Augmentation

Ref: https://zhuanlan.zhihu.com/p/430563265

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

Ref: https://opt-ml.org/papers/2021/paper53.pdf

choose SGD

## Finetune

1st round: lr = 0.1   -> valid acc: 0.8

2st round: lr = 0.01  -> valid acc: 0.89

3st round: lr = 0.001 -> valid acc: 0.91

## Train Time Augmentation

final = test*0.4 + (train*0.12) * 5

## Five Fold Cross Validation Ensemble

train 5 models with 5-fold validation, and ensemble 5 results with voting.