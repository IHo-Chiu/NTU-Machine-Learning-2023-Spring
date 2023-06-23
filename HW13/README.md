# Kaggle

public: 0.86654


# Train teacher model with HW3 Approach 

old teacher model validation accuracy: 0.95

new teacher model validation accuracy: 0.98

## Train

```
# set hyperparameters in line 65~82
# set do_train = True
# set save_model = model path
# set pretrained = True when finetuning
# set pretrained_model = pretrained model path when finetuning
# set cross_valid_num = 1,2,3,4,5 with 5-fold cross validation
# run
python train_teacher.py
```

## Model

Resnet50

## Data Augmentation

https://zhuanlan.zhihu.com/p/430563265

## Mix

https://www.kaggle.com/code/ar2017/pytorch-efficientnet-train-aug-cutmix-fmix

https://github.com/ecs-vlc/FMix/blob/master/fmix.py

https://github.com/CoinCheung/pytorch-loss/blob/master/focal_loss.py

cutmix + fmix + focal loss

## SGD v.s. Adam

https://opt-ml.org/papers/2021/paper53.pdf

choose SGD

## 3 rounds training

1st round: lr = 0.1

2nd round: lr = 0.01

3rd round: lr = 0.001

## Train Time Augmentation

final = test*0.4 + (train*0.12) * 5


# Train student model

## Train

```
python main.py
```

## Model Architecture

resnet architecture with Depthwise and Pointwise Convolution

## loss function

Implement the loss function with KL divergence loss for knowledge distillation.

## Same Approach as training teacher model

cutmix + fmix + focal loss

3 rounds training

Train Time Augmentation

# Pruning Ratio v.s. Model Accuracy

```
python pruning.py
```