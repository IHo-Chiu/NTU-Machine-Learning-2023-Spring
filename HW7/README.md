# Train

```
# set hyperparameters in line 113
# set do_train = True
# set model_save_dir = model path
# run
python main.py
```

# Test
```
# set do_train = False
# set do_test = True
# set ensemble_list = ["<model_path>_<epoch>"]
# run
python main.py
```

# voting
```
python voting --csv result*.csv
```

# Approach

## cosine learning rate decay

## tuning doc stride 

tried 32, 64, 128

the best doc stride is not sure after comparing different models

## implement gradient accumulation

add gradient accumulation on accelerator

## improve preprocessing

random answer position in paragraph window

## try other pretrained models 

luhua/chinese_pretrain_mrc_macbert_large

## improve postprocessing

skip invalid start and end index

reconstruct unknown words from paragraph

remove invalid brackets

## train all

2 epoch

## ensemble

voting method

trained 3 models with different learning rate: 1e-5, 3e-5, 5e-5, epoch: 2

for each model, produce 3 results with different doc stride: 32, 64, 128

we got 9 different results, and then do voting method.

