# Train

```
# set do_train = True
# set parameters in code line 50~70
# set cross validation part in code line 162~173
python main.py
```

# Test

```
# set do_train = False
# set do_test = True
# set model_path
python main.py

# voting
python voting --csv 1.csv 2.csv 3.csv ...etc
```

# Approach

public: 0.87034

## Step 1. Choose model

Tried RNN, LSTM, GRU.
LSTM is better.

## Step2. Tune model parameters

search with optuna

rnn_layers = 7
rnn_dim = 256
fc_layers = 0
fc_dim = 256

public: 0.8

## Step3. Tune training parameters

search with optuna

batch_size = 8
learning_rate = 2e-3
weight_decay = 0.05
dropout = 0.4

public: 0.82

## Step4. Finetune

first try lr=2e-3 -> 0.82

finetune lr=2e-4 -> 0.84

public: 0.84

## Step5. Ensemble

With 5-fold validation, we got 5 different models as same structure.

Do voting and get final result.

public: 0.86


# Reference

optuna: https://optuna.org/