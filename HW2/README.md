# Approach

public: 0.87034

## Step 1. Choose model

Tried RNN, LSTM, GRU.
LSTM is better.

## Step2. Tune model parameters

search with optuna: https://optuna.org/

rnn_layers = 7
rnn_dim = 256
fc_layers = 0
fc_dim = 256

public: 0.8

## Step3. Tune training parameters

search with optuna: https://optuna.org/

batch_size = 8
learning_rate = 2e-3
weight_decay = 0.05
dropout = 0.4

public: 0.82

## Step4. Two step training.

Train first version with lr=2e-3.
Then finetune with lr=2e-4 to get second version.

public: 0.84

## Step5. Ensemble

With 5-fold validation, we got 5 different models as same structure.
Do voting and get final result.

public: 0.86