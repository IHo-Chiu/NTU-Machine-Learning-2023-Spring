# Train

```
# set hyperparameters in line 75~100
# set do_train = True
# set save_path = model path
# run
python main.py
```

# Test
```
# set do_train = False
# set do_test = True
# set test_path = model path
# run
python main.py
```

# Approach

public: 0.94225

## Conformer

ref: https://pytorch.org/audio/main/generated/torchaudio.models.Conformer.html

## Self-Attention Pooling

ref: https://gist.github.com/pohanchi/c77f6dbfbcbc21c5215acde4f62e4362

## label smoothing cross entropy loss

## tune hyperparameter with optuna

ref: https://optuna.org/