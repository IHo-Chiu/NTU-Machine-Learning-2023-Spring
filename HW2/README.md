# Approach

Following Google tuning playbook: https://github.com/google-research/tuning_playbook

## Choosing the model architecture

I asked ChatGPT: "i want to train a model do framewise phoneme classification, which model should i use?"

ChatGPT replied:
```
There are several types of models that can be used for framewise phoneme classification. Here are some common ones:

Convolutional Neural Networks (CNNs): CNNs are often used for image classification, but they can also be used for audio classification tasks. The input audio frames can be treated as 2D images, where time is the horizontal axis and frequency is the vertical axis. The CNN can then learn to extract useful features from these frames for classification.

Recurrent Neural Networks (RNNs): RNNs are a type of neural network that can model sequential data. They are commonly used for speech recognition tasks, where the input audio is a sequence of frames. RNNs can also be used for framewise phoneme classification by treating each audio frame as a sequence of features.

Convolutional Recurrent Neural Networks (CRNNs): CRNNs combine the strengths of CNNs and RNNs. They use CNNs to extract features from the audio frames and RNNs to model the temporal dependencies between the frames.

Transformers: Transformers are a type of neural network that were originally designed for natural language processing tasks. However, they have also been shown to be effective for audio classification tasks. Transformers can model long-range dependencies between the audio frames and can learn to attend to the most relevant parts of the audio.

The choice of model depends on various factors, including the size of the dataset, the complexity of the task, and the available computing resources. You can start with a simple model, such as a CNN or an RNN, and gradually increase the complexity as needed.
```

And I tried RNN and CRNN. CRNN is better than RNN in this task.

My CRNN architecture is resnet18 + biLSTM(2 layers, 512 hidden size).

model | public
----- | ------
RNN   | 0.81778
CRNN  |
