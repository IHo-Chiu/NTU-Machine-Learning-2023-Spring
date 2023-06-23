# -*- coding: utf-8 -*-

# Import some useful packages for this homework
import numpy as np
import pandas as pd
import torch
import os
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import ConcatDataset, DataLoader, Subset, Dataset # "ConcatDataset" and "Subset" are possibly useful
from torchvision.datasets import DatasetFolder, VisionDataset
from torchsummary import summary
from tqdm.auto import tqdm
import random
import torch.nn.utils.prune as prune
import matplotlib.pyplot as plt

"""### Configs
In this part, you can specify some variables and hyperparameters as your configs.
"""

cfg = {
    'dataset_root': './Food-11',
    'save_dir': './outputs',
    'exp_name': "pruning",
    'batch_size': 64,
    'seed': 20220013,
}

myseed = cfg['seed']  # set a random seed for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(myseed)
torch.manual_seed(myseed)
random.seed(myseed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(myseed)

save_path = os.path.join(cfg['save_dir'], cfg['exp_name']) # create saving directory
os.makedirs(save_path, exist_ok=True)

# define simple logging functionality
log_fw = open(f"{save_path}/log.txt", 'w') # open log file to save log outputs
def log(text):     # define a logging function to trace the training process
    print(text)
    log_fw.write(str(text)+'\n')
    log_fw.flush()

log(cfg)  # log your configs to the log file

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# define training/testing transforms
test_tfm = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize,
])

class FoodDataset(Dataset):
    def __init__(self, path, tfm=test_tfm, files = None):
        super().__init__()
        self.path = path
        self.files = sorted([os.path.join(path,x) for x in os.listdir(path) if x.endswith(".jpg")])
        if files != None:
            self.files = files
        print(f"One {path} sample",self.files[0])
        self.transform = tfm

    def __len__(self):
        return len(self.files)

    def __getitem__(self,idx):
        fname = self.files[idx]
        im = Image.open(fname)
        im = self.transform(im)
        try:
            label = int(fname.split("/")[-1].split("_")[0])
        except:
            label = -1 # test has no label
        return im,label

valid_set = FoodDataset(os.path.join(cfg['dataset_root'], "validation"), tfm=test_tfm)
valid_loader = DataLoader(valid_set, batch_size=cfg['batch_size'], shuffle=False, num_workers=8, pin_memory=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
log(f"device: {device}")

x = []
y = []
for ratio_iter in range(0, 100, 5):
    # teacher_model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', weights=None, num_classes=11)
    # teacher_ckpt_path = "resnet50_2_0.001_test.ckpt"
    # teacher_model.load_state_dict(torch.load(teacher_ckpt_path, map_location='cpu')['model_state_dict'])
    teacher_model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=False, num_classes=11)
    teacher_ckpt_path = os.path.join(cfg['dataset_root'], "resnet18_teacher.ckpt")
    teacher_model.load_state_dict(torch.load(teacher_ckpt_path, map_location='cpu'))
    teacher_model.to(device)
    teacher_model.eval()
    
    ratio = ratio_iter / 100 # specify the pruning ratio
    for name, module in teacher_model.named_modules():
        if isinstance(module, torch.nn.Conv2d): # if the nn.module is torch.nn.Conv2d
            prune.l1_unstructured(module, name='weight', amount=ratio) # use 'prune' method provided by 'torch.nn.utils.prune' to 
            
    # These are used to record information in validation.
    valid_loss = []
    valid_accs = []
    valid_lens = []

    # Iterate the validation set by batches.
    for batch in tqdm(valid_loader):

        # A batch consists of image data and corresponding labels.
        imgs, labels = batch
        imgs = imgs.to(device)
        labels = labels.to(device)

        # We don't need gradient in validation.
        # Using torch.no_grad() accelerates the forward process.
        with torch.no_grad():
            logits = teacher_model(imgs)

        # Compute the accuracy for current batch.
        acc = (logits.argmax(dim=-1) == labels).float().sum()

        # Record the loss and accuracy.
        batch_len = len(imgs)
        valid_accs.append(acc)
        valid_lens.append(batch_len)

    # The average loss and accuracy for entire validation set is the average of the recorded values.
    valid_acc = sum(valid_accs) / sum(valid_lens)

    # update logs
    log(f"ratio = {ratio}, acc = {valid_acc:.5f}")
    x.append(ratio)
    y.append(valid_acc.item())

plt.plot(x, y, color='red', linestyle="-", linewidth="2", markersize="16", marker=".")
plt.xlabel('Pruning Ratio')
plt.ylabel('Model Accuracy')
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.savefig("pruning.png")