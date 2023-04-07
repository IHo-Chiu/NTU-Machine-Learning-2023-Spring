import torch
import torch.nn as nn
import torch.nn.functional as F

# https://github.com/Leethony/Additive-Margin-Softmax-Loss-Pytorch/blob/master/AdMSLoss.py
class AdMSoftmaxLoss(nn.Module):

    def __init__(self, s=30.0, m=0.4):
        super(AdMSoftmaxLoss, self).__init__()
        self.s = s
        self.m = m

    def forward(self, wf, labels):
        numerator = self.s * (torch.diagonal(wf.transpose(0, 1)[labels]) - self.m)
        excl = torch.cat([torch.cat((wf[i, :y], wf[i, y+1:])).unsqueeze(0) for i, y in enumerate(labels)], dim=0)
        denominator = torch.exp(numerator) + torch.sum(torch.exp(self.s * excl), dim=1)
        L = numerator - torch.log(denominator)
        return -torch.mean(L)
