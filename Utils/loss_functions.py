import time

import torch
import torch.nn.functional as F
import numpy as np
import torch.nn as nn


class Loss_alpha(nn.Module):
    def __init__(self, args):
        super(Loss_alpha, self).__init__()

        self.log_sftp = torch.nn.LogSoftmax().cuda()
        self.loss_ = torch.nn.NLLLoss().cuda()

    def forward(self, alpha,labels):
        #Loss functions
        S = torch.sum(alpha, dim=1, keepdim=True)
        prob = torch.nn.functional.softmax(alpha)
        loss = self.loss_(self.log_sftp(alpha), labels) + prob * (1 - prob) / (S + 1.0)
        loss = torch.mean(loss, dim=1)
        return torch.mean(loss)

def _jensen_shannon_reg(logit1, logit2, T=1.):
    """
    :param logit1:
    :param logit2: output of last classifier based on SoftPlus activation function
    :param T: Corresponding to \tau in paper
    :return:
    """
    prob1 = F.softmax(logit1/T, dim=1)
    prob2 = F.softmax(logit2/T, dim=1)
    mean_prob = 0.5 * (prob1 + prob2)
    logsoftmax = torch.log(mean_prob.clamp(min=1e-8))
    jsd = F.kl_div(logsoftmax, prob1, reduction='batchmean')
    jsd += F.kl_div(logsoftmax, prob2, reduction='batchmean')
    return jsd * 0.5


