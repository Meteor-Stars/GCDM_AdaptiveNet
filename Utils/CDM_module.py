import time

import torch
import torch.nn.functional as F
import numpy as np

def Uncertainty_aware_Fusion( alpha,classes,balance_term=False):
    """
    Uncertainty-aware Fusion (UF) to realize Collaborative Decision Making (CDM).

    """
    def Sub_UF(alpha1, alpha2):
        alpha = dict()
        alpha[0], alpha[1] = alpha1, alpha2
        b, S, E, u = dict(), dict(), dict(), dict()
        for v in range(2):
            S[v] = torch.sum(alpha[v], dim=1, keepdim=True)
            E[v] = alpha[v] - 1
            b[v] = E[v] / (S[v].expand(E[v].shape))
            u[v] = classes / S[v]
        bu = torch.mul(b[0], 1-u[0])
        ub = torch.mul(b[1], 1 - u[1])

        if balance_term:
            b_a = ((b[0] * 0.5 + b[1] * 0.5 + b[0] * b[1]) / 2 + bu + ub)
            u_a = u[0] + u[1] + u[0] * u[1]
        else:
            #Fusion without balance term
            b_a = b[0] * b[1] + bu + ub
            u_a = u[0] * u[1]

        S_a = classes / u_a
        e_a = torch.mul(b_a, S_a.expand(b_a.shape))
        alpha_a = e_a + 1
        return alpha_a
    alpha_fuse = alpha[0]
    for v in range(len(alpha) - 1):
        if v == 0:
            alpha_fuse = Sub_UF(alpha[0], alpha[1])
        else:
            alpha_fuse = Sub_UF(alpha_fuse, alpha[v + 1])
    return alpha_fuse